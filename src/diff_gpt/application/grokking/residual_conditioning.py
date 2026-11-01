"""
Frozen comparison of original and conditioned depth-query optimization.

Reuse the independent teacher, splits, query direction and Schedule-Free
recipe from residual_attention_probe.py at 386ae01. Thirty declared cases
include fresh scales and continuations of all three huge-query failures.
The correction preserves forward/state coordinates before subsequent updates;
it intentionally changes optimization geometry. No full GPT is replaced.
"""

import copy
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from diff_gpt.application.grokking import residual_attention_probe as original
from diff_gpt.application.grokking.residual_attention_replay import require
from diff_gpt.application.residual_conditioning import DepthQueryConditioner, stable_rms
from diff_gpt.domain.model.conditioned_attention_residual import (
    ConditionedQKAttentionResidual,
)
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from torch import Tensor

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "research/residual_conditioning/protocol.json"


def sha256(path: Path) -> str:
    """Identify a frozen source, protocol or checkpoint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_protocol() -> dict:
    """Refuse to run a changed implementation under a frozen protocol."""
    protocol = json.loads(PROTOCOL.read_text())
    for name, digest in protocol["sources"].items():
        require(sha256(ROOT / name) == digest, f"Changed source: {name}")
    require(protocol["width"] == original.WIDTH, "Teacher width mismatch")
    require(
        protocol["sources_per_example"] == original.SOURCES, "Source count mismatch"
    )
    require(protocol["eps"] == original.EPS, "RMS epsilon mismatch")
    return protocol


def make_model(case: dict, device: str) -> torch.nn.Module:
    """Use exactly the old query direction, including the zero controls."""
    legacy = original.Probe(original.ProbeCase("qk", case["scale"], case["seed"]))
    if case["mode"] == "legacy":
        return legacy.to(device)
    model = ConditionedQKAttentionResidual(original.WIDTH)
    with torch.no_grad():
        model.query.copy_(legacy.query)
    return model.to(device)


def observe(
    model: torch.nn.Module,
    optimizer: ScheduleFreeAdamW,
    splits: dict[str, tuple[Tensor, Tensor, Tensor]],
    step: int,
) -> dict:
    """Retain average metrics and both coordinate and physical query gradients."""
    row = original.observe(model, optimizer, splits, step)
    scale = (
        float(model.query_scale)
        if isinstance(model, ConditionedQKAttentionResidual)
        else 1.0
    )
    base = optimizer.state[model.query]["base"]
    beta = optimizer.param_groups[0]["betas"][0]
    with torch.no_grad():
        average = torch.lerp(model.query, base, 1 - 1 / beta)
        row.update(
            coordinate_scale=scale,
            physical_query_rms=float(stable_rms(model.query)) * scale,
            physical_query_gradient_l2=row["query_gradient_l2"] / scale,
            base_coordinate_rms=float(stable_rms(base)),
            average_coordinate_rms=float(stable_rms(average)),
        )
    return row


def save_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: ScheduleFreeAdamW,
    sampler: torch.Generator,
    step: int,
) -> None:
    """Save the chart, complete optimizer history and next minibatch draw."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.get_state(),
            "step": step,
        },
        path,
    )


def restore_source(
    case: dict,
    model: torch.nn.Module,
    optimizer: ScheduleFreeAdamW,
    sampler: torch.Generator,
) -> int:
    """Continue a failed original query without discarding any saved state."""
    if case["resume_source"] is None:
        return 0
    path = ROOT / case["resume_source"]
    require(sha256(path) == case["resume_sha256"], "Changed continuation source")
    state = torch.load(path, map_location="cuda", weights_only=True)
    with torch.no_grad():
        model.query.copy_(state["model"]["query"])
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    sampler.set_state(state["sampler"].cpu())
    torch.testing.assert_close(model.query, state["model"]["query"], rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state_dict(), state["optimizer"], rtol=0, atol=0
    )
    return state["step"]


def run_case(case: dict, output: Path, protocol: dict, deadline: float) -> dict:
    """Execute the whole declared interval, retaining an interrupted endpoint."""
    started = time.monotonic()
    torch.compiler.reset()
    torch.manual_seed(case["seed"])
    model = make_model(case, "cuda")
    optimizer = ScheduleFreeAdamW(model.parameters(), **protocol["optimizer"])
    sampler = torch.Generator().manual_seed(case["seed"] + 3000)
    source_step = restore_source(case, model, optimizer, sampler)
    splits = {
        name: tuple(
            x.cuda()
            for x in original.dataset(
                case["seed"] + offset, rows=protocol[f"{name}_rows"]
            )
        )
        for name, offset in (("train", 1000), ("heldout", 2000))
    }
    source_hashes = {
        name: hashlib.sha256(values.cpu().numpy().tobytes()).hexdigest()
        for name, (values, _, _) in splits.items()
    }
    path = output / case["label"]
    path.mkdir()
    save_state(path / "initial.pt", model, optimizer, sampler, source_step)
    before = observe(model, optimizer, splits, source_step)
    conditioner = (
        DepthQueryConditioner(
            model,
            optimizer,
            maximum_rms=protocol["maximum_coordinate_rms"],
            compiled=True,
        )
        if case["mode"] == "conditioned"
        else None
    )
    compiled = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
    events = torch.zeros((), device="cuda", dtype=torch.int64)
    optimizer.zero_grad(set_to_none=True)
    if conditioner is not None:
        torch.compiler.cudagraph_mark_step_begin()
        factor = conditioner()
        events.add_((factor > 1).to(events.dtype))
    save_state(path / "conditioned_initial.pt", model, optimizer, sampler, source_step)
    curve = [observe(model, optimizer, splits, source_step)]
    values, targets, _ = splits["train"]
    step = source_step
    for iteration in range(case["updates"]):
        if time.monotonic() >= deadline:
            break
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        if conditioner is not None:
            factor = conditioner()
            events.add_((factor > 1).to(events.dtype))
        indices = torch.randint(
            len(values), (protocol["batch"],), generator=sampler
        ).cuda()
        loss = torch.nn.functional.mse_loss(compiled(values[indices]), targets[indices])
        loss.backward()
        optimizer.step()
        step = source_step + iteration + 1
        if step % protocol["observe_every"] == 0:
            curve.append(observe(model, optimizer, splits, step))
    if curve[-1]["step"] != step:
        curve.append(observe(model, optimizer, splits, step))
    torch.cuda.synchronize()
    save_state(path / "checkpoint.pt", model, optimizer, sampler, step)
    final = curve[-1]["splits"]["heldout"]
    record = {
        "case": case,
        "before_conditioning": before,
        "curves": curve,
        "source_step": source_step,
        "checkpoint_step": step,
        "completed": step == source_step + case["updates"],
        "success": final["mse"] <= protocol["success"]["heldout_mse_at_most"]
        and final["routing_accuracy"]
        >= protocol["success"]["heldout_routing_at_least"],
        "conditioning_events": int(events),
        "input_sha256": source_hashes,
        "wall_seconds": time.monotonic() - started,
        "checkpoint_sha256": sha256(path / "checkpoint.pt"),
    }
    (path / "result.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    return record


def run(output: Path) -> dict:
    """Freeze the actual inputs once and execute every declared case in order."""
    require(platform.python_version().startswith("3.15."), "Requires Python3.15")
    require(torch.__version__.startswith("2.15."), "Requires Torch2.15")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    protocol = read_protocol()
    output.mkdir(parents=True, exist_ok=True)
    require(
        not (output / "protocol.json").exists(), "Use a new declared attempt directory"
    )
    (output / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    started = time.monotonic()
    report = {
        "protocol": protocol,
        "protocol_sha256": sha256(PROTOCOL),
        "started_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "results": [],
    }
    try:
        for case in protocol["cases"]:
            if time.monotonic() - started >= protocol["wall_budget_seconds"]:
                break
            row = run_case(
                case, output, protocol, started + protocol["wall_budget_seconds"]
            )
            report["results"].append(row)
            (output / "results.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "label": case["label"],
                        "completed": row["completed"],
                        "success": row["success"],
                        "heldout": row["curves"][-1]["splits"]["heldout"],
                        "conditioning_events": row["conditioning_events"],
                    }
                ),
                flush=True,
            )
            if not row["completed"]:
                break
    except BaseException as error:
        report["failure"] = {
            "case": case["label"],
            "kind": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        report.update(
            ended_utc=datetime.now(UTC).isoformat(),
            wall_seconds=time.monotonic() - started,
        )
        report["not_started"] = [
            c["label"]
            for c in protocol["cases"][
                len(report["results"]) + int("failure" in report) :
            ]
        ]
        (output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report
