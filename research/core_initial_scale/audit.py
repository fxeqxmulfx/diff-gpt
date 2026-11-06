"""Measure actual initialized residual branches without any scientific updates."""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from research.current_grokking.data import build, tensor_digest
from research.early99.experiments.factories import baseline_model, peri_rms_scaled_model
from research.early99.protocol import selected, specification
from research.early99_core_ablations.factories import core_model
from research.early99_core_ablations.readiness import verify_training_ready
from research.early99_reporting.round2_audit import require, verify_hash

if TYPE_CHECKING:
    from research.current_grokking.interfaces import Task
    from research.current_grokking.train import Logits

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"


def digest(path: Path) -> str:
    """Pin diagnostic sources and fixed experiment plans."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensors(model: torch.nn.Module) -> dict:
    """Include names, order and every parameter/buffer value in the fingerprint."""
    return {
        kind: [(name, tensor_digest(value)) for name, value in members]
        for kind, members in (
            ("parameters", model.named_parameters()),
            ("buffers", model.named_buffers()),
        )
    }


def rms(value: torch.Tensor) -> float:
    """Reduce actual FP32 observations in FP64, including all zero positions."""
    require(torch.isfinite(value).all().item(), "Nonfinite observed activation")
    return value.double().square().mean().sqrt().item()


def branch_metrics(branch: torch.Tensor, skip: torch.Tensor) -> dict:
    """Report global and per-position branch-to-skip magnitude ratios."""
    skip_rms, branch_rms = rms(skip), rms(branch)
    denominators = skip.double().square().mean(-1).sqrt()
    numerators = branch.double().square().mean(-1).sqrt()
    require((denominators > 0).all().item(), "Undefined zero-skip ratio")
    ratios = (numerators / denominators).flatten().tolist()
    return {
        "branch_rms": branch_rms,
        "skip_rms": skip_rms,
        "global_rms_ratio": branch_rms / skip_rms,
        "position_ratio_min": min(ratios),
        "position_ratio_median": statistics.median(ratios),
        "position_ratio_max": max(ratios),
        "zero_branch_positions": int((numerators == 0).sum()),
        "positions": len(ratios),
    }


@torch.inference_mode()
def observe(model: Logits, task: Task, batch: torch.Tensor) -> dict:
    """Validate hooks and both actual guarded residual additions exactly."""
    model.eval()
    before, rng = tensors(model), torch.get_rng_state().clone()
    expected_logits, expected_targets = task.forward(model, batch)
    saved = {}

    def save(name: str, value: torch.Tensor) -> None:
        require(name not in saved, f"Repeated observed module: {name}")
        saved[name] = value.detach().clone()

    def block_input(_module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        save("skip", args[0])
        require(kwargs.get("constant") is not None, "Expected actual prefix guard")
        save("initial_constant", kwargs["constant"])

    def attention_output(_module: torch.nn.Module, _args: tuple, output: tuple) -> None:
        require(isinstance(output, tuple), "Expected attention output and guard")
        save("raw_attention", output[0])
        save("head_guard", output[1])

    def feedforward_input(_module: torch.nn.Module, args: tuple) -> None:
        save("after_attention", args[0])

    def feedforward_output(
        _module: torch.nn.Module, _args: tuple, output: torch.Tensor
    ) -> None:
        save("raw_feedforward", output)

    def block_output(_module: torch.nn.Module, _args: tuple, output: tuple) -> None:
        require(isinstance(output, tuple), "Expected block output and guard")
        save("output", output[0])
        save("constant", output[1])

    def embedding_input(_module: torch.nn.Module, args: tuple) -> None:
        save("tokens", args[0])

    block = model.model.blocks[0]
    hooks = [
        block.register_forward_pre_hook(block_input, with_kwargs=True),
        block.sa.register_forward_hook(attention_output),
        block.ln_2.register_forward_pre_hook(feedforward_input),
        block.ffwd.register_forward_hook(feedforward_output),
        block.register_forward_hook(block_output),
        model.model.token_embedding_table.register_forward_pre_hook(embedding_input),
    ]
    try:
        actual_logits, actual_targets = task.forward(model, batch)
    finally:
        for hook in hooks:
            hook.remove()
    require(torch.equal(actual_logits, expected_logits), "Hooks changed model logits")
    require(torch.equal(actual_targets, expected_targets), "Task targets changed")
    require(torch.isfinite(actual_logits).all().item(), "Nonfinite actual logits")
    require(before == tensors(model), "Diagnostic mutated model tensors")
    require(torch.equal(torch.get_rng_state(), rng), "Diagnostic consumed RNG")
    require(
        torch.equal(
            saved["constant"],
            saved["initial_constant"] & saved["head_guard"][:, None],
        ),
        "Observed prefix sharing differs from the actual guard",
    )
    applied = {
        name: torch.where(
            saved["constant"][..., None],
            saved[f"raw_{name}"][:, :1],
            saved[f"raw_{name}"],
        )
        for name in ("attention", "feedforward")
    }
    require(
        torch.equal(saved["skip"] + applied["attention"], saved["after_attention"]),
        "Observed attention contribution does not reconstruct the actual addition",
    )
    require(
        torch.equal(saved["after_attention"] + applied["feedforward"], saved["output"]),
        "Observed feed-forward contribution does not reconstruct the actual addition",
    )
    return {
        "input_tokens_sha256": tensor_digest(saved["tokens"]),
        "token_shape": list(saved["tokens"].shape),
        "embedding_rms": rms(saved["skip"]),
        "attention": branch_metrics(applied["attention"], saved["skip"]),
        "feedforward": branch_metrics(applied["feedforward"], saved["after_attention"]),
        "output_rms": rms(saved["output"]),
        "guarded_prefix_positions": int(saved["constant"].sum()),
        "actual_logits_sha256": tensor_digest(actual_logits),
        "observed_activations_sha256": {
            name: tensor_digest(value) for name, value in {**saved, **applied}.items()
        },
        "exact_hook_and_residual_checks_passed": True,
        "unchanged_model_and_rng": True,
    }


def main() -> None:
    """Evaluate all declared initial states and retain every measurement."""
    verify_training_ready()
    plan = json.loads((STUDY / "plan.json").read_text())
    for path, expected in plan["source_sha256"].items():
        verify_hash(ROOT / path, expected)
    lookup = {row["name"]: row for row in selected("short")}
    rows = []
    for name in plan["cases"]:
        row = lookup[name]
        case = specification(row)
        task = build(case, "cpu")
        require(task.fingerprints == row["data_fingerprints"], "Changed task data")
        count = min(plan["examples_per_case"], len(task.splits["train"]))
        batch = task.inputs(torch.arange(count))
        for seed in plan["seeds"]:
            reference, original_rng, token_hash = None, None, None
            for arm in plan["arms"]:
                torch.manual_seed(seed)
                if arm == "original":
                    model = baseline_model(task, case, "cpu")
                elif arm == "both":
                    model = peri_rms_scaled_model(task, case, "cpu")
                else:
                    model = core_model(task, case, "cpu", mechanism=arm)
                state = tensors(model)
                if reference is None:
                    reference, original_rng = state, torch.get_rng_state().clone()
                require(
                    state == reference, "Initial parameters, order or buffers differ"
                )
                require(torch.equal(torch.get_rng_state(), original_rng), "RNG differs")
                require(
                    model.model.token_embedding_table.weight
                    is model.model.lm_head.weight,
                    "Embedding/readout tying changed",
                )
                observation = observe(model, task, batch)
                if token_hash is None:
                    token_hash = observation["input_tokens_sha256"]
                require(
                    observation["input_tokens_sha256"] == token_hash, "Inputs differ"
                )
                record = {
                    "case": name,
                    "seed": seed,
                    "arm": arm,
                    "examples": count,
                    "initial_state_sha256": hashlib.sha256(
                        json.dumps(state).encode()
                    ).hexdigest(),
                    **observation,
                }
                rows.append(record)
                print(json.dumps(record), flush=True)
    expected = len(plan["cases"]) * len(plan["seeds"]) * len(plan["arms"])
    require(len(rows) == expected, "Incomplete diagnostic grid")
    summaries = []
    for name in plan["cases"]:
        for arm in plan["arms"]:
            group = [r for r in rows if r["case"] == name and r["arm"] == arm]
            summaries.append(
                {
                    "case": name,
                    "arm": arm,
                    "median_embedding_rms": statistics.median(
                        r["embedding_rms"] for r in group
                    ),
                    **{
                        f"median_{branch}_to_skip_rms_ratio": statistics.median(
                            r[branch]["global_rms_ratio"] for r in group
                        )
                        for branch in ("attention", "feedforward")
                    },
                }
            )
    result = {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "measurements": len(rows),
        "optimizer_updates": 0,
        "heldout_scores_computed": False,
        "selection_uses_this_diagnostic": False,
        "rows": rows,
        "task_medians": summaries,
        "limitations": plan["limitations"],
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (Path(__file__), STUDY / "plan.json")
        },
        "torch": str(torch.__version__),
        "command": (
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python "
            "-m research.core_initial_scale.audit"
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "initial-scales.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
