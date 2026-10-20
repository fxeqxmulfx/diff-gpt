"""Inference interventions on mqar_div97 snapshots; never train or save weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.grokking.mqar_div97 import (
    EMPTY,
    Task,
    build,
    config_from_record,
    score,
    verify_checkpoint,
)
from diff_gpt.model.rope import apply_rope

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import nn

    from diff_gpt.model.gpt import GPT
    from diff_gpt.model.kv_cache import KVCache
    from diff_gpt.model.mha import MultiHeadAttention

type AuditValue = float | int | str | list[float] | None
WRAPPER_TOLERANCE = 1e-5


@contextmanager
def intervention(
    model: GPT, kind: str, layer: int | None = None, value: AuditValue = None
) -> Iterator[None]:
    """Restore every hook and attribute after one forward-only intervention."""
    handles = []
    restore = []
    try:
        block = model.blocks[layer] if layer is not None else None
        if kind in {"attention_gain", "ffn_gain"}:
            module = (
                block.attn_out_norm if kind == "attention_gain" else block.ffn_out_norm
            )
            handles.append(
                module.register_forward_hook(lambda _m, _a, out: out * value)
            )
        elif kind == "skip_norm":
            module = model.ln_f if layer is None else getattr(block, value)
            handles.append(module.register_forward_hook(lambda _m, args, _out: args[0]))
        elif kind == "zero_head":

            def zero_head(
                _module: nn.Module, args: tuple[torch.Tensor, ...]
            ) -> tuple[torch.Tensor]:
                x = args[0].clone()
                start = value * block.sa.head_dim
                x[..., start : start + block.sa.head_dim] = 0
                return (x,)

            handles.append(block.sa.c_proj.register_forward_pre_hook(zero_head))
        elif kind in {"xsa", "qknorm", "logit_gain"}:
            sa = block.sa
            restore.append((sa, "forward", sa.forward))

            def alternate_forward(
                self: MultiHeadAttention,
                x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                kv_cache: KVCache | None,
            ) -> torch.Tensor:
                if kv_cache is not None:
                    msg = "this audit only supports the uncached plain run"
                    raise ValueError(msg)
                batch, length, width = x.shape
                q, k, v = (
                    part.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
                    for part in self.c_attn(x).split(width, dim=-1)
                )
                scale = None
                q = F.rms_norm(q, (self.head_dim,))
                k = F.rms_norm(k, (self.head_dim,))
                if kind == "qknorm":
                    q = F.normalize(q, dim=-1)
                    k = F.normalize(k, dim=-1)
                    tau = torch.as_tensor(value, device=x.device).view(1, -1, 1, 1)
                    q = q * tau
                    scale = 1.0
                elif kind == "logit_gain":
                    scale = value / math.sqrt(self.head_dim)
                if self.use_rope:
                    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
                unit = F.normalize(v, dim=-1)
                coefficient = value if kind == "xsa" else 1.0
                y = y - coefficient * (y * unit).sum(-1, keepdim=True) * unit
                y[:, :, 0, :] = (1 - coefficient) * v[:, :, 0, :]
                return self.c_proj(
                    y.transpose(1, 2).contiguous().view(batch, length, width)
                )

            sa.forward = MethodType(alternate_forward, sa)
        elif kind != "baseline":
            raise ValueError(kind)
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, name, original in restore:
            setattr(module, name, original)


def variants(
    record: dict[str, Any],
    head_dim: int,
) -> Iterator[tuple[str, int | None, AuditValue]]:
    """Small controls separating magnitude, direction, branch mixing and projection."""
    yield "baseline", None, None
    yield "skip_norm", None, "ln_f"
    for layer in range(len(record["layers"])):
        for value in (0.0, 0.25, 0.5, 2.0):
            yield "attention_gain", layer, value
        yield "ffn_gain", layer, 0.0
        for name in ("ln_1", "ln_2"):
            yield "skip_norm", layer, name
        for value in (0.0, 0.5):
            yield "xsa", layer, value
        for value in (0.5, 2.0, 4.0):
            yield "logit_gain", layer, value
        heads = record["layers"][layer]["heads"]
        # Per head, the answer-to-value logit scale of the raw projections,
        # which the diagnostics measure before QK-RMSNorm.
        tau = [
            h["answer_q_l2_mean"]
            * (h["key_a_l2_mean"] + h["key_b_l2_mean"])
            / (2 * math.sqrt(head_dim))
            for h in heads
        ]
        yield "qknorm", layer, tau
        for head in range(len(heads)):
            yield "zero_head", layer, head


def input_variants(
    task: Task, inputs: torch.Tensor, targets: torch.Tensor
) -> Iterator[tuple[str, str, torch.Tensor, torch.Tensor]]:
    """Label-preserving layout checks and explicitly marked OOD sensitivity checks."""
    yield "original", "in_distribution", inputs, targets
    swapped = inputs.clone()
    swapped[:, :3], swapped[:, 3:6] = inputs[:, 3:6], inputs[:, :3]
    yield "swap_load_rows", "in_distribution", swapped, targets
    empty = int(task.codec.encode(np.array(EMPTY)))
    for position in (0, 3, 6, 7):
        masked = inputs.clone()
        masked[:, position] = empty
        yield f"erase_key_at_{position}", "ood_sensitivity", masked, targets
    for position in (0, 3, 6, 7):
        changed = inputs.clone()
        # A key from the training alphabet absent from the original two loads.
        for raw_key in range(-1, -task.config.keys - 1, -1):
            key = int(task.codec.encode(np.array(raw_key)))
            eligible = (inputs[:, 0] != key) & (inputs[:, 3] != key)
            pending = changed[:, position] == inputs[:, position]
            changed[eligible & pending, position] = key
        yield f"unused_key_at_{position}", "ood_sensitivity", changed, targets
    # Only the independent copying controls train on same-key queries. The
    # first key remains unchanged, so a dividend label is still correct.
    same = inputs.clone()
    same[:, 7] = inputs[:, 6]
    yield "both_queries_equal_first", "ood_semantic", same, targets
    # Rename all occurrences of each key with a fixed alphabet permutation.
    mapping = torch.arange(len(task.codec.levels), device=inputs.device)
    keys = torch.as_tensor(
        task.codec.encode(-np.arange(1, task.config.keys + 1)), device=inputs.device
    )
    mapping[keys] = keys.roll(7)
    yield "rename_all_keys", "in_distribution", mapping[inputs], targets


def score_inputs(
    task: Task, model: GPT, inputs: torch.Tensor, targets: torch.Tensor, device: str
) -> list[dict[str, Any]]:
    """Score the input variants, then the same weights on frames with more loads."""
    rows = []
    for name, category, x, y in input_variants(task, inputs, targets):
        metrics, _ = score(model, x, y)
        rows.append({"name": name, "category": category, **metrics})
    for loads in (3, 4):
        larger_config = replace(task.config, loads=loads)
        larger_task = Task(larger_config)
        larger_model = build(larger_config, device).eval()
        larger_model.load_state_dict(model.state_dict(), strict=True)
        metrics, _ = score(
            larger_model,
            *larger_task.tensors(larger_task.evaluation["test"], device),
        )
        rows.append(
            {
                "name": f"loads_{loads}",
                "category": "ood_length_and_distractors",
                **metrics,
            }
        )
    return rows


@torch.inference_mode()
def audit(directory: Path, device: str) -> None:
    """Score interventions without changing the training run or its checkpoints."""
    started = time.monotonic()
    output = directory / "transition-audit"
    output.mkdir(exist_ok=True)
    records = {}
    for line in (directory / "diagnostics.jsonl").open():
        record = json.loads(line)
        records[record["epoch"], record["state"]] = record
    final = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    config = config_from_record(final)
    task = Task(config)
    data = {
        split: task.tensors(frames, device) for split, frames in task.evaluation.items()
    }
    model = build(config, device).eval()
    states = []
    for epoch, state in (
        (250, "query"),
        (270, "query"),
        (280, "query"),
        (280, "evaluation"),
    ):
        saved = torch.load(
            directory / "snapshots" / f"epoch-{epoch:05d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        states.append((epoch, state, saved[f"{state}_model"]))
    states.append((final["epoch"], "evaluation", final["model"]))
    results, input_results, checks = [], [], []
    for epoch, state, weights in states:
        model.load_state_dict(weights)
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}
        probe = data["test"][0][:32]
        original_logits = model(probe)[0]
        for layer in range(config.layers):
            with intervention(model, "xsa", layer, 1.0):
                error = (model(probe)[0] - original_logits).abs().max().item()
            if error >= WRAPPER_TOLERANCE:
                msg = f"the XSA wrapper changed the logits: {error}"
                raise ValueError(msg)
            checks.append(
                {"epoch": epoch, "state": state, "layer": layer, "wrapper_error": error}
            )
        for kind, layer, value in variants(
            records[epoch, state], config.width // config.heads
        ):
            row = {
                "epoch": epoch,
                "state": state,
                "kind": kind,
                "layer": layer,
                "value": value,
            }
            with intervention(model, kind, layer, value):
                for split, (inputs, targets) in data.items():
                    metrics, _ = score(model, inputs, targets)
                    row[f"{split}_accuracy"] = metrics["accuracy"]
                    row[f"{split}_loss"] = metrics["loss"]
            results.append(row)
        if (epoch in (250, 280) and state == "query") or epoch == final["epoch"]:
            input_results.extend(
                {"epoch": epoch, "state": state, **measured}
                for measured in score_inputs(task, model, *data["test"], device)
            )
        torch.testing.assert_close(model.state_dict(), original_weights, atol=0, rtol=0)
        torch.testing.assert_close(model(probe)[0], original_logits, atol=0, rtol=0)
        print(
            json.dumps(
                {
                    "finished": [epoch, state],
                    "seconds": time.monotonic() - started,
                    "baseline": next(
                        r
                        for r in results
                        if r["epoch"] == epoch
                        and r["state"] == state
                        and r["kind"] == "baseline"
                    ),
                }
            ),
            flush=True,
        )
    metadata = {
        "source_run": str(directory.resolve()),
        "config": final["config"],
        "device": device,
        "training_steps": 0,
        "checkpoint_parameters_unchanged": True,
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "checks": checks,
        "seconds": time.monotonic() - started,
        "caveats": [
            "Post-hoc interventions cannot establish convergence speed from scratch.",
            "Input corruption, equal queries and larger loads are out of distribution.",
            "QK-norm temperature is fixed per head, calibrated on the saved probe.",
        ],
    }
    (output / "results.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "interventions": results,
                "inputs": input_results,
            },
            indent=2,
        )
        + "\n"
    )
    for name, rows in (("interventions", results), ("inputs", input_results)):
        with (output / f"{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps({"output": str(output), "seconds": metadata["seconds"]}), flush=True
    )


@torch.inference_mode()
def audit_inputs(directory: Path, device: str) -> None:
    """Verify the checkpoint and test input dependence, without any training."""
    verified = verify_checkpoint(directory, device)
    saved = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    config = config_from_record(saved)
    task = Task(config)
    model = build(config, device).eval()
    model.load_state_dict(saved["model"], strict=True)
    rows = score_inputs(
        task, model, *task.tensors(task.evaluation["test"], device), device
    )
    result = {"verification": verified, "training_steps": 0, "inputs": rows}
    (directory / "input-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    """Run the analysis with the same arithmetic precision as the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.inputs_only:
        audit_inputs(args.directory, args.device)
    else:
        audit(args.directory, args.device)


if __name__ == "__main__":
    main()
