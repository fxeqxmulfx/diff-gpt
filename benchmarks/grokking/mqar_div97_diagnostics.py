"""
Read-only probes of both Schedule-Free weight states of mqar_div97 runs.

The fixed test sample never participates in backpropagation. Probes use no RNG
and do not change parameters, gradients, optimizer iterates or model mode.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from diff_gpt.model.rope import apply_rope

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from benchmarks.grokking.mqar_div97 import Task
    from diff_gpt.model.gpt import GPT
    from diff_gpt.model.mha import MultiHeadAttention
    from diff_gpt.optimizer.schedule_free import ScheduleFreeAdamW

EPS = 1e-12
TWO_OPERANDS = 2


@dataclass(frozen=True)
class DiagnosticConfig:
    """Log every `every` epochs and save both weight states every `snapshot_every`."""

    every: int = 1
    frames: int = 512
    snapshot_every: int = 10

    def __post_init__(self) -> None:
        """Require every >= 1, frames >= 1 and snapshot_every >= 0 (0 disables)."""
        if self.every < 1 or self.frames < 1 or self.snapshot_every < 0:
            msg = "require every >= 1, frames >= 1 and snapshot_every >= 0"
            raise ValueError(msg)

    @staticmethod
    def validate_task(task: Task) -> None:
        """Only one compute row with two recalled operands has these probe semantics."""
        if task.config.computed != 1 or task.config.operands != TWO_OPERANDS:
            msg = "extended diagnostics require one compute row and two operands"
            raise ValueError(msg)


def _append_json(path: Path, record: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def _summary(x: torch.Tensor) -> dict[str, float]:
    x = x.float()
    values = torch.stack((x.mean(), x.std(unbiased=False), x.min(), x.max()))
    return dict(zip(("mean", "std", "min", "max"), values.cpu().tolist(), strict=True))


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.square().mean(dim=-1).sqrt()


def _at(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Select one key per frame from a (B, H, T) tensor."""
    index = positions[:, None, None].expand(-1, x.shape[1], 1)
    return x.gather(-1, index).squeeze(-1)


def _head_stats(
    sa: MultiHeadAttention,
    normalized: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sources: torch.Tensor,
    writes: int,
    columns: int,
) -> tuple[list[dict[str, float]], torch.Tensor, torch.Tensor]:
    """Reconstruct attention and expose scale, direction, routing and XSA separately."""
    B, T, _ = normalized.shape
    q_raw, k_raw, v = (
        part.view(B, T, sa.n_head, sa.head_dim).transpose(1, 2)
        for part in sa.c_attn(normalized).split(sa.n_embd, dim=-1)
    )
    q, k = F.rms_norm(q_raw, (sa.head_dim,)), F.rms_norm(k_raw, (sa.head_dim,))
    if sa.use_rope:
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scale = 1 / math.sqrt(sa.head_dim)
    scores = (q @ k.transpose(-1, -2)) * scale
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    p = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    answer, answer_scores = p[:, :, -1], scores[:, :, -1]
    cosines = F.normalize(q[:, :, -1:], dim=-1) @ F.normalize(k, dim=-1).transpose(
        -1, -2
    )
    raw_cosines = F.normalize(q_raw[:, :, -1:], dim=-1) @ F.normalize(
        k_raw, dim=-1
    ).transpose(-1, -2)
    value_positions = sources * columns + columns - 1
    a_pos, b_pos = value_positions.unbind(-1)
    a, b = _at(answer, a_pos), _at(answer, b_pos)
    mixed = p @ v
    unit = F.normalize(v, dim=-1)
    removed = (mixed * unit).sum(-1, keepdim=True) * unit
    exclusive = mixed - removed
    exclusive[:, :, 0, :] = 0
    per_frame = {
        "attention_a": a,
        "attention_b": b,
        "attention_a_minus_b": a - b,
        "a_over_b_fraction": (a > b).float(),
        "argmax_a_fraction": (answer.argmax(-1) == a_pos[:, None]).float(),
        "argmax_b_fraction": (answer.argmax(-1) == b_pos[:, None]).float(),
        "answer_self_attention": answer[:, :, -1],
        "answer_reads_first_query": answer[:, :, -2],
        "answer_entropy": -(answer * answer.clamp_min(EPS).log()).sum(-1),
        "answer_logit_std": answer_scores.std(-1, unbiased=False),
        "answer_logit_range": answer_scores.max(-1).values
        - answer_scores.min(-1).values,
        "a_minus_b_logit": _at(answer_scores, a_pos) - _at(answer_scores, b_pos),
        "a_minus_b_cosine": _at(cosines.squeeze(-2), a_pos)
        - _at(cosines.squeeze(-2), b_pos),
        "a_minus_b_cosine_before_rope": _at(raw_cosines.squeeze(-2), a_pos)
        - _at(raw_cosines.squeeze(-2), b_pos),
        "answer_q_l2": q_raw[:, :, -1].norm(dim=-1),
        "key_a_l2": _at(k_raw.norm(dim=-1), a_pos),
        "key_b_l2": _at(k_raw.norm(dim=-1), b_pos),
        "mixed_value_l2": mixed[:, :, -1].norm(dim=-1),
        "xsa_output_l2": exclusive[:, :, -1].norm(dim=-1),
        "xsa_removed_energy_fraction": removed[:, :, -1].square().sum(-1)
        / mixed[:, :, -1].square().sum(-1).clamp_min(EPS),
    }
    load_keys = torch.arange(writes, device=q.device) * columns
    load_values = load_keys + columns - 1
    per_frame["value_reads_own_key"] = p[:, :, load_values, load_keys].mean(-1)
    per_frame["value_reads_previous_token"] = p[
        :, :, load_values, load_values - 1
    ].mean(-1)
    heads = [{} for _ in range(sa.n_head)]
    for name, values in per_frame.items():
        for head, stats in zip(heads, values.transpose(0, 1), strict=True):
            head.update({f"{name}_{key}": val for key, val in _summary(stats).items()})
    distinct = sources[:, 0] != sources[:, 1]
    for label, selected in (("same", ~distinct), ("distinct", distinct)):
        for head in heads:
            head[f"{label}_frames"] = int(selected.sum().item())
        if selected.any():
            for name in (
                "attention_a",
                "attention_b",
                "attention_a_minus_b",
                "a_over_b_fraction",
                "a_minus_b_cosine",
                "answer_reads_first_query",
            ):
                for head, stats in zip(
                    heads, per_frame[name][selected].transpose(0, 1), strict=True
                ):
                    head[f"{label}_{name}_mean"] = stats.mean().item()
    for name, vectors in (("q", q_raw), ("k", k_raw), ("v", v)):
        norms = vectors.norm(dim=-1)
        for h, head in enumerate(heads):
            head.update(
                {f"{name}_l2_{key}": val for key, val in _summary(norms[:, h]).items()}
            )
    for name, weight in zip(
        ("q", "k", "v"), sa.c_attn.weight.split(sa.n_embd), strict=True
    ):
        for head, norm in zip(
            heads,
            weight.reshape(sa.n_head, -1).norm(dim=-1).cpu().tolist(),
            strict=True,
        ):
            head[f"{name}_weight_l2"] = norm
    reconstructed = sa.c_proj(exclusive.transpose(1, 2).reshape(B, T, sa.n_embd))
    return heads, p.mean(0), reconstructed


class Diagnostics:
    """Record both weight states, training gradients, updates and paired snapshots."""

    def __init__(
        self,
        directory: Path,
        task: Task,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        config: DiagnosticConfig,
    ) -> None:
        """Fix the first test frames once, without drawing from any RNG."""
        config.validate_task(task)
        self.directory, self.config = directory, config
        count = min(config.frames, len(inputs))
        self.inputs, self.targets = inputs[:count], targets[:count]
        self.task_config = asdict(task.config)
        self.writes = task.config.writes
        self.data_protocol = task.config.data_protocol
        frames = task.evaluation["test"]
        self.columns = frames.raw.shape[-1]
        self.sources = torch.as_tensor(frames.sources[:count, 0], device=inputs.device)
        self.pending_snapshot: dict[str, Any] | None = None
        for name in ("diagnostics.jsonl", "steps.jsonl"):
            (directory / name).touch(exist_ok=False)
        metadata = {
            "schema_version": 2,
            "data_protocol": self.data_protocol,
            "config": asdict(config),
            "probe_split": "test",
            "probe_indices": list(range(count)),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "states": {
                "query": "y = (1-beta1)*z + beta1*x; gradient evaluation weights",
                "evaluation": "x; weighted average used for primary train/test scores",
            },
            "semantics": {
                "head_statistics": "mean/std/min/max over fixed probe frames, per head",
                "q_k_norms": "raw projections, before per-head QK-RMSNorm and RoPE",
                "a_minus_b_cosine": "unit Q/K after RoPE; direction without norm",
                "a_minus_b": "first recalled operand minus second, regardless of label",
                "same_distinct": (
                    "conditional routing by query-key equality; same-query a-b is "
                    "necessarily zero, not failed binding"
                ),
                "mean_attention": "per-head causal matrices averaged over probe frames",
                "steps": "every minibatch of sampled epochs; update = y_after-y_before",
                "gradients": "training gradients; c_attn split into Q/K/V",
                "snapshots": "both model states for analysis, without optimizer state",
                "probing": "no RNG, backpropagation, parameter or model-mode changes",
            },
        }
        (directory / "diagnostics-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        torch.save(
            {
                "inputs": self.inputs.cpu(),
                "targets": self.targets.cpu(),
                "sources": self.sources.cpu(),
                "raw": torch.as_tensor(frames.raw[:count]),
            },
            directory / "probe.pt",
        )
        if config.snapshot_every:
            (directory / "snapshots").mkdir()

    def sampled(self, epoch: int) -> bool:
        """Whether this epoch receives full forward and training diagnostics."""
        return epoch % self.config.every == 0

    def before_step(self, epoch: int, model: GPT) -> dict[str, torch.Tensor] | None:
        """Save the pre-step query weights only when a step is being measured."""
        if not self.sampled(epoch):
            return None
        return {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def after_step(
        self,
        epoch: int,
        model: GPT,
        optimizer: ScheduleFreeAdamW,
        before: dict[str, torch.Tensor] | None,
        loss: torch.Tensor,
    ) -> None:
        """Record gradient norms and actual optimizer updates without modifying them."""
        if before is None:
            return
        names, values = [], []
        for name, p in model.named_parameters():
            old, grad = before[name], p.grad
            parts = [(name, p, old, grad)]
            if name.endswith("c_attn.weight"):
                parts = [
                    (name.replace("c_attn", component), *tensors)
                    for component, tensors in zip(
                        ("q", "k", "v"),
                        zip(
                            p.split(model.n_embd),
                            old.split(model.n_embd),
                            grad.split(model.n_embd),
                            strict=True,
                        ),
                        strict=True,
                    )
                ]
            for part_name, current, previous, gradient in parts:
                names.append(part_name)
                weight_l2, grad_l2 = previous.norm(), gradient.norm()
                update_l2 = (current - previous).norm()
                values.append(
                    torch.stack(
                        (
                            weight_l2,
                            grad_l2,
                            update_l2,
                            update_l2 / weight_l2.clamp_min(EPS),
                        )
                    )
                )
        fields = ("weight_l2_before", "gradient_l2", "update_l2", "relative_update")
        record = {
            "epoch": epoch,
            "step": optimizer.param_groups[0]["step"],
            "loss": loss.item(),
            "parameters": {
                name: dict(zip(fields, row, strict=True))
                for name, row in zip(
                    names, torch.stack(values).cpu().tolist(), strict=True
                )
            },
        }
        _append_json(self.directory / "steps.jsonl", record)

    @torch.no_grad()
    def capture(
        self,
        epoch: int,
        state: str,
        model: GPT,
        optimizer: ScheduleFreeAdamW,
        metrics: dict[str, dict[str, float]],
    ) -> None:
        """Probe the scored weight state before the optimizer swaps it."""
        if state not in {"query", "evaluation"}:
            msg = f"unknown optimizer state {state}"
            raise ValueError(msg)
        if model.training:
            msg = "diagnostics require evaluation mode"
            raise ValueError(msg)
        if self.config.snapshot_every and epoch % self.config.snapshot_every == 0:
            weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if state == "query":
                self.pending_snapshot = {
                    "epoch": epoch,
                    "config": self.task_config,
                    "normalization": "peri_rms_qk_rms",
                    "data_protocol": self.data_protocol,
                    "query_model": weights,
                    "query_metrics": metrics,
                }
            else:
                if (
                    self.pending_snapshot is None
                    or self.pending_snapshot["epoch"] != epoch
                ):
                    msg = "evaluation snapshot needs the query state of the same epoch"
                    raise ValueError(msg)
                snapshot = self.pending_snapshot | {
                    "evaluation_model": weights,
                    "metrics": metrics,
                }
                target = self.directory / "snapshots" / f"epoch-{epoch:05d}.pt"
                temporary = target.with_suffix(".pt.tmp")
                torch.save(snapshot, temporary)
                temporary.replace(target)
                self.pending_snapshot = None
        if not self.sampled(epoch):
            return
        record = self.measure(model)
        record.update(epoch=epoch, state=state, metrics=metrics)
        distances = [
            torch.stack(
                (
                    (p - optimizer.state[p]["evaluation"]).square().sum(),
                    (optimizer.state[p]["base"] - optimizer.state[p]["evaluation"])
                    .square()
                    .sum(),
                )
            )
            for p in model.parameters()
        ]
        distance = torch.stack(distances).sum(0).sqrt().cpu().tolist()
        record["optimizer"] = {
            "step": optimizer.param_groups[0]["step"],
            "state_to_average_l2": distance[0],
            "base_to_average_l2": distance[1],
        }
        _append_json(self.directory / "diagnostics.jsonl", record)
        rows = [
            {"epoch": epoch, "state": state, "layer": layer["layer"], "head": h, **head}
            for layer in record["layers"]
            for h, head in enumerate(layer["heads"])
        ]
        csv_path = self.directory / "heads.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    @torch.no_grad()
    def measure(self, model: GPT) -> dict[str, Any]:
        """Inspect real block inputs/outputs, removing every hook even after failure."""
        captured: dict[str, tuple[tuple[torch.Tensor, ...], torch.Tensor]] = {}

        def hook(
            name: str,
        ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
            def capture(
                _module: nn.Module,
                args: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                captured[name] = (args, output)

            return capture

        hooks = [
            getattr(block, name).register_forward_hook(hook(f"{i}.{name}"))
            for i, block in enumerate(model.blocks)
            for name in ("ln_1", "ln_2", "sa", "attn_out_norm", "ffn_out_norm")
        ]
        try:
            logits, _ = model(self.inputs)
        finally:
            for handle in hooks:
                handle.remove()
        layers = []
        for i, block in enumerate(model.blocks):
            (normalized, cos, sin), raw_attn = captured[f"{i}.sa"]
            (residual,), _ = captured[f"{i}.ln_1"]
            (after_attention,), _ = captured[f"{i}.ln_2"]
            _, attn = captured[f"{i}.attn_out_norm"]
            _, ffwd = captured[f"{i}.ffn_out_norm"]
            heads, mean_attention, reconstructed = _head_stats(
                block.sa,
                normalized,
                cos,
                sin,
                self.sources,
                self.writes,
                self.columns,
            )
            streams = {
                "residual_rms": _rms(residual),
                "attention_output_rms": _rms(attn),
                "attention_to_residual_ratio": _rms(attn)
                / _rms(residual).clamp_min(EPS),
                "after_attention_rms": _rms(after_attention),
                "ffn_output_rms": _rms(ffwd),
                "ffn_to_residual_ratio": _rms(ffwd)
                / _rms(after_attention).clamp_min(EPS),
                "after_ffn_rms": _rms(after_attention + ffwd),
            }
            layers.append(
                {
                    "layer": i,
                    "heads": heads,
                    "streams": {name: _summary(x) for name, x in streams.items()},
                    "answer_streams": {
                        name: _summary(x[:, -1]) for name, x in streams.items()
                    },
                    "mean_attention": mean_attention.cpu().tolist(),
                    "attention_reconstruction_max_error": (reconstructed - raw_attn)
                    .abs()
                    .max()
                    .item(),
                }
            )
        answer = logits[:, -1]
        truth = self.targets[:, -1]
        predicted = answer.argmax(-1)
        wrong = predicted != truth
        values = self.inputs.gather(1, self.sources * self.columns + self.columns - 1)
        errors = {
            "accuracy": (~wrong).float().mean().item(),
            "wrong_is_a": (wrong & (predicted == values[:, 0])).float().mean().item(),
            "wrong_is_b": (wrong & (predicted == values[:, 1])).float().mean().item(),
            "true_probability": answer.softmax(-1)
            .gather(1, truth[:, None])
            .mean()
            .item(),
        }
        same = self.sources[:, 0] == self.sources[:, 1]
        for name, selected in (("same", same), ("distinct", ~same)):
            errors[f"{name}_frames"] = int(selected.sum().item())
            if selected.any():
                errors[f"{name}_accuracy"] = (~wrong[selected]).float().mean().item()
        tokens = self.inputs.unique()
        embeddings = model.token_embedding_table(tokens)
        return {
            "probe_frames": len(self.inputs),
            "probe_predictions": errors,
            "embedding_row_l2": _summary(embeddings.norm(dim=-1)),
            "embedding_common_direction_ratio": (
                embeddings.mean(0).norm()
                / embeddings.norm(dim=-1).mean().clamp_min(EPS)
            ).item(),
            "parameter_l2": {
                name: p.norm().item() for name, p in model.named_parameters()
            },
            "layers": layers,
        }
