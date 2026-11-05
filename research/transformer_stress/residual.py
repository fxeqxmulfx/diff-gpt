"""Paired residual learning and structural gradient counterexamples."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch
from torch import nn

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.rms_norm import rms_norm
from research.transformer_stress.common import (
    EPS,
    digest,
    finite,
    gradient_norm,
    require,
    rms,
    write_json,
)

if TYPE_CHECKING:
    from pathlib import Path


def merge_branch(
    raw: torch.Tensor, gate: torch.Tensor, kind: str, depth: int
) -> torch.Tensor:
    """Define each declared residual adaptation with explicit gate placement."""
    if kind == "pre_unit":
        return raw
    if kind == "peri_pre_gate":
        return rms_norm(raw * (2 * gate.sigmoid()), EPS) / math.sqrt(depth)
    normalized = rms_norm(raw, EPS)
    if kind == "peri_unit":
        return normalized
    if kind == "peri_depth":
        return normalized / depth
    if kind == "peri_post_gate":
        return normalized * (2 * gate.sigmoid()) / math.sqrt(depth)
    if kind == "peri_rezero":
        return normalized * gate / math.sqrt(depth)
    require(kind == "peri_sqrt", "Undeclared residual variant")
    return normalized / math.sqrt(depth)


class ResidualNetwork(nn.Module):
    """A residual-only SwiGLU learner with no attention or learned readout."""

    def __init__(self, settings: dict, depth: int, kind: str) -> None:
        """Draw identical matrices before appending deterministic scalar gates."""
        super().__init__()
        self.branches = nn.ModuleList(
            FeedForward(settings["dimension"]) for _ in range(depth)
        )
        for parameter in self.branches.parameters():
            nn.init.normal_(parameter, std=settings["initial_weight_std"])
        self.kind, self.depth = kind, depth
        if kind in {"peri_post_gate", "peri_pre_gate", "peri_rezero"}:
            self.gates = nn.Parameter(torch.zeros(depth))
        else:
            self.register_buffer("gates", torch.zeros(depth))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply pre-RMS in every arm and the declared branch output operation."""
        for branch, gate in zip(self.branches, self.gates, strict=True):
            value = value + merge_branch(
                branch(rms_norm(value, EPS)), gate, self.kind, self.depth
            )
        return value


def datasets(settings: dict) -> dict:
    """Freeze independent sphere inputs and a common orthogonal nonlinear teacher."""
    rng = torch.Generator().manual_seed(settings["data_seed"])
    rotation, _ = torch.linalg.qr(
        torch.randn(settings["dimension"], settings["dimension"], generator=rng)
    )
    result = {"rotation": rotation}
    for split in ("train", "heldout"):
        x = torch.randn(
            settings[f"{split}_examples"], settings["dimension"], generator=rng
        )
        x = x / x.square().mean(-1, keepdim=True).sqrt()
        result[split] = {"input": x, "residual": (x @ rotation).tanh()}
    return result


@torch.inference_mode()
def evaluate(model: ResidualNetwork, split: dict, strength: float) -> dict:
    """Score every fixed example; identity has relative MSE exactly one."""
    value = model(split["input"])
    target = split["input"] + strength * split["residual"]
    error = (value.double() - target.double()).square().mean()
    denominator = (target.double() - split["input"].double()).square().mean()
    return {
        "relative_mse": (error / denominator).item(),
        "mse": error.item(),
        "output_rms": rms(value),
        "maximum_row_rms": value.double().square().mean(-1).sqrt().max().item(),
        "finite": finite(value),
    }


def checks(settings: dict, splits: dict) -> dict:
    """Verify actual gate derivatives, common weights and a nonconstant teacher."""
    x = torch.tensor([1.0, -1.0], dtype=torch.float64)
    derivatives = {}
    for kind in ("peri_post_gate", "peri_pre_gate"):
        gate = torch.zeros((), dtype=torch.float64, requires_grad=True)
        scalar = (merge_branch(x, gate, kind, 4) * x).mean()
        derivatives[kind] = torch.autograd.grad(scalar, gate)[0].item()
    expected = EPS / (1 + EPS)
    actual = derivatives["peri_pre_gate"] / derivatives["peri_post_gate"]
    require(
        math.isclose(actual, expected, rel_tol=1e-10),
        "Gate derivative ratio disagrees with finite-epsilon equation",
    )
    fingerprints = set()
    forward = {}
    for kind in settings["variants"]:
        torch.manual_seed(7)
        model = ResidualNetwork(settings, settings["branches"][0], kind)
        fingerprints.add(digest(dict(model.branches.named_parameters())))
        forward[kind] = model(splits["train"]["input"][:8]).detach()
    require(len(fingerprints) == 1, "Common branch initialization differs")
    require(
        torch.equal(forward["peri_sqrt"], forward["peri_post_gate"]),
        "Post gate must preserve initial parent output",
    )
    require(
        torch.equal(forward["peri_sqrt"], forward["peri_pre_gate"]),
        "Pre gate must preserve initial parent output",
    )
    require(
        torch.equal(forward["peri_rezero"], splits["train"]["input"][:8]),
        "Zero gate must initially be identity",
    )
    require(rms(splits["heldout"]["residual"]) > 0, "Teacher must be nonconstant")
    return {
        "common_initialization": True,
        "post_pre_gate_derivative_ratio": actual,
        "expected_ratio": expected,
        "zero_gate_initial_identity": True,
    }


def train_cell(
    settings: dict,
    splits: dict,
    kind: str,
    depth: int,
    strength: float,
    seed: int,
    directory: Path,
) -> dict:
    """Run a full fixed budget and preserve first crossings and later regressions."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = ResidualNetwork(settings, depth, kind)
    initial = digest(dict(model.branches.named_parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        betas=tuple(settings["betas"]),
        eps=settings["optimizer_epsilon"],
        weight_decay=settings["weight_decay"],
    )
    rng = torch.Generator().manual_seed(seed)
    history, first = [], None
    maximum_gradient, nonfinite, completed = 0.0, False, 0

    def observe(step: int) -> None:
        nonlocal first
        row = {
            "updates": step,
            "epochs": step * settings["batch"] / settings["train_examples"],
            **{
                split: evaluate(model, splits[split], strength)
                for split in ("train", "heldout")
            },
        }
        if (
            first is None
            and row["heldout"]["relative_mse"] <= settings["target_relative_mse"]
        ):
            first = step
        history.append(row)
        write_json(directory / "history.json", history)

    observe(0)
    batches = settings["train_examples"] // settings["batch"]
    for step in range(settings["updates"]):
        if step % batches == 0:
            order = torch.randperm(settings["train_examples"], generator=rng)
        indices = order[
            (step % batches) * settings["batch"] : (step % batches + 1)
            * settings["batch"]
        ]
        x = splits["train"]["input"][indices]
        target = x + strength * splits["train"]["residual"][indices]
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        grad = gradient_norm(model)
        if not finite(loss) or not math.isfinite(grad):
            nonfinite = True
            break
        maximum_gradient = max(maximum_gradient, grad)
        for group in optimizer.param_groups:
            group["lr"] = settings["learning_rate"] * min(
                1, step / settings["warmup_updates"]
            )
        optimizer.step()
        completed = step + 1
        if completed % settings["observe_updates"] == 0:
            observe(completed)
    if history[-1]["updates"] != completed:
        observe(completed)
    singulars = []
    if not nonfinite:
        for row in splits["heldout"]["input"][:4]:
            jacobian = torch.autograd.functional.jacobian(model, row)
            if finite(jacobian):
                singulars.extend(torch.linalg.svdvals(jacobian.double()).tolist())
    result = {
        "variant": kind,
        "branches": depth,
        "strength": strength,
        "seed": seed,
        "status": "numerical_failure"
        if nonfinite
        else "passed"
        if first is not None
        else "budget_miss",
        "completed_updates": completed,
        "first_threshold_updates": first,
        "final_heldout_relative_mse": history[-1]["heldout"]["relative_mse"],
        "final_train_relative_mse": history[-1]["train"]["relative_mse"],
        "initial_heldout_relative_mse": history[0]["heldout"]["relative_mse"],
        "final_maximum_row_rms": history[-1]["heldout"]["maximum_row_rms"],
        "maximum_gradient_norm": maximum_gradient,
        "final_input_jacobian_min_singular": min(singulars) if singulars else None,
        "final_input_jacobian_max_singular": max(singulars) if singulars else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "common_initialization_sha256": initial,
        "final_gates": model.gates.detach().tolist(),
        "seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_updates": completed,
            "settings": settings,
            "result": result,
            "rng_state": rng.get_state(),
        },
        directory / "checkpoint.pt",
    )
    write_json(directory / "result.json", result)
    return result


def structural(settings: dict, variants: list[str]) -> list[dict]:
    """Stress identity skips and normalized branches with exact scalar controls."""
    records = []
    for depth in settings["residual_branches"]:
        for coefficient in settings["residual_linear_coefficients"]:
            x = torch.ones((), dtype=torch.float64, requires_grad=True)
            value = x
            for _ in range(depth):
                value = value + coefficient * value
            derivative = torch.autograd.grad(value, x)[0].item()
            records.append(
                {
                    "component": "linear_residual",
                    "branches": depth,
                    "coefficient": coefficient,
                    "derivative": derivative,
                    "analytic_derivative": (1 + coefficient) ** depth,
                    "depth_scaled_contraction": (1 - 1 / (2 * depth)) ** depth,
                }
            )
        for kind in variants:
            for sign in (-1, 1):

                def chain(
                    value: torch.Tensor,
                    *,
                    count: int = depth,
                    coefficient: int = sign,
                    variant: str = kind,
                ) -> torch.Tensor:
                    for _ in range(count):
                        value = value + merge_branch(
                            coefficient * value, value.new_zeros(()), variant, count
                        )
                    return value

                initial = torch.ones(8, dtype=torch.float64)
                result = chain(initial)
                jacobian = torch.autograd.functional.jacobian(chain, initial)
                singular = torch.linalg.svdvals(jacobian) if finite(jacobian) else None
                records.append(
                    {
                        "component": "normalized_residual",
                        "variant": kind,
                        "branches": depth,
                        "branch_sign": sign,
                        "output_rms": rms(result),
                        "finite": finite(result) and finite(jacobian),
                        "jacobian_min_singular": singular.min().item()
                        if singular is not None
                        else None,
                        "jacobian_max_singular": singular.max().item()
                        if singular is not None
                        else None,
                    }
                )
    return records
