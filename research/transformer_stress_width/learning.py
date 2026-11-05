"""Original training loop with only the declared expanded constructor."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch

from research.transformer_stress.common import digest, finite, gradient_norm, write_json
from research.transformer_stress.residual import evaluate
from research.transformer_stress_width.models import ExpandedNetwork

if TYPE_CHECKING:
    from pathlib import Path


def train_expanded_cell(
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
    model = ExpandedNetwork(settings, depth, kind)
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
