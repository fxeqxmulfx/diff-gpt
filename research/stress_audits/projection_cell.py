"""Replay each projection checkpoint and verify unchanged parent capacity."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import torch

from research.transformer_stress.common import digest, finite, require
from research.transformer_stress.residual import evaluate
from research.transformer_stress_bias.models import BiasedNetwork, is_new_bias
from research.transformer_stress_projection.models import ProjectedNetwork

if TYPE_CHECKING:
    from pathlib import Path


def audit_cell(directory: Path, settings: dict, data: dict, plan: dict) -> dict:
    """Replay final full-split scores and verify the unchanged optimizer and sampler."""
    result = json.loads((directory / "result.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=False)
    require(checkpoint["settings"] == settings, "Undeclared settings change")
    require(checkpoint["result"] == result, "Checkpoint result differs")
    require(result["variant"] == plan["variant"], "Changed residual variant")
    completed = result["completed_updates"]
    failed = result["status"] == "numerical_failure"
    require(
        0 <= completed <= settings["updates"]
        and (failed or completed == settings["updates"]),
        "Incomplete original budget",
    )
    expected_steps = list(range(0, completed + 1, settings["observe_updates"]))
    if expected_steps[-1] != completed:
        expected_steps.append(completed)
    require([r["updates"] for r in history] == expected_steps, "Cadence differs")
    require(
        all(
            r["epochs"] == r["updates"] * settings["batch"] / settings["train_examples"]
            for r in history
        ),
        "Exposure epochs differ",
    )
    for split in ("train", "heldout"):
        require(
            result[f"final_{split}_relative_mse"] == history[-1][split]["relative_mse"],
            "Final result metric differs",
        )
    require(
        result["initial_heldout_relative_mse"] == history[0]["heldout"]["relative_mse"],
        "Initial result metric differs",
    )
    first = next(
        (
            r["updates"]
            for r in history
            if r["heldout"]["relative_mse"] is not None
            and r["heldout"]["relative_mse"] <= settings["target_relative_mse"]
        ),
        None,
    )
    require(first == result["first_threshold_updates"], "Crossing differs")
    require(
        failed
        or result["status"] == ("passed" if first is not None else "budget_miss"),
        "Status differs",
    )
    torch.manual_seed(result["seed"])
    model = ProjectedNetwork(settings, result["branches"], result["variant"])
    candidate_rng = torch.get_rng_state().clone()
    require(
        digest(model.common_branch_parameters())
        == result["common_initialization_sha256"],
        "Initial state differs",
    )
    torch.manual_seed(result["seed"])
    parent = BiasedNetwork(settings, result["branches"], result["variant"])
    require(torch.equal(candidate_rng, torch.get_rng_state()), "Initial RNG differs")
    shared = dict(model.named_parameters())
    require(
        digest(shared) == digest(dict(parent.named_parameters())), "Matrices differ"
    )
    extra = model.added_bias_parameters()
    branch = parent.branches[0]
    declared = result["branches"] * (
        branch.c_proj.out_features
        + (branch.c_fc.out_features if settings["bias_mode"] == "all" else 0)
    )
    require(
        extra == declared == result["added_bias_parameters"],
        "Wrong added bias capacity",
    )
    require(
        all(
            torch.count_nonzero(p) == 0
            for n, p in model.named_parameters()
            if is_new_bias(n)
        ),
        "New bias did not initialize at zero",
    )
    require(
        sum(p.numel() for p in model.parameters())
        == result["parameters"]
        == sum(p.numel() for p in parent.parameters()),
        "Capacity differs",
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    require(
        model.gates.detach().tolist() == result["final_gates"], "Final gates differ"
    )
    for split in ("train", "heldout"):
        actual = evaluate(model, data[split], result["strength"])
        recorded = history[-1][split]
        require(actual["finite"] == recorded["finite"], "Finiteness differs")
        if actual["finite"]:
            require(
                math.isclose(
                    actual["relative_mse"], recorded["relative_mse"], rel_tol=1e-12
                ),
                "Final full-split metric differs",
            )
    if not failed:
        require(all(finite(p) for p in model.parameters()), "Nonfinite weights")
    for state in checkpoint["optimizer"]["state"].values():
        require(state["step"].item() == completed, "Optimizer step differs")
        if not failed:
            require(
                all(finite(x) for x in state.values() if isinstance(x, torch.Tensor)),
                "Nonfinite optimizer state",
            )
    groups = checkpoint["optimizer"]["param_groups"]
    require(len(groups) == 1, "Changed parameter grouping")
    require(
        len(groups[0]["params"]) == len(list(model.parameters())),
        "Missing optimizer parameters",
    )
    require(
        set(checkpoint["optimizer"]["state"]) == set(groups[0]["params"]),
        "Missing optimizer state",
    )
    for group in groups:
        require(
            group["lr"]
            == settings["learning_rate"]
            * min(1, (completed - 1) / settings["warmup_updates"])
            and group["betas"] == tuple(settings["betas"])
            and group["weight_decay"] == settings["weight_decay"]
            and group["eps"] == settings["optimizer_epsilon"],
            "Optimizer configuration differs",
        )
    rng = torch.Generator().manual_seed(result["seed"])
    attempted = completed + int(failed and completed < settings["updates"])
    for _ in range(
        math.ceil(attempted / (settings["train_examples"] // settings["batch"]))
    ):
        torch.randperm(settings["train_examples"], generator=rng)
    require(torch.equal(rng.get_state(), checkpoint["rng_state"]), "Sampler differs")
    return result
