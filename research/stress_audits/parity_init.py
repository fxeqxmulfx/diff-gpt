"""Audit initialization-only cells and preserve both earlier activation controls."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from research.transformer_stress.common import ROOT, digest, finite, require, write_json
from research.transformer_stress.residual import evaluate
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork
from research.transformer_stress_parity.run import data_digest
from research.transformer_stress_parity_init.models import InitializedNetwork
from research.transformer_stress_parity_init.run import PARENT, control_rows, summarize


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
    model = InitializedNetwork(settings, result["branches"], result["variant"])
    require(
        digest(dict(model.branches.named_parameters()))
        == result["common_initialization_sha256"],
        "Initial state differs",
    )
    torch.manual_seed(result["seed"])
    parent = ShiftedResidualNetwork(settings, result["branches"], result["variant"])
    shared = {
        name: value
        for name, value in model.named_parameters()
        if not name.endswith("even_gain")
    }
    require(
        digest(shared) == digest(dict(parent.named_parameters())), "Matrices differ"
    )
    extra = (
        result["branches"] if settings["activation_variant"] == "learned_even" else 0
    )
    require(
        sum(p.numel() for p in model.parameters())
        == result["parameters"]
        == sum(p.numel() for p in parent.parameters()) + extra,
        "Capacity differs",
    )
    model.load_state_dict(checkpoint["model"], strict=True)
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
    for group in checkpoint["optimizer"]["param_groups"]:
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


def main() -> None:
    """Verify all 48 new checkpoints and the complete 96-row comparison."""
    torch.set_num_threads(1)
    output = ROOT / "runs/transformer-stress/parity-init-v1"
    manifest = json.loads((output / "manifest.json").read_text())
    plan, original = manifest["plan"], manifest["parent_settings"]
    for name, expected in manifest["source_hashes"].items():
        for path in (ROOT / name, output / "sources" / name):
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"Changed source: {path}",
            )
    for name, expected in manifest["parent_files_sha256"].items():
        require(
            hashlib.sha256((PARENT / name).read_bytes()).hexdigest() == expected,
            f"Changed parent file: {name}",
        )
        if name != "summary.json":
            require(
                hashlib.sha256((output / name).read_bytes()).hexdigest() == expected,
                f"Changed copied data: {name}",
            )
    data = {
        "original_odd": torch.load(output / "data.pt", weights_only=True),
        "mixed": torch.load(output / "mixed-data.pt", weights_only=True),
    }
    require(
        {name: data_digest(split) for name, split in data.items()}
        == json.loads((output / "data.json").read_text()),
        "Data digest mismatch",
    )
    rows = control_rows()
    require(len(rows) == plan["reused_control_cells"], "Missing control rows")
    for initial, teacher in itertools.product(
        plan["new_initial_values"], plan["teachers"]
    ):
        settings = dict(
            original,
            learning_rate=plan["learning_rate"],
            activation_variant=plan["activation_variant"],
            even_gain_initial=initial,
        )
        for depth, strength, seed in itertools.product(
            original["branches"], original["teacher_strengths"], original["seeds"]
        ):
            directory = (
                output
                / f"init{initial:g}"
                / teacher
                / f"d{depth}-a{strength}-seed{seed}"
            )
            result = audit_cell(directory, settings, data[teacher], plan)
            require(
                (result["branches"], result["strength"], result["seed"])
                == (depth, strength, seed),
                "Cell identity mismatch",
            )
            rows.append(
                {
                    "initialization_arm": f"learned{initial:g}",
                    "activation_variant": plan["activation_variant"],
                    "teacher": teacher,
                    **result,
                }
            )
    require(
        len(rows) == plan["expected_new_cells"] + plan["reused_control_cells"],
        "Missing scientific cells",
    )
    summary = json.loads((output / "summary.json").read_text())
    require(summary["complete"], "Incomplete comparison")
    require(summary == summarize(rows, plan, original), "Changed comparison")
    audit = {
        "source_files_verified": len(manifest["source_hashes"]),
        "new_cells": plan["expected_new_cells"],
        "reused_control_cells": plan["reused_control_cells"],
        "both_original_data_files_unchanged": True,
        "all_new_full_split_final_scores_reproduced": True,
        "initial_states_capacity_optimizer_sampler_and_histories_verified": True,
        "all_comparison_rows_and_selection_recomputed": True,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
