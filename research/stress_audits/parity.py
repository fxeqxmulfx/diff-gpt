"""Audit every paired activation/teacher cell and the preserved original controls."""

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
from research.transformer_stress_parity.models import ConfiguredNetwork
from research.transformer_stress_parity.run import data_digest, mixed_splits, summarize


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
    model = ConfiguredNetwork(settings, result["branches"], result["variant"])
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
    """Keep every teacher, seed, first crossing, late regression and budget miss."""
    torch.set_num_threads(1)
    output = ROOT / "runs/transformer-stress/parity-v1"
    manifest = json.loads((output / "manifest.json").read_text())
    plan, original = manifest["plan"], manifest["parent_settings"]
    for name, expected in manifest["source_hashes"].items():
        for path in (ROOT / name, output / "sources" / name):
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"Source changed: {path}",
            )
    control_path = ROOT / "runs/transformer-stress/rates-v1/summary.json"
    require(
        hashlib.sha256(control_path.read_bytes()).hexdigest()
        == manifest["control_summary_sha256"],
        "Control rows changed",
    )
    require(
        hashlib.sha256((output / "data.pt").read_bytes()).hexdigest()
        == hashlib.sha256(
            (ROOT / "runs/transformer-stress/v1/data.pt").read_bytes()
        ).hexdigest(),
        "Original data file changed",
    )
    data = {
        "original_odd": torch.load(output / "data.pt", weights_only=True),
        "mixed": torch.load(output / "mixed-data.pt", weights_only=True),
    }
    recorded_hashes = json.loads((output / "data.json").read_text())
    require(
        {name: data_digest(splits) for name, splits in data.items()} == recorded_hashes,
        "Data fingerprints differ",
    )
    require(
        data_digest(mixed_splits(data["original_odd"], plan["mixed_even_coefficient"]))
        == recorded_hashes["mixed"],
        "Mixed teacher formula differs",
    )
    controls = [
        {"teacher": "original_odd", "activation_variant": "shifted", **r}
        for r in json.loads(control_path.read_text())["cells"]
        if r["variant"] == plan["variant"]
        and r["learning_rate"] == plan["learning_rate"]
    ]
    rows = list(controls)
    for teacher, arm in itertools.product(
        plan["teachers"], plan["activation_variants"]
    ):
        if teacher == "original_odd" and arm == "shifted":
            continue
        settings = dict(
            original, learning_rate=plan["learning_rate"], activation_variant=arm
        )
        for depth, strength, seed in itertools.product(
            original["branches"], original["teacher_strengths"], original["seeds"]
        ):
            directory = output / teacher / arm / f"d{depth}-a{strength}-seed{seed}"
            result = audit_cell(directory, settings, data[teacher], plan)
            require(
                (result["branches"], result["strength"], result["seed"])
                == (depth, strength, seed),
                "Cell identity differs",
            )
            rows.append({"teacher": teacher, "activation_variant": arm, **result})
    require(len(rows) - len(controls) == plan["expected_new_cells"], "Missing cells")
    summary = json.loads((output / "summary.json").read_text())
    require(summary["complete"], "Incomplete comparison")
    require(summary == summarize(rows, plan, original), "Comparison rows differ")
    audit = {
        "source_files_verified": len(manifest["source_hashes"]),
        "new_cells": len(rows) - len(controls),
        "reused_control_cells": len(controls),
        "new_checkpoint_full_split_scores_reproduced": True,
        "original_data_unchanged_and_mixed_teacher_verified": True,
        "all_initial_matrices_capacity_optimizer_sampler_and_histories_verified": True,
        "all_comparison_rows_and_selection_recomputed": True,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
