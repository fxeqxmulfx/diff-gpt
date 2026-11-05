"""Independent archive/checkpoint audit of the frozen transformer stress fixes."""

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


def main() -> None:
    """Verify all learning checkpoints and numerical records without retraining."""
    torch.set_num_threads(1)
    output = ROOT / "runs/transformer-stress/fixes-v1"
    manifest = json.loads((output / "manifest.json").read_text())
    plan = manifest["parent_plan"]
    settings = plan["residual_learning"]
    for name, expected in manifest["source_hashes"].items():
        for path in (ROOT / name, output / "sources" / name):
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"Source mismatch: {path}",
            )
    splits = torch.load(output / "data.pt", weights_only=True)
    flattened = {
        "rotation": splits["rotation"],
        **{
            f"{split}_{key}": value
            for split in ("train", "heldout")
            for key, value in splits[split].items()
        },
    }
    data_hash = digest(flattened)
    require(
        data_hash == json.loads((output / "data.json").read_text())["sha256"],
        "Data digest mismatch",
    )
    audited, paired = [], {}
    for depth, strength, seed, kind in itertools.product(
        settings["branches"],
        settings["teacher_strengths"],
        settings["seeds"],
        settings["variants"],
    ):
        directory = output / "learning" / f"{kind}-d{depth}-a{strength}-seed{seed}"
        result = json.loads((directory / "result.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        checkpoint = torch.load(directory / "checkpoint.pt", weights_only=False)
        require(
            checkpoint["settings"] == settings, "Checkpoint settings differ from plan"
        )
        require(checkpoint["result"] == result, "Checkpoint/result mismatch")
        require(
            result["completed_updates"] == settings["updates"],
            "Incomplete scientific cell",
        )
        require(
            [row["updates"] for row in history]
            == list(range(0, settings["updates"] + 1, settings["observe_updates"])),
            "Missing or changed observation cadence",
        )
        crossing = next(
            (
                row["updates"]
                for row in history
                if row["heldout"]["relative_mse"] <= settings["target_relative_mse"]
            ),
            None,
        )
        require(
            crossing == result["first_threshold_updates"],
            "Incorrect first threshold crossing",
        )
        require(
            result["status"] == ("passed" if crossing is not None else "budget_miss"),
            "Status disagrees with full history",
        )
        torch.manual_seed(seed)
        model = ShiftedResidualNetwork(settings, depth, kind)
        initial_hash = digest(dict(model.branches.named_parameters()))
        require(
            initial_hash == result["common_initialization_sha256"],
            "Initial weights changed",
        )
        paired.setdefault((depth, seed), set()).add(initial_hash)
        model.load_state_dict(checkpoint["model"], strict=True)
        require(
            all(finite(p) for p in model.parameters()), "Nonfinite checkpoint weights"
        )
        require(
            sum(p.numel() for p in model.parameters()) == result["parameters"],
            "Parameter count mismatch",
        )
        for split in ("train", "heldout"):
            actual = evaluate(model, splits[split], strength)
            require(
                math.isclose(
                    actual["relative_mse"],
                    history[-1][split]["relative_mse"],
                    rel_tol=1e-12,
                ),
                "Checkpoint does not reproduce final split evaluation",
            )
        for state in checkpoint["optimizer"]["state"].values():
            require(
                state["step"].item() == settings["updates"],
                "Optimizer update count mismatch",
            )
            require(
                all(
                    finite(value)
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                ),
                "Nonfinite optimizer state",
            )
        for group in checkpoint["optimizer"]["param_groups"]:
            require(
                group["lr"] == settings["learning_rate"]
                and group["weight_decay"] == settings["weight_decay"]
                and group["betas"] == tuple(settings["betas"]),
                "Optimizer settings mismatch",
            )
        rng = torch.Generator().manual_seed(seed)
        batches = settings["train_examples"] // settings["batch"]
        for _ in range(math.ceil(settings["updates"] / batches)):
            torch.randperm(settings["train_examples"], generator=rng)
        require(
            torch.equal(rng.get_state(), checkpoint["rng_state"]),
            "Sampler state disagrees with declared seed/budget",
        )
        audited.append(result)
    require(
        all(len(hashes) == 1 for hashes in paired.values()),
        "Paired initialization mismatch",
    )
    rows = json.loads((output / "numerical.json").read_text())["rows"]
    expected_variants = ["control", *manifest["plan"]["numerical_variants"]]
    expected_keys = set(
        itertools.product(
            [
                tuple(shape)
                for shape in plan["components"]["full_stack_shapes_layers_context"]
            ],
            [tuple(case) for case in plan["components"]["full_stack_interventions"]],
        )
    )
    numerical_summary = []
    tolerance = plan["components"]["diagnostic_relative_error_threshold"]
    for variant in expected_variants:
        cells = [row for row in rows if row["variant"] == variant]
        keys = {
            ((row["layers"], row["context"]), (row["intervention"], row["magnitude"]))
            for row in cells
        }
        require(
            keys == expected_keys and len(cells) == len(expected_keys),
            "Missing/duplicated numerical case",
        )
        require(
            all(
                row["all_parameters_fp32"]
                and row["tied_parameter_identity"]
                and row["reference_finite"]
                for row in cells
            ),
            "Precision, tie or reference mismatch",
        )
        controls = [row for row in cells if row["intervention"] == "control"]
        require(
            all(
                row["logits_finite"]
                and row["gradients_finite"]
                and row["logits_relative_error"] < tolerance
                for row in controls
            ),
            "Ordinary control failed",
        )
        good = [
            row
            for row in cells
            if row["logits_finite"]
            and row["gradients_finite"]
            and row["logits_relative_error"] < tolerance
        ]
        numerical_summary.append(
            {"variant": variant, "cases": len(cells), "finite_and_accurate": len(good)}
        )
    require(
        numerical_summary[-1]["finite_and_accurate"] == len(expected_keys),
        "Bounded combined fix has unresolved numerical grid failures",
    )
    audit = {
        "audited_learning_cells": len(audited),
        "passed": sum(row["status"] == "passed" for row in audited),
        "budget_misses": sum(row["status"] == "budget_miss" for row in audited),
        "numerical_learning_failures": 0,
        "numerical_probe_rows": len(rows),
        "data_sha256": data_hash,
        "source_files_verified": len(manifest["source_hashes"]),
        "all_checkpoints_histories_optimizer_steps_and_sampler_states_verified": True,
        "paired_initializations_and_final_full_split_metrics_verified": True,
        "ordinary_controls_verified_per_variant": len(controls),
        "numerical_summary": numerical_summary,
        "numerical_stress_failures_preserved": True,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
