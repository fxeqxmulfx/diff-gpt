"""Verify complete rate and capacity follow-ups against original stress artifacts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from research.transformer_stress.common import ROOT, digest, finite, require, write_json
from research.transformer_stress.residual import evaluate
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork
from research.transformer_stress_width.models import ExpandedNetwork


def main() -> None:
    """Reproduce all final metrics and verify original budgets and sampler states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["rates", "width"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = ROOT / f"runs/transformer-stress/{args.study}-v1"
    manifest = json.loads((output / "manifest.json").read_text())
    plan, original = manifest["plan"], manifest["parent_settings"]
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
    require(
        digest(flattened) == json.loads((output / "data.json").read_text())["sha256"],
        "Data digest mismatch",
    )
    if args.study == "rates":
        values = plan["new_learning_rates"]
        constructor = ShiftedResidualNetwork
    else:
        values = plan["new_hidden_widths"]
        constructor = ExpandedNetwork
    rows, merged_new = [], []
    for value, depth, strength, seed in itertools.product(
        values, original["branches"], original["teacher_strengths"], original["seeds"]
    ):
        group = f"lr{value}" if args.study == "rates" else f"hidden{value}"
        directory = output / group / f"d{depth}-a{strength}-seed{seed}"
        result = json.loads((directory / "result.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        checkpoint = torch.load(directory / "checkpoint.pt", weights_only=False)
        settings = dict(original)
        settings["learning_rate"] = (
            value if args.study == "rates" else plan["learning_rate"]
        )
        if args.study == "width":
            settings["expanded_hidden"] = value
        require(checkpoint["settings"] == settings, "Undeclared settings change")
        require(checkpoint["result"] == result, "Checkpoint/result disagreement")
        require(result["completed_updates"] == settings["updates"], "Budget mismatch")
        require(
            [row["updates"] for row in history]
            == list(range(0, settings["updates"] + 1, settings["observe_updates"])),
            "Changed evaluation cadence",
        )
        first = next(
            (
                row["updates"]
                for row in history
                if row["heldout"]["relative_mse"] <= settings["target_relative_mse"]
            ),
            None,
        )
        require(first == result["first_threshold_updates"], "First crossing mismatch")
        require(
            result["status"] == ("passed" if first is not None else "budget_miss"),
            "Incorrect status",
        )
        torch.manual_seed(seed)
        model = constructor(settings, depth, plan["variant"])
        require(
            digest(dict(model.branches.named_parameters()))
            == result["common_initialization_sha256"],
            "Initial state mismatch",
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        require(all(finite(p) for p in model.parameters()), "Nonfinite weights")
        require(
            sum(p.numel() for p in model.parameters()) == result["parameters"],
            "Hidden capacity change",
        )
        for split in ("train", "heldout"):
            observed = evaluate(model, splits[split], strength)["relative_mse"]
            require(
                math.isclose(
                    observed, history[-1][split]["relative_mse"], rel_tol=1e-12
                ),
                "Final split metric mismatch",
            )
        for state in checkpoint["optimizer"]["state"].values():
            require(
                state["step"].item() == settings["updates"], "Optimizer steps changed"
            )
            require(
                all(
                    finite(item)
                    for item in state.values()
                    if isinstance(item, torch.Tensor)
                ),
                "Nonfinite optimizer state",
            )
        for group in checkpoint["optimizer"]["param_groups"]:
            require(
                group["lr"] == settings["learning_rate"]
                and group["betas"] == tuple(settings["betas"])
                and group["weight_decay"] == settings["weight_decay"],
                "Optimizer configuration mismatch",
            )
        rng = torch.Generator().manual_seed(seed)
        for _ in range(
            math.ceil(
                settings["updates"] / (settings["train_examples"] // settings["batch"])
            )
        ):
            torch.randperm(settings["train_examples"], generator=rng)
        require(
            torch.equal(rng.get_state(), checkpoint["rng_state"]), "Sampler mismatch"
        )
        rows.append(result)
        tag = "learning_rate" if args.study == "rates" else "hidden_width"
        merged_new.append({tag: value, **result})
    require(len(rows) == plan["expected_new_cells"], "Missing new cases")
    summary = json.loads((output / "summary.json").read_text())
    require(summary["complete"], "Incomplete comparison")
    if args.study == "rates":
        control_path = ROOT / "runs/transformer-stress/fixes-v1/learning-summary.json"
        controls = [
            {"learning_rate": plan["control_learning_rate"], **row}
            for row in json.loads(control_path.read_text())["cells"]
            if row["variant"] == plan["variant"]
        ]
    else:
        control_path = ROOT / "runs/transformer-stress/rates-v1/summary.json"
        require(
            hashlib.sha256(control_path.read_bytes()).hexdigest()
            == manifest["control_summary_sha256"],
            "Control summary hash changed",
        )
        controls = [
            {"hidden_width": plan["control_hidden"], **row}
            for row in json.loads(control_path.read_text())["cells"]
            if row["variant"] == plan["variant"]
            and row["learning_rate"] == plan["learning_rate"]
        ]
    require(summary["cells"] == controls + merged_new, "Comparison rows changed")
    audit = {
        "study": args.study,
        "new_cells": len(rows),
        "reused_control_cells": len(summary["cells"]) - len(rows),
        "passed_new_cells": sum(row["status"] == "passed" for row in rows),
        "budget_misses": sum(row["status"] == "budget_miss" for row in rows),
        "source_files_verified": len(manifest["source_hashes"]),
        "all_sources_data_training_states_and_comparison_rows_verified": True,
        "final_full_split_metrics_reproduced": True,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
