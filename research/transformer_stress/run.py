"""Freeze, execute and report independent transformer stress experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path

import torch

from research.transformer_stress.common import (
    ROOT,
    digest,
    require,
    source_manifest,
    write_json,
)
from research.transformer_stress.components import run_components
from research.transformer_stress.residual import (
    checks,
    datasets,
    structural,
    train_cell,
)

PLAN = Path(__file__).with_name("plan.json")


def prepare(output: Path, plan: dict) -> dict:
    """Archive sources and data; refuse a changed manifest on a repeated command."""
    manifest = source_manifest()
    path = output / "manifest.json"
    if path.exists():
        require(
            json.loads(path.read_text())["source_hashes"] == manifest,
            "Stress sources changed; use a separate experiment version",
        )
    else:
        output.mkdir(parents=True, exist_ok=True)
        for source in manifest:
            destination = output / "sources" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, destination)
        write_json(
            path,
            {
                "source_hashes": manifest,
                "plan": plan,
                "torch": torch.__version__,
                "device": "cpu",
                "threads": torch.get_num_threads(),
            },
        )
    splits = datasets(plan["residual_learning"])
    flattened = {
        "rotation": splits["rotation"],
        **{
            f"{split}_{name}": value
            for split in ("train", "heldout")
            for name, value in splits[split].items()
        },
    }
    fingerprint = digest(flattened)
    data_path = output / "data.json"
    if data_path.exists():
        require(
            json.loads(data_path.read_text())["sha256"] == fingerprint,
            "Data fingerprint changed",
        )
    else:
        write_json(data_path, {"sha256": fingerprint})
        torch.save(splits, output / "data.pt")
    write_json(output / "preflight.json", checks(plan["residual_learning"], splits))
    return splits


def report(output: Path, plan: dict) -> None:
    """Export every expected cell and rank only under the frozen rule."""
    rows = [
        json.loads(path.read_text())
        for path in sorted((output / "learning").glob("*/result.json"))
    ]
    settings = plan["residual_learning"]
    expected_per_variant = (
        len(settings["branches"])
        * len(settings["teacher_strengths"])
        * len(settings["seeds"])
    )
    variants = []
    for kind in settings["variants"]:
        cells = [row for row in rows if row["variant"] == kind]
        passed = [row for row in cells if row["status"] == "passed"]
        errors = [
            row["final_heldout_relative_mse"]
            if row["final_heldout_relative_mse"] is not None
            else math.inf
            for row in cells
        ]
        variants.append(
            {
                "variant": kind,
                "expected": expected_per_variant,
                "completed": len(cells),
                "passed": len(passed),
                "numerical_failures": sum(
                    row["status"] == "numerical_failure" for row in cells
                ),
                "geometric_mean_first_updates": math.exp(
                    statistics.mean(
                        math.log(row["first_threshold_updates"]) for row in cells
                    )
                )
                if len(passed) == expected_per_variant
                else None,
                "worst_final_relative_mse": max(errors) if errors else None,
                "median_final_relative_mse": statistics.median(errors)
                if errors
                else None,
            }
        )
    complete = len(rows) == expected_per_variant * len(settings["variants"])
    winner = None
    if complete:
        ordered = sorted(
            variants,
            key=lambda row: (
                -row["passed"],
                row["geometric_mean_first_updates"]
                if row["passed"] == expected_per_variant
                else row["worst_final_relative_mse"],
                row["median_final_relative_mse"],
            ),
        )
        winner = ordered[0]["variant"]
    write_json(
        output / "learning-summary.json",
        {
            "complete": complete,
            "expected_cells": expected_per_variant * len(settings["variants"]),
            "completed_cells": len(rows),
            "best_tested_residual_variant": winner,
            "scope": plan["scope"],
            "ranking": settings["ranking"],
            "variants": variants,
            "cells": rows,
        },
    )
    if rows:
        with (output / "learning-cells.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    """Run CPU stress separately while the frozen GPU benchmark continues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/transformer-stress/v1"
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--learning", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    torch.set_num_threads(plan["threads"])
    torch.set_float32_matmul_precision("highest")
    if args.check_only:
        print(
            json.dumps(
                checks(plan["residual_learning"], datasets(plan["residual_learning"]))
            )
        )
        return
    splits = prepare(args.output, plan)
    if args.components:
        require(
            not (args.output / "components.json").exists(),
            "Component results already exist; do not overwrite",
        )
        rows = run_components(plan["components"])
        rows.extend(
            structural(plan["components"], plan["residual_learning"]["variants"])
        )
        write_json(
            args.output / "components.json", {"scope": plan["scope"], "rows": rows}
        )
        print(json.dumps({"components_completed": len(rows)}), flush=True)
    if args.learning:
        settings = plan["residual_learning"]
        for depth in settings["branches"]:
            for strength in settings["teacher_strengths"]:
                for seed in settings["seeds"]:
                    for kind in settings["variants"]:
                        directory = (
                            args.output
                            / "learning"
                            / f"{kind}-d{depth}-a{strength}-seed{seed}"
                        )
                        if (directory / "result.json").exists():
                            continue
                        row = train_cell(
                            settings, splits, kind, depth, strength, seed, directory
                        )
                        print(
                            json.dumps(
                                {
                                    key: row[key]
                                    for key in (
                                        "variant",
                                        "branches",
                                        "strength",
                                        "seed",
                                        "status",
                                        "first_threshold_updates",
                                        "final_heldout_relative_mse",
                                        "seconds",
                                    )
                                }
                            ),
                            flush=True,
                        )
                        report(args.output, plan)
    if args.report:
        report(args.output, plan)


if __name__ == "__main__":
    main()
