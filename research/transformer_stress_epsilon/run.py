"""Run the unchanged residual learner at separately declared branch epsilons."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

import torch

from research.transformer_stress.common import (
    ROOT,
    require,
    source_manifest,
    write_json,
)
from research.transformer_stress.run import PLAN as ORIGINAL_PLAN
from research.transformer_stress_epsilon.learning import train_epsilon_cell

PLAN = Path(__file__).with_name("plan.json")
OUTPUT = ROOT / "runs/transformer-stress/epsilon-v1"
CONTROL_SUMMARY = ROOT / "runs/transformer-stress/rates-v1/summary.json"


def summarize(rows: list[dict], plan: dict) -> dict:
    """Rank complete epsilons without excluding a single missed seed or condition."""
    summaries = []
    expected = plan["expected_new_cells"] // len(plan["new_branch_epsilons"])
    for epsilon in [plan["control_epsilon"], *plan["new_branch_epsilons"]]:
        cells = [row for row in rows if row["branch_epsilon"] == epsilon]
        passed = [row for row in cells if row["status"] == "passed"]
        complete = len(cells) == expected
        errors = [
            row["final_heldout_relative_mse"]
            if row["final_heldout_relative_mse"] is not None
            else math.inf
            for row in cells
        ]
        summaries.append(
            {
                "branch_epsilon": epsilon,
                "expected": expected,
                "completed": len(cells),
                "passed": len(passed),
                "geometric_mean_first_updates": math.exp(
                    statistics.mean(
                        math.log(row["first_threshold_updates"]) for row in cells
                    )
                )
                if complete and len(passed) == len(cells)
                else None,
                "worst_final_relative_mse": max(errors) if errors else None,
                "median_final_relative_mse": statistics.median(errors)
                if errors
                else None,
            }
        )
    complete = len(rows) == expected * (1 + len(plan["new_branch_epsilons"]))
    best = None
    if complete:
        best = min(
            summaries,
            key=lambda row: (
                -row["passed"],
                row["geometric_mean_first_updates"]
                if row["passed"] == row["expected"]
                else row["worst_final_relative_mse"],
                row["median_final_relative_mse"],
            ),
        )["branch_epsilon"]
    return {
        "complete": complete,
        "scope": plan["scope"],
        "best_branch_epsilon": best,
        "epsilons": summaries,
        "cells": rows,
    }


def main() -> None:
    """Freeze sources, preserve controls, and complete both fixed-budget arms."""
    plan = json.loads(PLAN.read_text())
    original = json.loads(ORIGINAL_PLAN.read_text())
    torch.set_num_threads(original["threads"])
    torch.set_float32_matmul_precision("highest")
    sources = source_manifest()
    paths = [
        *Path(__file__).parent.glob("*.py"),
        PLAN,
        *ROOT.joinpath("research/transformer_stress_fixes").glob("*.py"),
        ROOT / "research/transformer_stress_fixes/plan.json",
        ROOT / "research/transformer_stress_rates/plan.json",
        ROOT / "research/transformer_stress_rates/run.py",
    ]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    destination = OUTPUT / "manifest.json"
    if destination.exists():
        require(
            json.loads(destination.read_text())["source_hashes"] == sources,
            "Epsilon-study sources changed",
        )
        require(
            json.loads(destination.read_text())["control_summary_sha256"]
            == hashlib.sha256(CONTROL_SUMMARY.read_bytes()).hexdigest(),
            "Control summary changed",
        )
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name in sources:
            target = OUTPUT / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        write_json(
            destination,
            {
                "source_hashes": sources,
                "plan": plan,
                "parent_settings": original["residual_learning"],
                "torch": torch.__version__,
                "control_summary_sha256": hashlib.sha256(
                    CONTROL_SUMMARY.read_bytes()
                ).hexdigest(),
            },
        )
        for name in ("data.pt", "data.json"):
            shutil.copyfile(ROOT / "runs/transformer-stress/v1" / name, OUTPUT / name)
    splits = torch.load(OUTPUT / "data.pt", weights_only=True)
    controls = json.loads(CONTROL_SUMMARY.read_text())["cells"]
    rows = [
        {"branch_epsilon": plan["control_epsilon"], **row}
        for row in controls
        if row["variant"] == plan["variant"]
        and row["learning_rate"] == plan["learning_rate"]
    ]
    for epsilon in plan["new_branch_epsilons"]:
        settings = dict(
            original["residual_learning"],
            learning_rate=plan["learning_rate"],
            branch_epsilon=epsilon,
        )
        for depth in settings["branches"]:
            for strength in settings["teacher_strengths"]:
                for seed in settings["seeds"]:
                    directory = (
                        OUTPUT / f"eps{epsilon}" / f"d{depth}-a{strength}-seed{seed}"
                    )
                    if (directory / "result.json").exists():
                        result = json.loads((directory / "result.json").read_text())
                    else:
                        result = train_epsilon_cell(
                            settings,
                            splits,
                            plan["variant"],
                            depth,
                            strength,
                            seed,
                            directory,
                        )
                    rows.append({"branch_epsilon": epsilon, **result})
                    write_json(OUTPUT / "summary.json", summarize(rows, plan))
                    print(
                        json.dumps(
                            {
                                "branch_epsilon": epsilon,
                                **{
                                    key: result[key]
                                    for key in (
                                        "branches",
                                        "strength",
                                        "seed",
                                        "status",
                                        "first_threshold_updates",
                                        "final_heldout_relative_mse",
                                    )
                                },
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
