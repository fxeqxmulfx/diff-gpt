"""Run paired even-coefficient initialization comparisons on both frozen teachers."""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from pathlib import Path

import torch

from research.transformer_stress.common import ROOT, require, write_json
from research.transformer_stress_parity.run import metrics
from research.transformer_stress_parity_init.learning import train_initialized_cell

PLAN = Path(__file__).with_name("plan.json")
OUTPUT = ROOT / "runs/transformer-stress/parity-init-v1"
PARENT = ROOT / "runs/transformer-stress/parity-v1"


def control_rows() -> list[dict]:
    """Keep all earlier rows, including the three mixed-teacher regressions."""
    return [
        {
            "initialization_arm": "shifted"
            if row["activation_variant"] == "shifted"
            else "learned0",
            **row,
        }
        for row in json.loads((PARENT / "summary.json").read_text())["cells"]
    ]


def summarize(rows: list[dict], plan: dict, original: dict) -> dict:
    """Compare four complete arms without using teacher-specific initialization."""
    per_teacher = (
        len(original["branches"])
        * len(original["teacher_strengths"])
        * len(original["seeds"])
    )
    expected = per_teacher * len(plan["teachers"])
    names = ["shifted", "learned0"] + [
        f"learned{value:g}" for value in plan["new_initial_values"]
    ]
    arms = []
    for name in names:
        cells = [row for row in rows if row["initialization_arm"] == name]
        arms.append(
            {
                "initialization_arm": name,
                **metrics(cells, expected),
                "teachers": {
                    teacher: metrics(
                        [r for r in cells if r["teacher"] == teacher], per_teacher
                    )
                    for teacher in plan["teachers"]
                },
            }
        )
    complete = len(rows) == expected * len(names)
    best = None
    if complete:
        best = min(
            arms,
            key=lambda row: (
                -row["passed"],
                row["geometric_mean_first_updates"]
                if row["passed"] == row["expected"]
                else row["worst_final_relative_mse"],
                row["median_final_relative_mse"],
            ),
        )["initialization_arm"]
    return {
        "complete": complete,
        "best_initialization_arm": best,
        "scope": plan["scope"],
        "arms": arms,
        "cells": rows,
    }


def main() -> None:
    """Pin every inherited source and data file before the 48 new training cells."""
    plan = json.loads(PLAN.read_text())
    parent = json.loads((PARENT / "manifest.json").read_text())
    original = parent["parent_settings"]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    sources = dict(parent["source_hashes"])
    for name, expected in sources.items():
        require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
            f"Changed inherited source: {name}",
        )
    for path in [*Path(__file__).parent.glob("*.py"), PLAN]:
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    parent_files = {
        name: hashlib.sha256((PARENT / name).read_bytes()).hexdigest()
        for name in ("summary.json", "data.pt", "mixed-data.pt", "data.json")
    }
    destination = OUTPUT / "manifest.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        require(saved["source_hashes"] == sources, "Initialization sources changed")
        require(saved["parent_files_sha256"] == parent_files, "Parent files changed")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name in sources:
            target = OUTPUT / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        write_json(
            destination,
            {
                "plan": plan,
                "parent_settings": original,
                "source_hashes": sources,
                "parent_files_sha256": parent_files,
                "torch": torch.__version__,
            },
        )
        for name in ("data.pt", "mixed-data.pt", "data.json"):
            shutil.copyfile(PARENT / name, OUTPUT / name)
    data = {
        "original_odd": torch.load(OUTPUT / "data.pt", weights_only=True),
        "mixed": torch.load(OUTPUT / "mixed-data.pt", weights_only=True),
    }
    rows = control_rows()
    require(len(rows) == plan["reused_control_cells"], "Missing inherited controls")
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
                OUTPUT
                / f"init{initial:g}"
                / teacher
                / f"d{depth}-a{strength}-seed{seed}"
            )
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text())
            else:
                result = train_initialized_cell(
                    settings,
                    data[teacher],
                    plan["variant"],
                    depth,
                    strength,
                    seed,
                    directory,
                )
            rows.append(
                {
                    "initialization_arm": f"learned{initial:g}",
                    "activation_variant": plan["activation_variant"],
                    "teacher": teacher,
                    **result,
                }
            )
            write_json(OUTPUT / "summary.json", summarize(rows, plan, original))
            print(
                json.dumps(
                    {
                        "initial": initial,
                        "teacher": teacher,
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
