"""Compare both activation arms on unchanged odd and new mixed-parity teachers."""

from __future__ import annotations

import hashlib
import itertools
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
from research.transformer_stress.run import PLAN as ORIGINAL_PLAN
from research.transformer_stress_parity.learning import train_parity_cell

PLAN = Path(__file__).with_name("plan.json")
OUTPUT = ROOT / "runs/transformer-stress/parity-v1"
CONTROL_SUMMARY = ROOT / "runs/transformer-stress/rates-v1/summary.json"


def mixed_splits(original: dict, coefficient: float) -> dict:
    """Retain inputs and rotation; change only the newly declared teacher response."""
    result = {"rotation": original["rotation"]}
    for split in ("train", "heldout"):
        x = original[split]["input"]
        z = x @ original["rotation"]
        result[split] = {
            "input": x,
            "residual": original[split]["residual"] + coefficient * (z.square() - 1),
        }
    return result


def data_digest(splits: dict) -> str:
    """Fingerprint all training and scoring inputs and labels, plus the teacher."""
    return digest(
        {
            "rotation": splits["rotation"],
            **{
                f"{split}_{name}": value
                for split in ("train", "heldout")
                for name, value in splits[split].items()
            },
        }
    )


def metrics(rows: list[dict], expected: int) -> dict:
    """Keep misses and nonfinite errors in the declared coverage-first ranking."""
    passed = sum(row["status"] == "passed" for row in rows)
    errors = [
        row["final_heldout_relative_mse"]
        if row["final_heldout_relative_mse"] is not None
        else math.inf
        for row in rows
    ]
    return {
        "expected": expected,
        "completed": len(rows),
        "passed": passed,
        "geometric_mean_first_updates": math.exp(
            statistics.mean(math.log(row["first_threshold_updates"]) for row in rows)
        )
        if len(rows) == passed == expected
        else None,
        "worst_final_relative_mse": max(errors) if errors else None,
        "median_final_relative_mse": statistics.median(errors) if errors else None,
    }


def summarize(rows: list[dict], plan: dict, original: dict) -> dict:
    """Select only after both teachers and both arms have complete paired results."""
    per_teacher = (
        len(original["branches"])
        * len(original["teacher_strengths"])
        * len(original["seeds"])
    )
    expected = per_teacher * len(plan["teachers"])
    arms = []
    for arm in plan["activation_variants"]:
        cells = [row for row in rows if row["activation_variant"] == arm]
        arms.append(
            {
                "activation_variant": arm,
                **metrics(cells, expected),
                "teachers": {
                    teacher: metrics(
                        [r for r in cells if r["teacher"] == teacher], per_teacher
                    )
                    for teacher in plan["teachers"]
                },
            }
        )
    complete = len(rows) == expected * len(plan["activation_variants"])
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
        )["activation_variant"]
    return {
        "complete": complete,
        "best_activation_variant": best,
        "scope": plan["scope"],
        "arms": arms,
        "cells": rows,
    }


def main() -> None:
    """Freeze sources and all data before executing the 36 new learning cells."""
    plan = json.loads(PLAN.read_text())
    parent = json.loads(ORIGINAL_PLAN.read_text())
    original = parent["residual_learning"]
    torch.set_num_threads(parent["threads"])
    torch.set_float32_matmul_precision("highest")
    sources = source_manifest()
    for path in [
        *Path(__file__).parent.glob("*.py"),
        PLAN,
        *ROOT.joinpath("research/transformer_stress_fixes").glob("*.py"),
        ROOT / "research/transformer_stress_fixes/plan.json",
        ROOT / "research/transformer_stress_rates/run.py",
        ROOT / "research/transformer_stress_rates/plan.json",
    ]:
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    control_hash = hashlib.sha256(CONTROL_SUMMARY.read_bytes()).hexdigest()
    destination = OUTPUT / "manifest.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        require(saved["source_hashes"] == sources, "Parity sources changed")
        require(saved["control_summary_sha256"] == control_hash, "Controls changed")
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
                "parent_settings": original,
                "torch": torch.__version__,
                "control_summary_sha256": control_hash,
            },
        )
        shutil.copyfile(ROOT / "runs/transformer-stress/v1/data.pt", OUTPUT / "data.pt")
        old = torch.load(OUTPUT / "data.pt", weights_only=True)
        mixed = mixed_splits(old, plan["mixed_even_coefficient"])
        torch.save(mixed, OUTPUT / "mixed-data.pt")
        write_json(
            OUTPUT / "data.json",
            {"original_odd": data_digest(old), "mixed": data_digest(mixed)},
        )
    data = {
        "original_odd": torch.load(OUTPUT / "data.pt", weights_only=True),
        "mixed": torch.load(OUTPUT / "mixed-data.pt", weights_only=True),
    }
    controls = json.loads(CONTROL_SUMMARY.read_text())["cells"]
    rows = [
        {"teacher": "original_odd", "activation_variant": "shifted", **row}
        for row in controls
        if row["variant"] == plan["variant"]
        and row["learning_rate"] == plan["learning_rate"]
    ]
    require(len(rows) == plan["reused_control_cells"], "Missing original controls")
    for teacher, arm in itertools.product(
        plan["teachers"], plan["activation_variants"]
    ):
        if teacher == "original_odd" and arm == "shifted":
            continue
        settings = dict(
            original,
            learning_rate=plan["learning_rate"],
            activation_variant=arm,
        )
        for depth, strength, seed in itertools.product(
            settings["branches"], settings["teacher_strengths"], settings["seeds"]
        ):
            directory = OUTPUT / teacher / arm / f"d{depth}-a{strength}-seed{seed}"
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text())
            else:
                result = train_parity_cell(
                    settings,
                    data[teacher],
                    plan["variant"],
                    depth,
                    strength,
                    seed,
                    directory,
                )
            rows.append({"teacher": teacher, "activation_variant": arm, **result})
            write_json(OUTPUT / "summary.json", summarize(rows, plan, original))
            print(
                json.dumps(
                    {
                        "teacher": teacher,
                        "activation_variant": arm,
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
