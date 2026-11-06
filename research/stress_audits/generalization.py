"""Inspect complete residual learning histories without training or reselection."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/transformer-stress"
DESTINATION = Path(__file__).with_name("results") / "residual-generalization.json"
ARMS = ("shifted", "learned0", "learned0.5", "learned1")
TEACHERS = ("original_odd", "mixed")
THRESHOLD = 0.05
UPDATES = 512


def require(condition: bool, message: str) -> None:  # noqa: FBT001 -- Assertion helper.
    """Reject an incomplete or inconsistent post-hoc comparison."""
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict | list:
    """Read existing evidence without changing it."""
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    """Bind the analysis to the exact input artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_directory(row: dict) -> Path:
    """Locate each original cell, including reused controls, exactly once."""
    arm, teacher = row["initialization_arm"], row["teacher"]
    name = f"d{row['branches']}-a{row['strength']}-seed{row['seed']}"
    if arm == "shifted" and teacher == "original_odd":
        return RUNS / "rates-v1/lr0.01" / name
    if arm in ("shifted", "learned0"):
        return RUNS / "parity-v1" / teacher / row["activation_variant"] / name
    return (
        RUNS / "parity-init-v1" / f"init{arm.removeprefix('learned')}" / teacher / name
    )


def inspect(row: dict, settings: dict) -> dict:
    """Reconstruct train and held-out crossings from the entire fixed history."""
    directory = cell_directory(row)
    result, history = read(directory / "result.json"), read(directory / "history.json")
    require(all(row.get(k) == v for k, v in result.items()), "Result row changed")
    require(result["completed_updates"] == settings["updates"], "Incomplete budget")
    require(
        [r["updates"] for r in history]
        == list(range(0, settings["updates"] + 1, settings["observe_updates"])),
        "Incomplete observation cadence",
    )
    threshold = settings["target_relative_mse"]
    output = {
        k: row[k]
        for k in ("initialization_arm", "teacher", "branches", "strength", "seed")
    }
    for split in ("train", "heldout"):
        require(
            all(
                r[split]["finite"] and math.isfinite(r[split]["relative_mse"])
                for r in history
            ),
            "Nonfinite observation cannot be omitted",
        )
        first = next(
            (r["updates"] for r in history if r[split]["relative_mse"] <= threshold),
            None,
        )
        final = history[-1][split]["relative_mse"]
        require(final == result[f"final_{split}_relative_mse"], "Final score differs")
        output[f"{split}_first_threshold_updates"] = first
        output[f"final_{split}_relative_mse"] = final
    require(
        output["heldout_first_threshold_updates"] == result["first_threshold_updates"],
        "Held-out crossing differs",
    )
    require(
        result["status"]
        == (
            "passed"
            if output["heldout_first_threshold_updates"] is not None
            else "budget_miss"
        ),
        "Held-out status differs",
    )
    output.update(
        heldout_status=result["status"],
        final_heldout_train_ratio=output["final_heldout_relative_mse"]
        / output["final_train_relative_mse"],
        directory=str(directory.relative_to(ROOT)),
        artifact_sha256={
            name: digest(directory / name)
            for name in ("result.json", "history.json", "checkpoint.pt")
        },
    )
    return output


def metrics(rows: list[dict]) -> dict:
    """Summarize all cells, including every held-out miss."""
    require(bool(rows), "Empty group")
    return {
        "cells": len(rows),
        "train_threshold_crossings": sum(
            r["train_first_threshold_updates"] is not None for r in rows
        ),
        "heldout_threshold_crossings": sum(
            r["heldout_first_threshold_updates"] is not None for r in rows
        ),
        "mean_final_train_relative_mse": statistics.mean(
            r["final_train_relative_mse"] for r in rows
        ),
        "mean_final_heldout_relative_mse": statistics.mean(
            r["final_heldout_relative_mse"] for r in rows
        ),
        "median_final_heldout_train_ratio": statistics.median(
            r["final_heldout_train_ratio"] for r in rows
        ),
        "worst_final_heldout_relative_mse": max(
            r["final_heldout_relative_mse"] for r in rows
        ),
    }


def analyze() -> dict:
    """Require the full 96-cell grid before publishing a descriptive diagnosis."""
    summary_path = ROOT / "research/transformer_stress_parity_init/results/summary.json"
    summary = read(summary_path)
    require(
        summary == read(RUNS / "parity-init-v1/summary.json"),
        "Published summary changed",
    )
    require(summary["complete"], "Incomplete original comparison")
    manifest_path = RUNS / "parity-init-v1/manifest.json"
    settings = read(manifest_path)["parent_settings"]
    require(
        settings["target_relative_mse"] == THRESHOLD and settings["updates"] == UPDATES,
        "Changed original protocol",
    )
    expected = set(itertools.product(ARMS, TEACHERS, (4, 16), (0.25, 1.0), (0, 1, 2)))
    keys = [
        (r["initialization_arm"], r["teacher"], r["branches"], r["strength"], r["seed"])
        for r in summary["cells"]
    ]
    require(
        len(keys) == len(expected) and set(keys) == expected,
        "Wrong or duplicated scientific grid",
    )
    rows = [inspect(row, settings) for row in summary["cells"]]
    require(
        len({r["directory"] for r in rows}) == len(expected),
        "Reused cell appears twice",
    )
    artifacts = [summary_path, manifest_path, Path(__file__)]
    artifacts += [
        ROOT / f"research/{study}/results/audit.json"
        for study in ("transformer_stress_parity", "transformer_stress_parity_init")
    ]
    return {
        "scope": (
            "Post-hoc reconstruction of existing histories only. No new training, "
            "changed observations, checkpoint selection or original pass criterion. "
            "Train crossings are diagnostic and never replace held-out crossings. "
            "This checks recorded histories and binds checkpoints by hash; it "
            "does not repeat the earlier checkpoint forward replay."
        ),
        "interpretation": (
            "A training/held-out gap shows that fitting the finite training set "
            "is insufficient. It does not by itself identify a causal architecture "
            "defect or prove that more regularization will help. These are "
            "adaptively reused synthetic tasks, not fresh transfer evidence."
        ),
        "threshold_relative_mse": settings["target_relative_mse"],
        "updates": settings["updates"],
        "complete_grid_verified": True,
        "overall": metrics(rows),
        "groups": [
            {
                "initialization_arm": arm,
                "teacher": teacher,
                "branches": depth,
                **metrics(
                    [
                        r
                        for r in rows
                        if (r["initialization_arm"], r["teacher"], r["branches"])
                        == (arm, teacher, depth)
                    ]
                ),
            }
            for arm, teacher, depth in itertools.product(ARMS, TEACHERS, (4, 16))
        ],
        "input_artifact_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in artifacts
        },
        "cells": rows,
    }


def main() -> None:
    """Publish the supplemental analysis separately from all frozen studies."""
    result = analyze()
    data = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if DESTINATION.exists():
        require(DESTINATION.read_text() == data, "Existing diagnostic changed")
    else:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        DESTINATION.write_text(data)
    print(json.dumps(result["overall"]))


if __name__ == "__main__":
    main()
