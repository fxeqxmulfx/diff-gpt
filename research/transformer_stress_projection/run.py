"""Run the declared projection experiment with frozen controls and protocol."""

from __future__ import annotations

import fcntl
import itertools
import json
import shutil
import subprocess
from pathlib import Path

import torch

from research.transformer_stress.common import ROOT, require, write_json
from research.transformer_stress_bias.run import read, sha256
from research.transformer_stress_parity.run import data_digest, metrics
from research.transformer_stress_projection.learning import train_projection_cell

PLAN = Path(__file__).with_name("plan.json")
PARENT = ROOT / "runs/transformer-stress/bias-v1"
OUTPUT = ROOT / "runs/transformer-stress/projection-v1"
RESULTS = PLAN.parent / "results"
VALIDATION = RESULTS / "validation.json"
VALIDATION_OUTPUT = RESULTS / "validation-pytest.txt"
TEST = ROOT / "tests/diff_gpt/application/test_transformer_stress_projection.py"
VALIDATION_TESTS = 7
PROOF_STATEMENTS = 6


def settings_for(plan: dict, original: dict, mode: str) -> dict:
    """Keep the selected biased parent settings and identify only the projection."""
    require(
        mode in [plan["control_projection_mode"], *plan["projection_modes"]],
        "Undeclared projection",
    )
    return dict(
        original,
        activation_variant=plan["activation_variant"],
        bias_mode=plan["bias_mode"],
        learning_rate=plan["learning_rate"],
        weight_decay=plan["weight_decay"],
        projection_mode=mode,
    )


def key(row: dict) -> tuple:
    """Identify every paired projection/teacher/depth/strength/seed cell."""
    return tuple(
        row[k] for k in ("projection_mode", "teacher", "branches", "strength", "seed")
    )


def expected_keys(plan: dict, original: dict) -> set[tuple]:
    """Preserve the entire declared Cartesian grid, including ordinary controls."""
    return set(
        itertools.product(
            [plan["control_projection_mode"], *plan["projection_modes"]],
            plan["teachers"],
            original["branches"],
            original["teacher_strengths"],
            original["seeds"],
        )
    )


def control_rows(plan: dict) -> list[dict]:
    """Retain all 24 completed selected-parent cells, including its seven misses."""
    published = ROOT / "research/transformer_stress_bias/results/summary.json"
    parent = read(published)
    require(parent == read(PARENT / "summary.json"), "Parent summary differs")
    require(
        parent["best_configuration"]
        == {
            "activation_variant": plan["activation_variant"],
            "bias_mode": plan["bias_mode"],
        },
        "Declared parent differs from completed selection",
    )
    return [
        dict(row, projection_mode=plan["control_projection_mode"])
        for row in parent["cells"]
        if row["activation_variant"] == plan["activation_variant"]
        and row["bias_mode"] == plan["bias_mode"]
    ]


def cell_directory(
    mode: str, teacher: str, depth: int, strength: float, seed: int
) -> Path:
    """Keep each new scientific cell separate from all frozen ancestors."""
    return OUTPUT / mode / teacher / f"d{depth}-a{strength}-seed{seed}"


def summarize(rows: list[dict], plan: dict, original: dict) -> dict:
    """Choose by complete pooled coverage; never average a successful subset."""
    expected = expected_keys(plan, original)
    keys = [key(row) for row in rows]
    require(len(keys) == len(set(keys)), "Duplicate scientific cell")
    require(set(keys) <= expected, "Undeclared scientific cell")
    require(len(expected) == plan["total_comparison_cells"], "Declared grid differs")
    per_teacher = (
        len(original["branches"])
        * len(original["teacher_strengths"])
        * len(original["seeds"])
    )
    per_arm = per_teacher * len(plan["teachers"])
    arms = []
    for mode in [plan["control_projection_mode"], *plan["projection_modes"]]:
        cells = [r for r in rows if r["projection_mode"] == mode]
        arms.append(
            {
                "projection_mode": mode,
                **metrics(cells, per_arm),
                "teachers": {
                    teacher: metrics(
                        [r for r in cells if r["teacher"] == teacher], per_teacher
                    )
                    for teacher in plan["teachers"]
                },
            }
        )
    complete = set(keys) == expected
    best = None
    if complete:
        winner = min(
            arms,
            key=lambda r: (
                -r["passed"],
                r["geometric_mean_first_updates"]
                if r["passed"] == per_arm
                else r["worst_final_relative_mse"],
                r["median_final_relative_mse"],
            ),
        )
        best = winner["projection_mode"]
    return {
        "complete": complete,
        "best_configuration": best,
        "scope": plan["scope"],
        "arms": arms,
        "cells": rows,
    }


def prepare(plan: dict) -> dict:
    """Require committed tested source, proof and numerical evidence before updates."""
    for name, expected in plan["evidence_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Changed evidence: {name}")
    parent = read(PARENT / "manifest.json")
    sources = dict(parent["source_hashes"])
    for name, expected in sources.items():
        require(sha256(ROOT / name) == expected, f"Changed inherited source: {name}")
    own = [
        *PLAN.parent.glob("*.py"),
        PLAN,
        TEST,
        VALIDATION,
        VALIDATION_OUTPUT,
        PLAN.parent / "Guarantees.lean",
        RESULTS / "guarantees.json",
        RESULTS / "lean-check.txt",
        RESULTS / "numeric.json",
    ]
    for path in own:
        relative = str(path.relative_to(ROOT))
        committed = subprocess.run(  # noqa: S603 -- Read exact local Git objects.
            ["/usr/bin/git", "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        require(committed == path.read_bytes(), f"Commit before training: {relative}")
        sources[relative] = sha256(path)
    validation = read(VALIDATION)
    require(
        validation["exit_code"] == 0
        and validation["passed"] == VALIDATION_TESTS
        and validation["test_source_sha256"] == sha256(TEST),
        "Missing validation",
    )
    require(
        sha256(VALIDATION_OUTPUT) == validation["stdout_sha256"],
        "Validation output changed",
    )
    for name, expected in validation["source_sha256"].items():
        require(sha256(ROOT / name) == expected, "Validated implementation changed")
    numeric = read(RESULTS / "numeric.json")
    grid = plan["numeric_grid"]
    require(
        numeric["passed"] == numeric["expected"] == grid["cases"],
        "Incomplete numeric checks",
    )
    numeric_keys = [(r["width"], r["scale"], r["dtype"]) for r in numeric["cells"]]
    require(
        len(numeric_keys) == len(set(numeric_keys))
        and set(numeric_keys)
        == set(itertools.product(grid["widths"], grid["scales"], grid["dtypes"]))
        and all(r["passed"] for r in numeric["cells"]),
        "Wrong numeric grid",
    )
    for name, expected in numeric["source_sha256"].items():
        require(sha256(ROOT / name) == expected, "Numeric implementation changed")
    proof = read(RESULTS / "guarantees.json")
    require(
        proof["exit_code"] == 0
        and proof["warnings"] == 0
        and proof["statements"] == PROOF_STATEMENTS
        and not proof["extra_axioms"],
        "Proof validation missing",
    )
    for name, expected in proof["source_sha256"].items():
        require(sha256(ROOT / name) == expected, "Proof source changed")
    require(
        sha256(ROOT / proof["output"]) == proof["output_sha256"], "Proof output changed"
    )
    parent_files = {
        n: sha256(PARENT / n)
        for n in ("summary.json", "data.pt", "mixed-data.pt", "data.json")
    }
    record = {
        "plan": plan,
        "parent_settings": parent["parent_settings"],
        "source_hashes": sources,
        "parent_files_sha256": parent_files,
        "validation_sha256": sha256(VALIDATION),
        "torch": torch.__version__,
    }
    manifest = OUTPUT / "manifest.json"
    if manifest.exists():
        require(read(manifest) == record, "Existing experiment provenance changed")
        for name, expected in sources.items():
            require(
                sha256(OUTPUT / "sources" / name) == expected, "Archived source changed"
            )
        for name in ("data.pt", "mixed-data.pt", "data.json"):
            require(sha256(OUTPUT / name) == parent_files[name], "Copied data changed")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        for name in sources:
            target = OUTPUT / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        for name in ("data.pt", "mixed-data.pt", "data.json"):
            shutil.copyfile(PARENT / name, OUTPUT / name)
        write_json(manifest, record)
    return record


def run() -> None:
    """Complete every fixed-budget cell, recording all failures and later declines."""
    plan = read(PLAN)
    manifest = prepare(plan)
    original = manifest["parent_settings"]
    data = {
        "original_odd": torch.load(OUTPUT / "data.pt", weights_only=True),
        "mixed": torch.load(OUTPUT / "mixed-data.pt", weights_only=True),
    }
    require(
        {n: data_digest(v) for n, v in data.items()} == read(OUTPUT / "data.json"),
        "Data changed",
    )
    rows = control_rows(plan)
    require(len(rows) == plan["reused_control_cells"], "Missing controls")
    summarize(rows, plan, original)
    for mode, teacher, depth, strength, seed in itertools.product(
        plan["projection_modes"],
        plan["teachers"],
        original["branches"],
        original["teacher_strengths"],
        original["seeds"],
    ):
        directory = cell_directory(mode, teacher, depth, strength, seed)
        settings = settings_for(plan, original, mode)
        if (directory / "result.json").exists():
            result = read(directory / "result.json")
        else:
            result = train_projection_cell(
                settings,
                data[teacher],
                plan["variant"],
                depth,
                strength,
                seed,
                directory,
            )
        require(
            (result["branches"], result["strength"], result["seed"])
            == (depth, strength, seed),
            "Cell identity changed",
        )
        row = dict(
            result,
            projection_mode=mode,
            activation_variant=plan["activation_variant"],
            bias_mode=plan["bias_mode"],
            teacher=teacher,
            weight_decay=plan["weight_decay"],
        )
        rows.append(row)
        write_json(OUTPUT / "summary.json", summarize(rows, plan, original))
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "projection_mode",
                        "teacher",
                        "branches",
                        "strength",
                        "seed",
                        "status",
                        "first_threshold_updates",
                        "final_heldout_relative_mse",
                    )
                }
            ),
            flush=True,
        )
    require(summarize(rows, plan, original)["complete"], "Incomplete declared grid")


def main() -> None:
    """Reject duplicate dispatchers; keep the original single-thread CPU settings."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    with OUTPUT.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()


if __name__ == "__main__":
    main()
