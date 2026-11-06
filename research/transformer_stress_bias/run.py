"""Run the frozen residual protocol with explicitly declared zero-bias additions."""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import shutil
import subprocess
from pathlib import Path

import torch

from research.transformer_stress.common import ROOT, require, write_json
from research.transformer_stress_bias.learning import train_bias_cell
from research.transformer_stress_parity.run import data_digest, metrics

PLAN = Path(__file__).with_name("plan.json")
PARENT = ROOT / "runs/transformer-stress/decay-v1"
OUTPUT = ROOT / "runs/transformer-stress/bias-v1"
VALIDATION = PLAN.parent / "results/validation.json"
VALIDATION_OUTPUT = VALIDATION.with_name("validation-pytest.txt")
TEST = ROOT / "tests/diff_gpt/application/test_transformer_stress_bias.py"
VALIDATION_TESTS = 5


def read(path: Path) -> dict:
    """Read a preserved scientific record."""
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    """Bind a setting, source or control to its exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings_for(plan: dict, original: dict, activation: str, bias_mode: str) -> dict:
    """Retain the parent protocol at decay 0.1 and select a declared bias wrapper."""
    require(activation in plan["activation_variants"], "Undeclared activation")
    require(
        bias_mode in [plan["control_bias_mode"], *plan["bias_modes"]],
        "Undeclared bias mode",
    )
    return dict(
        original,
        activation_variant=activation,
        learning_rate=plan["learning_rate"],
        bias_mode=bias_mode,
        weight_decay=plan["weight_decay"],
    )


def key(row: dict) -> tuple:
    """Identify a unique activation/mode/teacher/depth/strength/seed cell."""
    return tuple(
        row[name]
        for name in (
            "activation_variant",
            "bias_mode",
            "teacher",
            "branches",
            "strength",
            "seed",
        )
    )


def expected_keys(plan: dict, original: dict) -> set[tuple]:
    """Require the exact declared Cartesian grid, not merely a row count."""
    return set(
        itertools.product(
            plan["activation_variants"],
            [plan["control_bias_mode"], *plan["bias_modes"]],
            plan["teachers"],
            original["branches"],
            original["teacher_strengths"],
            original["seeds"],
        )
    )


def control_rows(plan: dict) -> list[dict]:
    """Reuse all 48 parent rows, including the mixed-teacher regressions."""
    published = ROOT / "research/transformer_stress_decay/results/summary.json"
    require(read(published) == read(PARENT / "summary.json"), "Parent summary differs")
    return [
        dict(row, bias_mode=plan["control_bias_mode"], added_bias_parameters=0)
        for row in read(published)["cells"]
        if row["weight_decay"] == plan["weight_decay"]
    ]


def cell_directory(
    activation: str,
    bias_mode: str,
    teacher: str,
    depth: int,
    strength: float,
    seed: int,
) -> Path:
    """Keep every new scientific cell in a distinct, stable directory."""
    return (
        OUTPUT / bias_mode / activation / teacher / f"d{depth}-a{strength}-seed{seed}"
    )


def summarize(rows: list[dict], plan: dict, original: dict) -> dict:
    """Rank complete configurations without excluding misses or either teacher."""
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
    for activation, mode in itertools.product(
        plan["activation_variants"],
        [plan["control_bias_mode"], *plan["bias_modes"]],
    ):
        cells = [
            row
            for row in rows
            if (row["activation_variant"], row["bias_mode"]) == (activation, mode)
        ]
        arms.append(
            {
                "activation_variant": activation,
                "bias_mode": mode,
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
        best = {k: winner[k] for k in ("activation_variant", "bias_mode")}
    return {
        "complete": complete,
        "best_configuration": best,
        "scope": plan["scope"],
        "arms": arms,
        "cells": rows,
    }


def prepare(plan: dict) -> dict:
    """Require committed, validated source and freeze all data and control hashes."""
    for name, expected in plan["evidence_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Changed declared evidence: {name}")
    parent = read(PARENT / "manifest.json")
    sources = dict(parent["source_hashes"])
    for name, expected in sources.items():
        require(sha256(ROOT / name) == expected, f"Changed inherited source: {name}")
    own = [*PLAN.parent.glob("*.py"), PLAN, TEST, VALIDATION, VALIDATION_OUTPUT]
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
        "Missing or changed validation",
    )
    require(
        sha256(VALIDATION_OUTPUT) == validation["stdout_sha256"],
        "Validation output changed",
    )
    for name, expected in validation["source_sha256"].items():
        require(sha256(ROOT / name) == expected, "Validated implementation changed")
    parent_files = {
        name: sha256(PARENT / name)
        for name in ("summary.json", "data.pt", "mixed-data.pt", "data.json")
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
    """Complete all 96 new fixed-budget cells using the original training function."""
    plan = read(PLAN)
    manifest = prepare(plan)
    original = manifest["parent_settings"]
    data = {
        "original_odd": torch.load(OUTPUT / "data.pt", weights_only=True),
        "mixed": torch.load(OUTPUT / "mixed-data.pt", weights_only=True),
    }
    require(
        {name: data_digest(value) for name, value in data.items()}
        == read(OUTPUT / "data.json"),
        "Data fingerprints changed",
    )
    rows = control_rows(plan)
    require(len(rows) == plan["reused_control_cells"], "Missing controls")
    summarize(rows, plan, original)
    for activation, mode, teacher, depth, strength, seed in itertools.product(
        plan["activation_variants"],
        plan["bias_modes"],
        plan["teachers"],
        original["branches"],
        original["teacher_strengths"],
        original["seeds"],
    ):
        directory = cell_directory(activation, mode, teacher, depth, strength, seed)
        settings = settings_for(plan, original, activation, mode)
        if (directory / "result.json").exists():
            result = read(directory / "result.json")
        else:
            result = train_bias_cell(
                settings,
                data[teacher],
                plan["variant"],
                depth,
                strength,
                seed,
                directory,
            )
        row = dict(
            result,
            activation_variant=activation,
            bias_mode=mode,
            teacher=teacher,
            weight_decay=plan["weight_decay"],
        )
        require(
            (result["branches"], result["strength"], result["seed"])
            == (depth, strength, seed),
            "Cell identity changed",
        )
        rows.append(row)
        write_json(OUTPUT / "summary.json", summarize(rows, plan, original))
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "activation_variant",
                        "bias_mode",
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
    """Reject duplicate dispatchers and retain the original CPU numerical settings."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    with OUTPUT.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()


if __name__ == "__main__":
    main()
