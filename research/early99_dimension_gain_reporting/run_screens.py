"""Run the declared inverse-width screen and preserve every paired observation."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from research.early99.experiments.compare import paired_cell
from research.early99.protocol import SEEDS, selected
from research.early99_dimension_gain.readiness import verify_training_ready
from research.early99_reporting.round2_audit import (
    read,
    require,
    screen_decision,
    task_metrics,
    verify_csv,
)

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "research/early99_dimension_gain"
RESULTS = STUDY / "results"
RUNS = ROOT / "runs/early99/candidates"
PROGRESS = ROOT / "runs/early99-dimension-gain/progress.json"
SCREEN_CELLS = 18
SCREEN_TASKS = 6


def write_json(path: Path, value: dict) -> None:
    """Keep reports separate from immutable scientific source and plans."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def report_screen(name: str) -> dict:
    """Reconstruct all 18 rows; only the declared parent determines promotion."""
    plan = read(STUDY / "plan.json")
    declared = next(c for c in plan["configurations"] if c["name"] == name)
    require(
        read(RUNS / name / "plan.json")["configuration"] == declared,
        "Executed candidate differs from the declared search configuration",
    )
    task_lookup = {t["name"]: t for t in selected("short")}
    controls = {
        "parent": ROOT / plan["screen_control"],
        "lr2": RUNS / "adamw-lr2",
        "baseline": ROOT / "runs/early99/baseline",
        "noqk": RUNS / "peri-rms-scaled-ssmax-noqk-lr2",
    }
    reports = {}
    for label, control in controls.items():
        rows, tasks = [], []
        for task in plan["screen_tasks"]:
            group = [
                paired_cell(task_lookup[task], seed, control, RUNS / name)
                for seed in SEEDS
            ]
            rows.extend(group)
            tasks.append(task_metrics(task, group))
        require(
            len(rows) == SCREEN_CELLS and len(tasks) == SCREEN_TASKS,
            "Incomplete screen grid",
        )
        require(
            all(r["candidate_status"] in {"passed", "budget_exhausted"} for r in rows),
            "Pending or erroneous cell cannot finalize a screen",
        )
        passed, improved, gate = screen_decision(rows, tasks)
        report = {
            "reported_at_utc": datetime.now(UTC).isoformat(),
            "name": name,
            "control": str(control),
            "expected_cells": len(rows),
            "passed": passed,
            "improved_task_medians": improved,
            "advance": gate if label == "parent" else None,
            "selection_control": label == "parent",
            "geometric_mean_epoch_ratio": None,
            "search_plan_sha256": hashlib.sha256(
                (STUDY / "plan.json").read_bytes()
            ).hexdigest(),
            "candidate_plan_sha256": hashlib.sha256(
                (RUNS / name / "plan.json").read_bytes()
            ).hexdigest(),
            "runs": rows,
            "tasks": tasks,
        }
        stem = RESULTS / f"screen-{name}-vs-{label}"
        write_json(stem.with_suffix(".json"), report)
        for suffix, records in [(".csv", rows), ("-tasks.csv", tasks)]:
            path = stem.with_name(stem.name + suffix)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(records[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(records)
            verify_csv(path, records)
        reports[label] = report
    return reports["parent"]


def main() -> None:
    """Explicitly run screens, or regenerate reports without scientific updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--report", action="store_true")
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "Run from the repository root")
    plan = read(STUDY / "plan.json")
    verify_training_ready()
    summaries = []
    for config in plan["configurations"]:
        name = config["name"]
        require(
            read(STUDY / "configs" / f"{name}.json") == config,
            "Configuration file differs from the declared search plan",
        )
        if args.run:
            write_json(PROGRESS, {"phase": "screen_running", "candidate": name})
            command = [
                sys.executable,
                "-m",
                "research.early99_dimension_gain.run",
                "--configuration",
                str(STUDY / "configs" / f"{name}.json"),
            ]
            subprocess.run([*command, "--write-plan"], check=True)  # noqa: S603
            for task in plan["screen_tasks"]:
                command.extend(["--case", task])
            with (RUNS / name / "screen-dispatcher.log").open("a") as handle:
                subprocess.run(  # noqa: S603 -- Fixed module and archived arguments.
                    [*command, "--run", "--tier", "short"],
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
        report = report_screen(name)
        summary = {
            k: report[k] for k in ["name", "passed", "improved_task_medians", "advance"]
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(PROGRESS, {"phase": "screens_complete", "screens": summaries})


if __name__ == "__main__":
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with (PROGRESS.parent / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
