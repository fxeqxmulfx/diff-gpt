"""Reconstruct complete screens and prove their gates despite uncertain replay cost."""

import csv
import json
import math
import statistics
from pathlib import Path

from research.early99.experiments.compare import paired_cell
from research.early99.protocol import SEEDS, selected
from research.early99_dimension_gain.readiness import sha256
from research.early99_reporting.round2_audit import (
    MIN_IMPROVED_SCREEN_TASKS,
    read,
    require,
    screen_decision,
    task_metrics,
    verify_csv,
    verify_hash,
)
from research.early99_schedule_free.readiness import verify_training_ready
from research.early99_schedule_free_recovery.accounting import (
    ROOT,
    trajectory_accounting,
)
from research.early99_schedule_free_reporting.run_screens import RESULTS, STUDY
from research.early99_schedule_free_reporting.verify_screen import controls

OUTPUT = Path(__file__).with_name("results")
RUNS = ROOT / "runs/early99/candidates"
EXPECTED_BOUND_TESTS = 6


def write_json(path: Path, value: dict) -> None:
    """Write supplemental results outside the frozen scientific package."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bounded_tasks(rows: list[dict]) -> list[dict]:
    """Use median monotonicity to bound every task's actual paired ratio."""
    tasks = []
    for name in dict.fromkeys(row["case"] for row in rows):
        group = [row for row in rows if row["case"] == name]
        require(len(group) == len(SEEDS), "Incomplete physical-exposure seed group")
        complete = all(row["epoch_ratio_lower_bound"] is not None for row in group)
        lower = (
            statistics.median(row["epoch_ratio_lower_bound"] for row in group)
            if complete
            else None
        )
        upper = (
            statistics.median(
                row["epoch_ratio_upper_bound"]
                if row["epoch_ratio_upper_bound"] is not None
                else math.inf
                for row in group
            )
            if complete
            else None
        )
        tasks.append(
            {
                "case": name,
                "median_epoch_ratio_lower_bound": lower,
                "median_epoch_ratio_upper_bound": upper
                if upper is not None and math.isfinite(upper)
                else None,
                "guaranteed_improved": upper is not None and upper < 1,
                "possibly_improved": lower is not None and lower < 1,
            }
        )
    return tasks


def main() -> None:
    """Finalize only a decision invariant under all unobserved replay exposure."""
    verify_training_ready()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = read(STUDY / "plan.json")
    audit_path = OUTPUT / "audit-screen.json"
    audit = read(audit_path)
    verify_hash(Path(__file__).with_name("audit.py"), audit["auditor_sha256"])
    recovery = trajectory_accounting()
    require(audit["recovery"] == recovery, "Audit used different recovery accounting")
    lookup = {row["name"]: row for row in selected("short")}
    summaries = {}
    validation_path = OUTPUT / "validation.json"
    validation = read(validation_path)
    require(
        validation["exit_code"] == 0 and validation["passed"] == EXPECTED_BOUND_TESTS,
        "Incomplete replay-bound validation",
    )
    files = {audit_path, validation_path}
    for path, digest in validation["files_sha256"].items():
        verify_hash(ROOT / path, digest)
        files.add(ROOT / path)
    for configuration in plan["configurations"]:
        name = configuration["name"]
        summaries[name] = {}
        for label, control in controls().items():
            path = RESULTS / f"screen-{name}-vs-{label}.json"
            canonical = read(path)
            rows, tasks = [], []
            for task in plan["screen_tasks"]:
                group = [
                    paired_cell(lookup[task], seed, control, RUNS / name)
                    for seed in SEEDS
                ]
                rows.extend(group)
                tasks.append(task_metrics(task, group))
            require(
                rows == canonical["runs"] and tasks == canonical["tasks"],
                "Canonical rows changed",
            )
            require(
                all(row["candidate_status"] != "pending" for row in rows),
                "Pending screen cell",
            )
            decision = screen_decision(rows, tasks)
            require(
                decision
                == (
                    canonical["passed"],
                    canonical["improved_task_medians"],
                    canonical["advance"] if label == "parent" else decision[2],
                ),
                "Canonical screen decision changed",
            )
            require(
                canonical["geometric_mean_epoch_ratio"] is None, "Forbidden screen GM"
            )
            verify_hash(STUDY / "plan.json", canonical["search_plan_sha256"])
            verify_hash(RUNS / name / "plan.json", canonical["candidate_plan_sha256"])
            verify_csv(path.with_suffix(".csv"), rows)
            verify_csv(path.with_name(path.stem + "-tasks.csv"), tasks)
            files.update(
                [
                    path,
                    path.with_suffix(".csv"),
                    path.with_name(path.stem + "-tasks.csv"),
                ]
            )
            physical = []
            for row in rows:
                affected = (
                    name == recovery["candidate"]
                    and row["case"] == recovery["case"]
                    and row["seed"] == recovery["seed"]
                )
                low = row["candidate_first99_epochs"]
                if affected and low is not None:
                    low += recovery["known_discarded_epochs"]
                base = row["baseline_first99_epochs"]
                paired = row["baseline_status"] == row["candidate_status"] == "passed"
                ratio = low / base if paired and base else None
                physical.append(
                    {
                        "case": row["case"],
                        "seed": row["seed"],
                        "candidate_status": row["candidate_status"],
                        "canonical_first99_epochs": row["candidate_first99_epochs"],
                        "actual_first99_epochs_lower_bound": low,
                        "actual_first99_epochs_upper_bound": None if affected else low,
                        "baseline_first99_epochs": base,
                        "epoch_ratio_lower_bound": ratio,
                        "epoch_ratio_upper_bound": None if affected else ratio,
                        "unobserved_work_uncertain": affected,
                    }
                )
            bounds = bounded_tasks(physical)
            minimum = sum(task["guaranteed_improved"] for task in bounds)
            maximum = sum(task["possibly_improved"] for task in bounds)
            gate_low = decision[0] == len(rows) and minimum >= MIN_IMPROVED_SCREEN_TASKS
            gate_high = (
                decision[0] == len(rows) and maximum >= MIN_IMPROVED_SCREEN_TASKS
            )
            checked = audit["candidates"][name]
            require(
                checked["completed"] == len(rows)
                and checked["passed"] == decision[0]
                and checked["candidate_plan_sha256"]
                == canonical["candidate_plan_sha256"],
                "Incomplete checkpoint audit",
            )
            report = {
                "candidate": name,
                "control": label,
                "passed": decision[0],
                "expected": len(rows),
                "canonical_improved_task_medians": decision[1],
                "guaranteed_improved_task_medians": minimum,
                "possible_improved_task_medians": maximum,
                "advance_for_all_replay_costs": gate_low if label == "parent" else None,
                "advance_for_some_replay_costs": gate_high
                if label == "parent"
                else None,
                "decision_invariant_to_unknown_work": gate_low == gate_high,
                "geometric_mean_epoch_ratio": None,
                "recovery": recovery if name == recovery["candidate"] else None,
                "tasks": bounds,
                "runs": physical,
            }
            output = OUTPUT / f"screen-{name}-vs-{label}-physical.json"
            write_json(output, report)
            csv_path = output.with_suffix(".csv")
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(physical[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(physical)
            verify_csv(csv_path, physical)
            files.update([output, csv_path])
            summaries[name][label] = {
                key: value
                for key, value in report.items()
                if key not in {"runs", "tasks", "recovery"}
            }
    parent_summaries = {name: report["parent"] for name, report in summaries.items()}
    require(
        all(
            row["decision_invariant_to_unknown_work"]
            for row in parent_summaries.values()
        ),
        "Replay uncertainty changes the gate: independent repetition required",
    )
    promoted = [
        name
        for name, row in parent_summaries.items()
        if row["advance_for_all_replay_costs"]
    ]
    summary = {
        "search_plan_sha256": sha256(STUDY / "plan.json"),
        "pretraining_source_commit": "4345819",
        "expected_cells": sum(row["expected"] for row in parent_summaries.values()),
        "passed": sum(row["passed"] for row in parent_summaries.values()),
        "promoted": promoted,
        "screens": summaries,
        "all_decisions_invariant_to_unobserved_work": True,
        "geometric_mean_epochs": None,
        "requires_full_short": promoted,
        "retained_parent_if_no_promotion": plan["parent"] if not promoted else None,
        "new_long_diagnostic_required_if_no_promotion": False,
        "long_outcomes_used_for_design_or_selection": False,
        "recovery": recovery,
    }
    summary_path = OUTPUT / "screen-summary.json"
    write_json(summary_path, summary)
    files.add(summary_path)
    complete = {
        "both_screens_complete_and_audited": True,
        "study_complete": not promoted,
        "requires_full_short": promoted,
        "completed_checkpoint_count": audit["completed_new_scientific_runs"],
        "decisions_invariant_to_unobserved_work": True,
        "retained_parent_if_no_promotion": summary["retained_parent_if_no_promotion"],
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
        "reporter_sha256": sha256(Path(__file__)),
        "accounting_sha256": sha256(Path(__file__).with_name("accounting.py")),
    }
    write_json(OUTPUT / "complete-audit.json", complete)
    print(json.dumps({"screens": parent_summaries, "promoted": promoted}, indent=2))


if __name__ == "__main__":
    main()
