"""Complete every promoted short arm, audit it and freeze the shared-model choice."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from research.early99.experiments.compare import comparison, paired_cell
from research.early99.protocol import SEEDS, selected
from research.early99_reporting.round2_audit import (
    geometric_mean_epochs,
    read,
    require,
    screen_decision,
    task_metrics,
    verify_csv,
    verify_hash,
)
from research.early99_score_scale_reporting.run_screens import (
    PROGRESS,
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    write_json,
)


def promoted(name: str, plan: dict) -> bool:
    """Reconstruct the declared gate and verify its completed checkpoint audit."""
    path = RESULTS / f"screen-{name}-vs-parent.json"
    saved = read(path)
    rows, tasks = [], []
    lookup = {t["name"]: t for t in selected("short")}
    for task in plan["screen_tasks"]:
        group = [
            paired_cell(lookup[task], seed, ROOT / plan["screen_control"], RUNS / name)
            for seed in SEEDS
        ]
        rows.extend(group)
        tasks.append(task_metrics(task, group))
    require(rows == saved["runs"] and tasks == saved["tasks"], "Screen records changed")
    decision = screen_decision(rows, tasks)
    require(
        decision == (saved["passed"], saved["improved_task_medians"], saved["advance"]),
        "Screen promotion differs from raw records",
    )
    verify_csv(path.with_suffix(".csv"), rows)
    verify_csv(path.with_name(path.stem + "-tasks.csv"), tasks)
    verify_hash(STUDY / "plan.json", saved["search_plan_sha256"])
    verify_hash(RUNS / name / "plan.json", saved["candidate_plan_sha256"])
    audit = read(RESULTS / f"audit-{name}-screen.json")["candidates"][name]
    require(
        audit["completed"] == len(rows) and audit["passed"] == decision[0],
        "Incomplete or mismatched screen audit",
    )
    require(
        audit["candidate_plan_sha256"] == saved["candidate_plan_sha256"],
        "Screen audit used another source plan",
    )
    return decision[2]


def main() -> None:
    """Run short cells only; a separate post-selection action owns any long run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "Run from repository root")
    selection_path = RESULTS / "shared-selection.json"
    require(not selection_path.exists(), "A shared selection is already frozen")
    plan = read(STUDY / "plan.json")
    verify_hash(
        ROOT / "research/early99/results/round2/selection.json",
        plan["frozen_round2_selection_sha256"],
    )
    for name, expected in plan["control_plans_sha256"].items():
        verify_hash(RUNS / name / "plan.json", expected)
    candidates = []
    for cfg in plan["configurations"]:
        name = cfg["name"]
        require(
            read(STUDY / "configs" / f"{name}.json") == cfg,
            "Configuration differs from the declared plan",
        )
        require(
            read(RUNS / name / "plan.json")["configuration"] == cfg,
            "Executed configuration differs from the declared plan",
        )
        if promoted(name, plan):
            candidates.append(name)
    measurements = {}
    controls = {
        "parent": RUNS / plan["parent"],
        "baseline": ROOT / "runs/early99/baseline",
        "lr2": RUNS / "adamw-lr2",
        "noqk": RUNS / "peri-rms-scaled-ssmax-noqk-lr2",
    }
    for name in candidates:
        write_json(
            PROGRESS,
            {"phase": "full_short_running", "candidate": name, "promoted": candidates},
        )
        command = [
            sys.executable,
            "-m",
            "research.early99_score_scale.run",
            "--run",
            "--configuration",
            str(STUDY / "configs" / f"{name}.json"),
            "--tier",
            "short",
        ]
        with (RUNS / name / "full-dispatcher.log").open("a") as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)  # noqa: S603
        audit_command = [
            sys.executable,
            "-m",
            "research.early99_score_scale_reporting.audit",
            "--name",
            name,
            "--full",
            "--output",
            str(RESULTS / f"audit-{name}-full-short.json"),
        ]
        with (RUNS / name / "full-audit.log").open("a") as handle:
            subprocess.run(  # noqa: S603 -- Fixed auditor and declared candidate.
                audit_command, check=True, stdout=handle, stderr=subprocess.STDOUT
            )
        reports = {}
        for label, control in controls.items():
            directory = RUNS / name / "comparisons" / label
            report = comparison(control, RUNS / name, "short", output=directory)
            require(
                all(
                    r["candidate_status"] in {"passed", "budget_exhausted"}
                    for r in report["runs"]
                ),
                "Incomplete short comparison",
            )
            for suffix in [".json", ".csv", "-tasks.csv"]:
                shutil.copyfile(
                    directory / f"comparison-short{suffix}",
                    RESULTS / f"{name}-vs-{label}-short{suffix}",
                )
            verify_csv(directory / "comparison-short.csv", report["runs"])
            verify_csv(directory / "comparison-short-tasks.csv", report["tasks"])
            reports[label] = report
        parent = reports["parent"]
        require(
            geometric_mean_epochs(parent, "baseline")
            == plan["short_evidence"]["parent_gm_epochs"],
            "Parent measurement changed",
        )
        measurements[name] = {
            "passed": parent["candidate_passed"],
            "expected": parent["expected_cells"],
            "geometric_mean_epochs": geometric_mean_epochs(parent, "candidate"),
            "ratios": {
                label: r["geometric_mean_epoch_ratio"] for label, r in reports.items()
            },
            "parent_earlier": parent["paired_earlier"],
            "parent_tied": parent["paired_tied"],
            "parent_later": parent["paired_later"],
            "worst_parent_ratio": max(
                (
                    r["epoch_ratio"]
                    for r in parent["runs"]
                    if r["epoch_ratio"] is not None
                ),
                default=None,
            ),
            "failures": parent["failures_and_pending"],
            "trainer_seconds": sum(
                r["candidate_seconds_last_run_segment"] for r in parent["runs"]
            ),
        }
        write_json(RESULTS / "full-short-summary.json", measurements)
        print(json.dumps({"name": name, **measurements[name]}), flush=True)
    eligible = {plan["parent"]: plan["short_evidence"]["parent_gm_epochs"]}
    for name, measured in measurements.items():
        if measured["geometric_mean_epochs"] is not None:
            eligible[name] = measured["geometric_mean_epochs"]
    winner = min(eligible, key=eligible.get)
    selection = {
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "selected_name": winner,
        "new_shared_winner": winner != plan["parent"],
        "eligible_full_short_gm_epochs": eligible,
        "policy": plan["full_policy"],
        "promoted": candidates,
        "search_plan_sha256": hashlib.sha256(
            (STUDY / "plan.json").read_bytes()
        ).hexdigest(),
        "candidate_plan_sha256": {
            name: hashlib.sha256((RUNS / name / "plan.json").read_bytes()).hexdigest()
            for name in eligible
        },
        "round2_selection_sha256": plan["frozen_round2_selection_sha256"],
        "long_outcomes_used_for_selection": False,
        "noqk_ablation_is_separate_measured_reference": True,
    }
    write_json(selection_path, selection)
    write_json(
        PROGRESS,
        {
            "phase": "full_short_audited_and_selection_frozen",
            "selected": winner,
            "new_shared_winner": selection["new_shared_winner"],
        },
    )


if __name__ == "__main__":
    main()
