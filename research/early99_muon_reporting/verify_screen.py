"""Reconstruct all four complete screens and all five paired control comparisons."""

import json
from pathlib import Path

from research.early99.protocol import SEEDS, selected
from research.early99_dimension_gain.readiness import sha256
from research.early99_muon.readiness import verify_training_ready
from research.early99_muon_reporting.run_screens import (
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    write_json,
)
from research.early99_optim_reporting.metrics import (
    paired_cell,
    screen_decision,
    task_metrics,
)
from research.early99_reporting.round2_audit import (
    read,
    require,
    verify_csv,
    verify_hash,
)

REPORTING_TESTS = 5


def controls() -> dict:
    """Use the inverse-width parent alone for promotion."""
    return {
        "parent": RUNS / read(STUDY / "plan.json")["parent"],
        "quarter": RUNS / "peri-rms-scaled-ssmax-gain-quarter-lr2",
        "lr2": RUNS / "adamw-lr2",
        "baseline": ROOT / "runs/early99/baseline",
        "noqk": RUNS / "peri-rms-scaled-ssmax-noqk-lr2",
    }


def verify() -> dict:
    """Check gates from every raw row, without rerunning or changing a report."""
    verify_training_ready()
    validation = read(RESULTS / "selection-validation.json")
    require(
        validation["exit_code"] == 0 and validation["passed"] == REPORTING_TESTS,
        "Incomplete replay-interval selection checks",
    )
    for field in ["source_sha256", "evidence_sha256"]:
        for path, digest in validation[field].items():
            verify_hash(ROOT / path, digest)
    plan = read(STUDY / "plan.json")
    tasks = {row["name"]: row for row in selected("short")}
    audit_path = RESULTS / "audit-screen.json"
    audit = read(audit_path)
    verify_hash(Path(__file__).with_name("audit.py"), audit["auditor_sha256"])
    verify_hash(
        Path(__file__).with_name("checkpoint.py"), audit["checkpoint_checker_sha256"]
    )
    files = {audit_path, RESULTS / "selection-validation.json"}
    summaries = {}
    for configuration in plan["configurations"]:
        name = configuration["name"]
        summaries[name] = {}
        for label, control in controls().items():
            path = RESULTS / f"screen-{name}-vs-{label}.json"
            saved = read(path)
            rows, groups = [], []
            for task in plan["screen_tasks"]:
                group = [
                    paired_cell(tasks[task], seed, control, RUNS / name)
                    for seed in SEEDS
                ]
                rows.extend(group)
                groups.append(task_metrics(task, group))
            require(rows == saved["runs"] and groups == saved["tasks"], "Rows changed")
            require(
                saved["expected_cells"] == len(rows)
                and Path(saved["control"]) == control
                and saved["errors"]
                == [
                    row
                    for row in rows
                    if row["candidate_status"]
                    in {"execution_error", "numerical_training_failure"}
                ],
                "Control, grid or error accounting changed",
            )
            require(
                all(r["candidate_status"] != "pending" for r in rows), "Pending cell"
            )
            passed, improved, advance = screen_decision(rows, groups)
            require(
                saved["passed"] == passed
                and saved["improved_task_medians"] == improved
                and saved["advance"] == (advance if label == "parent" else None)
                and saved["selection_control"] == (label == "parent")
                and saved["geometric_mean_epoch_ratio"] is None,
                "Screen decision differs from the declared gate",
            )
            verify_hash(STUDY / "plan.json", saved["search_plan_sha256"])
            verify_hash(RUNS / name / "plan.json", saved["candidate_plan_sha256"])
            cell_csv = path.with_suffix(".csv")
            task_csv = path.with_name(path.stem + "-tasks.csv")
            verify_csv(cell_csv, rows)
            verify_csv(task_csv, groups)
            files.update([path, cell_csv, task_csv])
            checked = audit["candidates"][name]
            require(
                checked["completed"] == len(rows)
                and checked["passed"] == passed
                and checked["candidate_plan_sha256"] == saved["candidate_plan_sha256"],
                "Incomplete checkpoint audit",
            )
            summaries[name][label] = {
                "passed": passed,
                "improved_task_medians": improved,
                "advance": saved["advance"],
                "errors": checked["execution_errors"],
            }
    return {
        "ready_for_full_short": True,
        "all_four_declared_screens_complete": True,
        "summaries": summaries,
        "source_sha256": sha256(Path(__file__)),
        "evidence_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
    }


if __name__ == "__main__":
    result = verify()
    write_json(RESULTS / "screen-integrity.json", result)
    print(json.dumps(result["summaries"], indent=2))
