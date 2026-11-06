"""Reconstruct every cosine screen comparison and enforce the declared gate."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from research.early99.protocol import SEEDS, selected
from research.early99_cosine.readiness import verify_training_ready
from research.early99_cosine_reporting.run_screens import RESULTS, ROOT, RUNS, STUDY
from research.early99_optim_reporting.metrics import (
    paired_cell,
    screen_decision,
    task_metrics,
)
from research.early99_reporting.round2_audit import read, require, verify_csv

REPORTING_TESTS = 5
SCREEN_CELLS = 18


def sha256(path: Path) -> str:
    """Bind the reconstructed decision to the exact evidence."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Close rejected screens; advancing arms still require the full protocol."""
    verify_training_ready()
    plan = read(STUDY / "plan.json")
    audit = read(RESULTS / "audit-screen.json")
    require(
        audit["auditor_sha256"] == sha256(Path(__file__).with_name("audit.py")),
        "Checkpoint auditor changed",
    )
    continuation = read(RESULTS / "continuation.json")
    for name, expected in continuation["completed_artifact_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Completed cell changed: {name}")
    empty = ROOT / continuation["empty_directory_archive"]
    require(empty.is_dir() and not list(empty.iterdir()), "Empty-start evidence lost")
    # Generated locally by the pinned pytest invocation, not externally supplied XML.
    tests = ET.parse(RESULTS / "reporting-validation.xml").getroot()  # noqa: S314
    cases = tests.findall(".//testcase")
    require(
        len(cases) == REPORTING_TESTS
        and all(
            c.find(tag) is None
            for c in cases
            for tag in ["failure", "error", "skipped"]
        ),
        "Incomplete actual-exposure reporting checks",
    )
    task_lookup = {t["name"]: t for t in selected("short")}
    controls = {
        "parent": ROOT / plan["screen_control"],
        "quarter": RUNS / "peri-rms-scaled-ssmax-gain-quarter-lr2",
        "lr2": RUNS / "adamw-lr2",
        "baseline": ROOT / "runs/early99/baseline",
        "noqk": RUNS / "peri-rms-scaled-ssmax-noqk-lr2",
    }
    summaries = []
    for config in plan["configurations"]:
        name = config["name"]
        record = audit["candidates"][name]
        require(record["completed"] == SCREEN_CELLS, "Incomplete checkpoint audit")
        for label, control in controls.items():
            rows, tasks = [], []
            for task in plan["screen_tasks"]:
                group = [
                    paired_cell(task_lookup[task], seed, control, RUNS / name)
                    for seed in SEEDS
                ]
                rows.extend(group)
                tasks.append(task_metrics(task, group))
            passed, improved, advance = screen_decision(rows, tasks)
            stem = RESULTS / f"screen-{name}-vs-{label}"
            saved = read(stem.with_suffix(".json"))
            require(saved["runs"] == rows and saved["tasks"] == tasks, "Report drift")
            require(
                saved["passed"] == passed
                and saved["improved_task_medians"] == improved
                and saved["advance"] == (advance if label == "parent" else None)
                and saved["geometric_mean_epoch_ratio"] is None,
                "Screen gate or no-filtered-mean policy changed",
            )
            require(
                saved["search_plan_sha256"] == sha256(STUDY / "plan.json")
                and saved["candidate_plan_sha256"]
                == sha256(RUNS / name / "plan.json")
                == record["candidate_plan_sha256"],
                "Report provenance differs from checkpoint audit",
            )
            verify_csv(stem.with_suffix(".csv"), rows)
            verify_csv(stem.with_name(stem.name + "-tasks.csv"), tasks)
            if label == "parent":
                summaries.append(
                    {
                        "name": name,
                        "passed": passed,
                        "improved_task_medians": improved,
                        "advance": advance,
                        "task_medians": {
                            t["case"]: t["median_paired_epoch_ratio"] for t in tasks
                        },
                        "trainer_seconds_recorded_segments": sum(
                            r["candidate_seconds_last_run_segment"] for r in rows
                        ),
                    }
                )
    advances = any(s["advance"] for s in summaries)
    evidence = [
        STUDY / "plan.json",
        *RESULTS.glob("*.json"),
        *RESULTS.glob("*.csv"),
        *RESULTS.glob("*.txt"),
        *RESULTS.glob("*.xml"),
        *Path(__file__).parent.glob("*.py"),
        *ROOT.glob("research/early99_optim_reporting/*.py"),
        ROOT / "tests/diff_gpt/application/test_early99_optim_metrics.py",
    ]
    output = RESULTS / "complete-audit.json"
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "study_complete": not advances,
        "requires_full_short": advances,
        "requires_new_long": False if not advances else None,
        "screens": summaries,
        "checkpoint_cells_verified": audit["completed_new_scientific_runs"],
        "comparison_reports_reconstructed": len(summaries) * len(controls),
        "reporting_tests_passed": len(cases),
        "interruption": (
            "Empty next-cell startup; all15completed cells preserved; no replay."
        ),
        "retained_parent": plan["parent"] if not advances else None,
        "retained_long_diagnostic": "Already complete; no duplicate long run.",
        "unresolved_full_model_numeric_failures": 10,
        "files_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted(set(evidence))
            if p.resolve() != output.resolve()
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "files_sha256"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
