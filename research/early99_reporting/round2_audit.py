"""Verify frozen short selection and completed long reports from raw histories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from research.early99.experiments.compare import comparison, paired_cell
from research.early99.protocol import SEEDS, selected

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research/early99/results/round2"
RUNS = ROOT / "runs/early99"
SELECTION_SHA256 = "ec56a5245f2c9626eab67e1d809bc8c6b7889010adf2cf3e774698cb5db24c81"
MIN_IMPROVED_SCREEN_TASKS = 3
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def require(condition: bool, message: str) -> None:  # noqa: FBT001
    """Keep integrity checks active even if Python is invoked with optimization."""
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict:
    """Read a preserved JSON record without modifying it."""
    return json.loads(path.read_text())


def verify_hash(path: Path, expected: str) -> None:
    """Reject a changed source, declared plan, frozen result or proof fixture."""
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Changed: {path}"
    )


def verify_csv(path: Path, records: list[dict]) -> None:
    """Check every cell, column and censoring marker against its full JSON record."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        actual = list(reader)
        require(
            len(reader.fieldnames or []) == len(set(reader.fieldnames or [])),
            f"Repeated CSV column: {path}",
        )
        require(set(reader.fieldnames or []) == set(records[0]), f"Columns: {path}")
    expected = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in records
    ]
    require(actual == expected, f"CSV disagrees with complete JSON: {path}")


def task_metrics(name: str, rows: list[dict]) -> dict:
    """Require all three paired seeds for per-task medians and worst ratios."""
    result = {"case": name, "tier": "short"}
    for prefix in ("baseline", "candidate"):
        values = [
            r[f"{prefix}_first99_epochs"]
            for r in rows
            if r[f"{prefix}_status"] == "passed"
        ]
        result[f"{prefix}_passed_seeds"] = len(values)
        result[f"{prefix}_median_first99_epochs"] = (
            statistics.median(values) if len(values) == len(SEEDS) else None
        )
        result[f"{prefix}_worst_first99_epochs"] = (
            max(values) if len(values) == len(SEEDS) else None
        )
    ratios = [r["epoch_ratio"] for r in rows]
    complete = all(value is not None for value in ratios)
    result["median_paired_epoch_ratio"] = (
        statistics.median(ratios) if complete else None
    )
    result["worst_paired_epoch_ratio"] = max(ratios) if complete else None
    return result


def screen_decision(rows: list[dict], tasks: list[dict]) -> tuple[int, int, bool]:
    """Implement the frozen full-coverage and three-improved-task gate."""
    passed = sum(r["candidate_status"] == "passed" for r in rows)
    improved = sum(
        t["median_paired_epoch_ratio"] is not None
        and t["median_paired_epoch_ratio"] < 1
        for t in tasks
    )
    return (
        passed,
        improved,
        passed == len(rows) and improved >= MIN_IMPROVED_SCREEN_TASKS,
    )


def verify_screen(name: str, selection: dict, search: dict) -> dict:
    """Recompute all 18 screen rows and the recorded promotion decision."""
    path = RESULTS / f"screen-{name}.json"
    saved = read(path)
    task_lookup = {task["name"]: task for task in selected("short")}
    rows, tasks = [], []
    for task_name in search["screen_tasks"]:
        group = [
            paired_cell(
                task_lookup[task_name],
                seed,
                Path(saved["control"]),
                RUNS / "candidates" / name,
            )
            for seed in SEEDS
        ]
        rows.extend(group)
        tasks.append(task_metrics(task_name, group))
    require(rows == saved["runs"], f"Raw screen rows differ: {name}")
    require(tasks == saved["tasks"], f"Screen task statistics differ: {name}")
    require(len(rows) == saved["expected_cells"], f"Screen count differs: {name}")
    passed, improved, advance = screen_decision(rows, tasks)
    require(
        (passed, improved, advance)
        == (saved["passed"], saved["improved_task_medians"], saved["advance"]),
        f"Screen decision differs: {name}",
    )
    require(saved["geometric_mean_epoch_ratio"] is None, "Screen aggregate forbidden")
    require(
        saved["candidate_plan_sha256"] == selection["candidate_plan_sha256"][name],
        f"Screen candidate plan differs: {name}",
    )
    plan_hash = saved.get("declared_plan_sha256", saved.get("search_plan_sha256"))
    require(
        plan_hash in selection["search_plan_sha256"].values(),
        f"Unrecognized screen plan: {name}",
    )
    verify_csv(path.with_suffix(".csv"), rows)
    verify_csv(path.with_name(path.stem + "-tasks.csv"), tasks)
    return {"completed": len(rows), "passed": passed, "advance": advance}


def verify_comparison(path: Path, temporary: Path) -> dict:
    """Recompute JSON and verify both CSVs without overwriting frozen reports."""
    saved = read(path)
    actual = comparison(
        Path(saved["baseline_directory"]),
        Path(saved["candidate_directory"]),
        saved["tier"],
        output=temporary / path.stem,
    )
    require(actual == saved, f"Raw comparison differs: {path}")
    verify_csv(path.with_suffix(".csv"), actual["runs"])
    verify_csv(path.with_name(path.stem + "-tasks.csv"), actual["tasks"])
    return actual


def geometric_mean_epochs(report: dict, prefix: str) -> float | None:
    """Never compute a successful-only mean when any expected cell is missing."""
    rows = report["runs"]
    if len(rows) != report["expected_cells"] or any(
        r[f"{prefix}_status"] != "passed" for r in rows
    ):
        return None
    return math.exp(
        statistics.mean(math.log(r[f"{prefix}_first99_epochs"]) for r in rows)
    )


def long_measurement(report: dict, prefix: str) -> dict:
    """Reproduce coverage, runtime and every budget-censored long observation."""
    rows = report["runs"]
    return {
        "passed": sum(r[f"{prefix}_status"] == "passed" for r in rows),
        "expected": report["expected_cells"],
        "trainer_seconds_including_setup_scoring": sum(
            r[f"{prefix}_seconds_last_run_segment"] for r in rows
        ),
        "censored": [
            {
                key: row[key]
                for key in (
                    "case",
                    "seed",
                    f"{prefix}_censor_epochs",
                    f"{prefix}_final_heldout_accuracy",
                )
            }
            for row in rows
            if row[f"{prefix}_status"] == "budget_exhausted"
        ],
    }


def long_contrasts(report: dict) -> dict:
    """Retain newly censored, recovered, jointly censored and regressed long cells."""
    fields = (
        "case",
        "seed",
        "baseline_status",
        "candidate_status",
        "baseline_first99_epochs",
        "candidate_first99_epochs",
        "baseline_censor_epochs",
        "candidate_censor_epochs",
        "epoch_ratio",
    )
    common = [r for r in report["runs"] if r["epoch_ratio"] is not None]

    def subset(rows: list[dict]) -> list[dict]:
        return [{key: row[key] for key in fields} for row in rows]

    return {
        "geometric_mean_epoch_ratio": report["geometric_mean_epoch_ratio"],
        "common_passes": len(common),
        "earlier": report["paired_earlier"],
        "tied": report["paired_tied"],
        "later": report["paired_later"],
        "recovered": subset(
            [
                r
                for r in report["runs"]
                if r["baseline_status"] == "budget_exhausted"
                and r["candidate_status"] == "passed"
            ]
        ),
        "newly_censored": subset(
            [
                r
                for r in report["runs"]
                if r["baseline_status"] == "passed"
                and r["candidate_status"] == "budget_exhausted"
            ]
        ),
        "both_censored": subset(
            [
                r
                for r in report["runs"]
                if r["baseline_status"] == r["candidate_status"] == "budget_exhausted"
            ]
        ),
        "worst_observed_ratio": max((r["epoch_ratio"] for r in common), default=None),
        "largest_regressions": subset(
            [
                r
                for r in sorted(common, key=lambda r: r["epoch_ratio"], reverse=True)[
                    :5
                ]
                if r["epoch_ratio"] > 1
            ]
        ),
    }


def verify_checkpoint_audit_records(
    screens: dict, promoted: set, selection: dict
) -> dict:
    """Link every candidate to its separately completed deep checkpoint audit."""
    records = {}
    full_count = len(selected("short")) * len(SEEDS)
    for path in sorted(RESULTS.glob("audit-*.json")):
        saved = read(path)
        for name, record in saved.get("candidates", {}).items():
            if name not in screens:
                continue
            expected = full_count if name in promoted else screens[name]["completed"]
            if record["completed"] != expected:
                continue
            require(
                record["candidate_plan_sha256"]
                == selection["candidate_plan_sha256"][name],
                f"Checkpoint-audit plan differs: {path}",
            )
            require(
                record["passed"]
                == (full_count if name in promoted else screens[name]["passed"]),
                f"Checkpoint-audit coverage differs: {path}",
            )
            for flag in (
                "all_checkpoints_histories_protocols_and_sources_retained",
                "checkpoint_optimizer_steps_and_settings_verified",
                "stop_at_first_exact_crossing_or_original_budget_verified",
            ):
                require(saved[flag] is True, f"Missing deep audit check: {path}")
            records[name] = {"file": str(path.relative_to(ROOT)), **record}
    require(set(records) == set(screens), "Missing candidate checkpoint audit")
    return records


def verify_proof_records(selection: dict) -> list[str]:
    """Verify existing checked fixtures and outputs; do not claim recompilation."""
    fixtures = []

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            if "fixture_sha256" in value:
                verify_hash(ROOT / value["fixture"], value["fixture_sha256"])
                fixtures.append(value["fixture"])
            if "output_sha256" in value:
                verify_hash(ROOT / value["output"], value["output_sha256"])
            for key in ("warnings", "exit_code", "unproved_placeholders"):
                if key in value:
                    require(value[key] == 0, f"Failed proof record: {key}")
            if "axioms" in value:
                axioms = value["axioms"]
                if isinstance(axioms, dict):
                    axioms = [a for names in axioms.values() for a in names]
                require(set(axioms) <= ALLOWED_AXIOMS, "Unexpected proof axiom")
            if "output_file" in value:
                output = (ROOT / value["output_file"]).read_text()
                require(
                    not any(x in output for x in ("sorryAx", "warning:", "error:")),
                    "Failed recorded proof output",
                )
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    for name, expected in selection["guarantee_record_sha256"].items():
        path = RESULTS / name
        verify_hash(path, expected)
        inspect(read(path))
    return fixtures


def audit() -> dict:
    """Verify short results now and long results only once all final artifacts exist."""
    require(Path.cwd().resolve() == ROOT, "Run this command from the repository root")
    selection_path = RESULTS / "selection.json"
    verify_hash(selection_path, SELECTION_SHA256)
    selection = read(selection_path)
    verify_hash(RESULTS / "final-short-summary.json", selection["short_summary_sha256"])
    summary = read(RESULTS / "final-short-summary.json")
    for name, expected in selection["search_plan_sha256"].items():
        verify_hash(RESULTS / name, expected)
    sources = {}
    for name, expected in selection["candidate_plan_sha256"].items():
        root = RUNS / "candidates" / name
        verify_hash(root / "plan.json", expected)
        plan = read(root / "plan.json")
        for filename, digest in plan["source_sha256"].items():
            verify_hash(root / "experiment_source" / filename, digest)
        sources[name] = len(plan["source_sha256"])
    search = read(RESULTS / "plan.json")
    screens = {
        name: verify_screen(name, selection, search) for name in summary["candidates"]
    }
    promoted = {name for name, record in screens.items() if record["advance"]}
    eligible = selection["eligible_full_coverage_geometric_mean_epochs"]
    require(promoted == set(eligible) - {"adamw-lr2"}, "Promotion coverage differs")
    checkpoint_audits = verify_checkpoint_audit_records(screens, promoted, selection)
    proofs = verify_proof_records(selection)
    comparisons = {}
    winner = selection["selected_name"]
    long_cells = [
        RUNS / "candidates" / winner / task["name"] / f"seed{seed}" / "result.json"
        for task in selected("long")
        for seed in SEEDS
    ]
    completed_long = sum(path.is_file() for path in long_cells)
    long_artifacts = [
        RESULTS / "long-summary.json",
        RESULTS / f"audit-{winner}-complete.json",
        RESULTS / f"{winner}-vs-baseline-long.json",
        RESULTS / f"{winner}-vs-lr2-long.json",
    ]
    long_verified = False
    with tempfile.TemporaryDirectory(prefix="early99-report-audit-") as directory:
        temporary = Path(directory)
        for key, expected in selection["comparison_sha256"].items():
            path = RESULTS / f"{key}-short.json"
            verify_hash(path, expected)
            comparisons[key] = verify_comparison(path, temporary)
        measured = {}
        for name in promoted:
            report = comparisons[f"{name}-vs-baseline"]
            measured[name] = geometric_mean_epochs(report, "candidate")
            require(
                measured[name]
                == summary["candidates"][name]["measurement"][
                    "geometric_mean_first99_epochs"
                ],
                f"Short summary differs: {name}",
            )
        measured["adamw-lr2"] = geometric_mean_epochs(
            comparisons[f"{winner}-vs-lr2"], "baseline"
        )
        require(measured == eligible, "Frozen eligible means differ")
        require(min(measured, key=measured.get) == winner, "Frozen winner differs")
        if completed_long == len(long_cells) and all(
            p.is_file() for p in long_artifacts
        ):
            long_summary = read(long_artifacts[0])
            require(
                long_summary["frozen_selection_sha256"] == SELECTION_SHA256
                and long_summary["selected_name"] == winner
                and long_summary["retained_short_selection"] == winner,
                "Long report changed the short selection",
            )
            deep = read(long_artifacts[1])["candidates"][winner]
            require(
                deep["completed"] == len(selected("all")) * len(SEEDS)
                and deep["candidate_plan_sha256"]
                == selection["candidate_plan_sha256"][winner],
                "Incomplete or mismatched final checkpoint audit",
            )
            for label, path in zip(
                ("baseline", "lr2"), long_artifacts[2:], strict=True
            ):
                report = verify_comparison(path, temporary)
                require(
                    report["candidate_passed"] == long_summary["winner"]["passed"],
                    "Long summary coverage differs",
                )
                require(
                    all(
                        r["candidate_status"] in {"passed", "budget_exhausted"}
                        for r in report["runs"]
                    ),
                    "Pending or erroneous long cell",
                )
                control_name = "baseline" if label == "baseline" else "incumbent_lr2"
                require(
                    long_summary[control_name] == long_measurement(report, "baseline")
                    and long_summary["winner"] == long_measurement(report, "candidate")
                    and long_summary[f"winner_vs_{label}"] == long_contrasts(report),
                    "Long summary metrics or censored/regressed rows differ",
                )
            long_verified = True
    return {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "frozen_selection_sha256": SELECTION_SHA256,
        "short_screens_recomputed": screens,
        "promoted_full_short_candidates_verified": sorted(promoted),
        "paired_short_comparisons_recomputed": len(comparisons),
        "candidate_source_archives_verified": sources,
        "recorded_checkpoint_audits_verified": checkpoint_audits,
        "deep_checkpoint_loading_repeated": False,
        "proof_fixtures_hash_verified": proofs,
        "proof_compilation_repeated": False,
        "selected_name": winner,
        "selected_long_cells_present": completed_long,
        "expected_long_cells": len(long_cells),
        "complete_long_reports_and_checkpoint_audit_verified": long_verified,
        "round2_results_ready_for_final_writeup": long_verified,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Read-only reconstruction of reported raw-history comparisons, screen "
            "gates, CSVs, source archives and frozen hashes. "
            "No training or reselection. Partial long outcomes are not analyzed. "
            "Deep checkpoint validation and "
            "proof compilation are separate recorded checks. This is not a claim "
            "that final documentation/commits or subsequent research are complete."
        ),
    }


def main() -> None:
    """Write only a separate accounting record, never a frozen scientific artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=RUNS / "round2-report-audit.json"
    )
    args = parser.parse_args()
    require(RESULTS not in args.output.resolve().parents, "Use a separate audit output")
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
