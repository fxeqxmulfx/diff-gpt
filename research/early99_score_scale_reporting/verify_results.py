"""Reconstruct complete short evidence and bind it to the frozen shared choice."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

from research.early99.experiments.compare import paired_cell
from research.early99.protocol import SEEDS, selected
from research.early99_reporting.round2_audit import (
    geometric_mean_epochs,
    read,
    require,
    screen_decision,
    task_metrics,
    verify_comparison,
    verify_csv,
    verify_hash,
    verify_proof_records,
)
from research.early99_score_scale_reporting.run_full import promoted
from research.early99_score_scale_reporting.run_screens import (
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
)


def digest(path: Path) -> str:
    """Hash a completed artifact without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen_files(name: str, plan: dict, controls: dict) -> set[Path]:
    """Reconstruct all four controls, keeping promotion exclusive to the parent."""
    lookup = {task["name"]: task for task in selected("short")}
    paths = set()
    for label, control in controls.items():
        path = RESULTS / f"screen-{name}-vs-{label}.json"
        saved = read(path)
        rows, tasks = [], []
        for task in plan["screen_tasks"]:
            group = [
                paired_cell(lookup[task], seed, control, RUNS / name) for seed in SEEDS
            ]
            rows.extend(group)
            tasks.append(task_metrics(task, group))
        require(
            rows == saved["runs"] and tasks == saved["tasks"],
            "Screen comparison differs from raw histories",
        )
        passed, improved, advance = screen_decision(rows, tasks)
        require(
            saved["passed"] == passed and saved["improved_task_medians"] == improved,
            "Screen counts or medians differ",
        )
        require(
            saved["advance"] == (advance if label == "parent" else None),
            "A secondary control changed promotion",
        )
        require(
            saved["selection_control"] == (label == "parent")
            and Path(saved["control"]).resolve() == control
            and saved["expected_cells"] == len(rows)
            and saved["geometric_mean_epoch_ratio"] is None,
            "Screen control, coverage or aggregate differs",
        )
        verify_hash(STUDY / "plan.json", saved["search_plan_sha256"])
        verify_hash(RUNS / name / "plan.json", saved["candidate_plan_sha256"])
        csv_path = path.with_suffix(".csv")
        tasks_path = path.with_name(path.stem + "-tasks.csv")
        verify_csv(csv_path, rows)
        verify_csv(tasks_path, tasks)
        paths.update({path, csv_path, tasks_path})
    return paths


def measurement(reports: dict) -> dict:
    """Derive coverage, every censored row and paired statistics from all cells."""
    report = reports["parent"]
    rows = report["runs"]
    ratios = [r["epoch_ratio"] for r in rows if r["epoch_ratio"] is not None]
    return {
        "passed": sum(r["candidate_status"] == "passed" for r in rows),
        "expected": len(rows),
        "geometric_mean_epochs": geometric_mean_epochs(report, "candidate"),
        "ratios": {k: r["geometric_mean_epoch_ratio"] for k, r in reports.items()},
        "parent_earlier": sum(r < 1 for r in ratios),
        "parent_tied": sum(r == 1 for r in ratios),
        "parent_later": sum(r > 1 for r in ratios),
        "worst_parent_ratio": max(ratios, default=None),
        "failures": [
            r
            for r in rows
            if r["baseline_status"] != "passed" or r["candidate_status"] != "passed"
        ],
        "trainer_seconds": sum(r["candidate_seconds_last_run_segment"] for r in rows),
    }


def verify_stress(stress: dict, settings: dict) -> None:
    """Reconstruct the declared grid, flags and aggregate from recorded rows."""
    expected = list(
        product(
            settings["multipliers"],
            settings["visible_keys"],
            settings["head_widths"],
            settings["qk_input_scales"],
        )
    )
    rows = stress["rows"]
    keys = ("multiplier", "visible_keys", "head_width", "qk_scale")
    require(
        [tuple(row[k] for k in keys) for row in rows] == expected,
        "Stress rows differ from the declared grid",
    )
    require(
        stress["expected_cases"] == settings["expected_cases"] == len(rows)
        and stress["scope"] == settings["scope"],
        "Stress coverage or scope differs",
    )
    failures = []
    for row in rows:
        bound = (
            row["multiplier"]
            * math.log(row["visible_keys"])
            * math.sqrt(row["head_width"])
        )
        require(row["real_score_bound"] == bound, "Stress score bound differs")
        numeric = [
            row[k]
            for k in (
                "maximum_absolute_score",
                "maximum_probability_absolute_error",
                "query_relative_gradient_error",
                "key_relative_gradient_error",
            )
        ]
        require(
            all(math.isfinite(v) and v >= 0 for v in numeric),
            "Stress error measurements are invalid",
        )
        flags = {
            "within_declared_bound_slack": numeric[0] <= bound + 1e-5 * max(1, bound),
            "accurate_probability": numeric[1]
            <= settings["maximum_probability_absolute_error"],
            "accurate_gradients": max(numeric[2:])
            <= settings["maximum_relative_gradient_error"],
        }
        require(
            all(row[k] is value for k, value in flags.items())
            and isinstance(row["finite"], bool),
            "Stress pass flags differ from error measurements",
        )
        if not row["finite"] or not all(flags.values()):
            failures.append(row)
    require(
        stress["failures"] == failures
        and stress["passed"] == len(rows) - len(failures),
        "Stress failures or pass count differ",
    )
    require(
        stress["maximum_probability_absolute_error"]
        == max(r["maximum_probability_absolute_error"] for r in rows)
        and stress["maximum_relative_gradient_error"]
        == max(
            max(r["query_relative_gradient_error"], r["key_relative_gradient_error"])
            for r in rows
        ),
        "Stress maximum error differs",
    )
    require(not failures, "Recorded numerical stress is incomplete")


def guarantee_files(old_selection: dict) -> set[Path]:
    """Verify unchanged proof artifacts and the explicit specialization record."""
    fixtures = verify_proof_records(old_selection)
    path = RESULTS / "score-guarantees.json"
    record = read(path)
    verify_hash(ROOT / record["implementation_file"], record["implementation_sha256"])
    verify_hash(STUDY / "plan.json", record["declared_plan_sha256"])
    reference = ROOT / record["proof_record"]
    verify_hash(reference, record["proof_record_sha256"])
    original = read(reference)
    artifacts = {path, reference, *(ROOT / p for p in fixtures)}
    for name, proof in record["proof_references"].items():
        require(proof == original["checks"][name], "Proof reference changed")
        verify_hash(ROOT / proof["fixture"], proof["fixture_sha256"])
        verify_hash(ROOT / proof["output"], proof["output_sha256"])
        artifacts.update({ROOT / proof["fixture"], ROOT / proof["output"]})
    inherited = record["inherited_bounds"]
    rms_path = ROOT / inherited["normalization_proof_record"]
    verify_hash(rms_path, inherited["normalization_proof_record_sha256"])
    rms = read(rms_path)
    verify_hash(ROOT / rms["fixture"], rms["fixture_sha256"])
    output = rms_path.with_name("lean-check.txt")
    verify_hash(output, rms["lean_output_sha256"])
    require(
        rms["exit_code"] == 0 and rms["warnings"] == 0 and not rms["extra_axioms"],
        "Normalization proof record is incomplete",
    )
    artifacts.update({rms_path, ROOT / rms["fixture"], output})
    validation_path = RESULTS / "validation.json"
    validation = read(validation_path)
    verify_hash(STUDY / "plan.json", validation["plan_sha256"])
    verify_hash(
        ROOT / "tests/diff_gpt/application/test_early99_score_scale.py",
        validation["test_source_sha256"],
    )
    for filename, sha in validation["source_sha256"].items():
        verify_hash(ROOT / filename, sha)
    stress_path = RESULTS / "score-stress.json"
    stress = read(stress_path)
    verify_hash(STUDY / "plan.json", stress["plan_sha256"])
    verify_hash(STUDY / "stress.py", stress["source_sha256"])
    verify_stress(stress, read(STUDY / "plan.json")["score_stress"])
    artifacts.update({validation_path, stress_path})
    return artifacts


def verify() -> dict:
    """Require both complete suites; do not inspect pending or partial long outcomes."""
    require(Path.cwd().resolve() == ROOT, "Run from the repository root")
    plan_path = STUDY / "plan.json"
    plan = read(plan_path)
    selection_path = RESULTS / "shared-selection.json"
    require(selection_path.is_file(), "Full short selection is not yet complete")
    selection = read(selection_path)
    verify_hash(plan_path, selection["search_plan_sha256"])
    verify_hash(
        ROOT / "research/early99/results/round2/selection.json",
        plan["frozen_round2_selection_sha256"],
    )
    require(
        selection["round2_selection_sha256"] == plan["frozen_round2_selection_sha256"],
        "Original round-two selection differs",
    )
    require(
        selection["policy"] == plan["full_policy"], "Shared selection policy differs"
    )
    require(
        selection["long_outcomes_used_for_selection"] is False,
        "Selection is not recorded as short-only",
    )
    require(
        selection["noqk_ablation_is_separate_measured_reference"] is True,
        "Shared and no-QK rankings were conflated",
    )
    artifacts = {plan_path, selection_path, RESULTS / "full-short-summary.json"}
    artifacts.update(
        guarantee_files(read(ROOT / "research/early99/results/round2/selection.json"))
    )
    controls = {
        "parent": RUNS / plan["parent"],
        "baseline": ROOT / "runs/early99/baseline",
        "lr2": RUNS / "adamw-lr2",
        "noqk": RUNS / "peri-rms-scaled-ssmax-noqk-lr2",
    }
    source_counts = {}
    for name, expected in plan["control_plans_sha256"].items():
        path = RUNS / name / "plan.json"
        verify_hash(path, expected)
        artifacts.add(path)
        for filename, sha in read(path)["source_sha256"].items():
            verify_hash(RUNS / name / "experiment_source" / filename, sha)
    advanced = []
    for cfg in plan["configurations"]:
        name = cfg["name"]
        root = RUNS / name
        source_plan = read(root / "plan.json")
        require(source_plan["configuration"] == cfg, "Executed configuration differs")
        config_path = STUDY / "configs" / f"{name}.json"
        require(read(config_path) == cfg, "Declared configuration file differs")
        require(read(root / "configuration.json") == cfg, "Run configuration differs")
        for filename, sha in source_plan["source_sha256"].items():
            verify_hash(ROOT / filename, sha)
            verify_hash(root / "experiment_source" / filename, sha)
        source_counts[name] = len(source_plan["source_sha256"])
        artifacts.update(
            {root / "plan.json", config_path, RESULTS / f"audit-{name}-screen.json"}
        )
        if promoted(name, plan):
            advanced.append(name)
        artifacts.update(screen_files(name, plan, controls))
    require(selection["promoted"] == advanced, "A promoted arm is missing or added")
    summary = read(RESULTS / "full-short-summary.json")
    require(set(summary) == set(advanced), "Full summary omits or adds candidates")
    expected_cells = len(selected("short")) * len(SEEDS)
    expected_pairs = {(t["name"], s) for t in selected("short") for s in SEEDS}
    eligible = {plan["parent"]: plan["short_evidence"]["parent_gm_epochs"]}
    with tempfile.TemporaryDirectory(prefix="early99-score-report-") as temporary:
        for name in advanced:
            reports = {}
            for label, control in controls.items():
                path = RESULTS / f"{name}-vs-{label}-short.json"
                report = verify_comparison(path, Path(temporary))
                require(
                    report["tier"] == "short"
                    and report["expected_cells"] == expected_cells,
                    "Wrong comparison tier or cell count",
                )
                require(
                    Path(report["baseline_directory"]).resolve() == control
                    and Path(report["candidate_directory"]).resolve() == RUNS / name,
                    "Comparison points to another candidate or control",
                )
                require(
                    {(r["case"], r["seed"]) for r in report["runs"]} == expected_pairs,
                    "Wrong or missing task/seed rows",
                )
                require(
                    all(
                        r["candidate_status"] in {"passed", "budget_exhausted"}
                        for r in report["runs"]
                    ),
                    "Pending or erroneous full-suite cell",
                )
                reports[label] = report
                artifacts.update(
                    {
                        path,
                        path.with_suffix(".csv"),
                        path.with_name(path.stem + "-tasks.csv"),
                    }
                )
            measured = measurement(reports)
            require(
                measured == summary[name], "Full summary differs from raw comparisons"
            )
            require(
                geometric_mean_epochs(reports["parent"], "baseline")
                == eligible[plan["parent"]],
                "Parent geometric mean differs",
            )
            audit_path = RESULTS / f"audit-{name}-full-short.json"
            audit = read(audit_path)
            require(
                audit["auditor_sha256"] == digest(Path(__file__).with_name("audit.py")),
                "Checkpoint audit implementation differs",
            )
            record = audit["candidates"][name]
            require(
                record["completed"] == expected_cells
                and record["passed"] == measured["passed"],
                "Incomplete or inconsistent full checkpoint audit",
            )
            require(
                record["candidate_plan_sha256"] == digest(RUNS / name / "plan.json"),
                "Checkpoint audit used another plan",
            )
            expected_censored = [
                {
                    "case": r["case"],
                    "seed": r["seed"],
                    "updates": next(
                        t["case"]["max_updates"]
                        for t in selected("short")
                        if t["name"] == r["case"]
                    ),
                }
                for r in measured["failures"]
                if r["candidate_status"] == "budget_exhausted"
            ]
            require(
                record["censored"] == expected_censored,
                "Checkpoint audit censoring differs",
            )
            require(
                record["execution_errors"] == 0 and record["resumed"] == 0,
                "Unexpected errors or resumed clocks require explicit accounting",
            )
            artifacts.add(audit_path)
            if measured["geometric_mean_epochs"] is not None:
                eligible[name] = measured["geometric_mean_epochs"]
    require(
        selection["eligible_full_short_gm_epochs"] == eligible,
        "Eligible full means differ",
    )
    winner = min(eligible, key=eligible.get)
    require(selection["selected_name"] == winner, "Frozen shared winner differs")
    require(
        selection["new_shared_winner"] == (winner != plan["parent"]),
        "Incorrect new-winner flag",
    )
    require(
        set(selection["candidate_plan_sha256"]) == set(eligible),
        "Selection omits or adds eligible source plans",
    )
    for name, sha in selection["candidate_plan_sha256"].items():
        verify_hash(RUNS / name / "plan.json", sha)
    return {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "selected_name": winner,
        "new_shared_winner": selection["new_shared_winner"],
        "selection_sha256": digest(selection_path),
        "source_files_verified": source_counts,
        "full_short_comparisons_reconstructed": len(advanced) * len(controls),
        "eligible_full_short_gm_epochs": eligible,
        "frozen_files_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in sorted(artifacts)
        },
        "verifier_source_sha256": digest(Path(__file__)),
        "long_outcomes_read": False,
        "proof_compilation_repeated": False,
        "scope": (
            "Complete raw-history/JSON/CSV comparison and "
            "short selection reconstruction. "
            "Checkpoint loading is the separately recorded completed audit. "
            "No long outcome is read."
        ),
    }


def main() -> None:
    """Audit the evidence and optionally freeze its manifest before long training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    result = verify()
    output = ROOT / "runs/early99-score-scale/report-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.freeze:
        manifest = RESULTS / "selection-integrity.json"
        if manifest.exists():
            old = read(manifest)
            require(
                old["frozen_files_sha256"] == result["frozen_files_sha256"]
                and old["selection_sha256"] == result["selection_sha256"],
                "Frozen short evidence changed",
            )
        else:
            manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "frozen_files_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
