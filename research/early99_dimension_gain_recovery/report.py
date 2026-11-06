"""Complete the original long report with explicit recovery/exposure accounting."""

import json
import tempfile
from pathlib import Path

from research.early99.experiments.compare import comparison
from research.early99.protocol import SEEDS, selected
from research.early99_dimension_gain_recovery.accounting import (
    RECOVERY,
    trajectory_accounting,
)
from research.early99_dimension_gain_reporting.run_long import (
    frozen_choice,
    publish,
    verify_launch,
)
from research.early99_dimension_gain_reporting.run_screens import (
    PROGRESS,
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    write_json,
)
from research.early99_dimension_gain_reporting.verify_results import digest
from research.early99_reporting.round2_audit import (
    geometric_mean_epochs,
    long_contrasts,
    long_measurement,
    read,
    require,
    verify_comparison,
    verify_hash,
)

OUTPUT = Path(__file__).with_name("results")


def report_long(selection: dict, integrity: dict) -> dict:
    """Keep all 30 cells and compare only controls with completed long suites."""
    verify_launch(selection, integrity)
    name = selection["selected_name"]
    controls = {
        "parent": RUNS / read(STUDY / "plan.json")["parent"],
        "baseline": ROOT / "runs/early99/baseline",
        "lr2": RUNS / "adamw-lr2",
        "noqk": RUNS / "peri-rms-scaled-ssmax-noqk-lr2",
    }
    expected_pairs = {(t["name"], seed) for t in selected("long") for seed in SEEDS}
    reports = {}
    with tempfile.TemporaryDirectory(prefix="early99-dimension-long-") as temporary:
        for label, control in controls.items():
            directory = Path(temporary) / label
            report = comparison(control, RUNS / name, "long", output=directory)
            require(
                len(report["runs"]) == len(expected_pairs)
                and {(r["case"], r["seed"]) for r in report["runs"]} == expected_pairs,
                "Wrong long task/seed grid",
            )
            require(
                all(
                    row[f"{side}_status"] in {"passed", "budget_exhausted"}
                    for row in report["runs"]
                    for side in ["baseline", "candidate"]
                ),
                "Pending or erroneous long cell cannot finalize the diagnostic",
            )
            for suffix in [".json", ".csv", "-tasks.csv"]:
                publish(
                    OUTPUT / f"{name}-vs-{label}-long{suffix}",
                    (directory / f"comparison-long{suffix}").read_bytes(),
                )
            reports[label] = verify_comparison(
                OUTPUT / f"{name}-vs-{label}-long.json",
                Path(temporary) / "verification",
            )
    audit_path = OUTPUT / f"audit-{name}-complete.json"
    audit = read(audit_path)
    verify_hash(Path(__file__).with_name("audit.py"), audit["auditor_sha256"])
    record = audit["candidates"][name]
    short_count = len(selected("short")) * len(SEEDS)
    winner = long_measurement(reports["lr2"], "candidate")
    recovery = trajectory_accounting()
    last_segments = winner.pop("trainer_seconds_including_setup_scoring")
    winner["sum_last_segments_seconds"] = last_segments
    winner["sum_observed_segments_seconds"] = (
        last_segments + recovery["original_segment_last_observed_seconds"]
    )
    winner["time_caveat"] = recovery["unobserved_time_before_process_loss"]
    require(
        record["completed"] == short_count + len(expected_pairs)
        and record["passed"] == short_count + winner["passed"]
        and record["execution_errors"] == 0
        and record["resumed"] == 1,
        "Incomplete or inconsistent 105-cell checkpoint audit",
    )
    verify_hash(RUNS / name / "plan.json", record["candidate_plan_sha256"])
    expected_censored = [
        {
            "case": row["case"],
            "seed": row["seed"],
            "updates": task["case"]["max_updates"],
        }
        for task in selected("long")
        for row in winner["censored"]
        if row["case"] == task["name"]
    ]
    require(record["censored"] == expected_censored, "Long audit censoring differs")
    summary = {
        "selected_name": name,
        "diagnostic": "Reused 30-cell long regression suite; no tuning or reselection.",
        "frozen_selection_sha256": integrity["selection_sha256"],
        "selection_integrity_sha256": digest(RESULTS / "selection-integrity.json"),
        "candidate_plan_sha256": record["candidate_plan_sha256"],
        "winner": winner,
        "winner_geometric_mean_epochs": geometric_mean_epochs(
            reports["lr2"], "candidate"
        ),
        "controls": {
            label: long_measurement(report, "baseline")
            for label, report in reports.items()
        },
        "comparisons": {label: long_contrasts(r) for label, r in reports.items()},
        "qk_parent_has_reused_long_measurement": True,
        "checkpoint_audit_sha256": digest(audit_path),
        "long_launch_sha256": digest(RESULTS / "long-launch.json"),
        "long_outcomes_used_for_selection_or_tuning": False,
        "recovery": recovery,
        "recovery_declaration_sha256": digest(RECOVERY),
    }
    publish(
        OUTPUT / "long-summary.json", (json.dumps(summary, indent=2) + "\n").encode()
    )
    # Reconstruct short selection again: reporting the diagnostic cannot change it.
    frozen_choice()
    files = [
        audit_path,
        RESULTS / "long-launch.json",
        OUTPUT / "long-summary.json",
        *(
            OUTPUT / f"{name}-vs-{label}-long{suffix}"
            for label in controls
            for suffix in [".json", ".csv", "-tasks.csv"]
        ),
    ]
    complete = {
        "selected_name": name,
        "selection_sha256": integrity["selection_sha256"],
        "full_short_comparisons_verified": integrity[
            "full_short_comparisons_reconstructed"
        ],
        "long_comparisons_reconstructed": len(reports),
        "completed_checkpoint_audit": record,
        "long_files_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "reporter_source_sha256": digest(Path(__file__)),
        "recovery_declaration_sha256": digest(RECOVERY),
        "recovery_accounting_sha256": digest(Path(__file__).with_name("accounting.py")),
        "ready_for_final_report": True,
    }
    publish(
        OUTPUT / "complete-audit.json",
        (json.dumps(complete, indent=2) + "\n").encode(),
    )
    return summary


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    choice, integrity = frozen_choice()
    result = report_long(choice, integrity)
    write_json(
        PROGRESS,
        {
            "phase": "complete_and_audited_with_declared_recovery",
            "selected": choice["selected_name"],
            "long_passed": result["winner"]["passed"],
            "long_expected": 30,
        },
    )
    print(json.dumps(result, indent=2))
