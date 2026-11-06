"""Run the frozen new shared winner once on the reused long diagnostic."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from research.early99.protocol import SEEDS, selected
from research.early99_dimension_gain.readiness import sha256 as digest
from research.early99_muon_reporting.long_metrics import (
    long_contrasts,
    long_measurement,
)
from research.early99_muon_reporting.run_screens import (
    PROGRESS,
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    write_json,
)
from research.early99_muon_reporting.verify_results import verify, verify_comparison
from research.early99_optim_reporting.metrics import comparison
from research.early99_reporting.round2_audit import (
    read,
    require,
    verify_hash,
)


def frozen_choice() -> tuple[dict, dict]:
    """Require an audited, committed short decision before long training."""
    selection_path = RESULTS / "shared-selection.json"
    integrity_path = RESULTS / "selection-integrity.json"
    require(
        integrity_path.is_file(), "Freeze and commit the complete short audit first"
    )
    integrity = read(integrity_path)
    actual = verify()
    require(
        integrity["frozen_files_sha256"] == actual["frozen_files_sha256"]
        and integrity["selection_sha256"] == actual["selection_sha256"],
        "Frozen short evidence changed",
    )
    selection = read(selection_path)
    require(
        selection["new_shared_winner"], "No new shared winner needs this diagnostic"
    )
    for path in [selection_path, integrity_path, Path(__file__)]:
        committed = subprocess.run(  # noqa: S603 -- Read the exact local Git object.
            ["/usr/bin/git", "show", f"HEAD:{path.relative_to(ROOT)}"],
            check=True,
            capture_output=True,
        ).stdout
        require(committed == path.read_bytes(), f"Commit before long training: {path}")
    return selection, integrity


def publish(path: Path, data: bytes) -> None:
    """Preserve a completed diagnostic artifact on repeated reporting."""
    if path.exists():
        require(path.read_bytes() == data, f"Completed long artifact changed: {path}")
    else:
        path.write_bytes(data)


def verify_launch(selection: dict, integrity: dict) -> None:
    """Bind a started diagnostic to the unchanged selection and implementation."""
    launch = read(RESULTS / "long-launch.json")
    name = selection["selected_name"]
    require(
        launch["selected_name"] == name
        and launch["selection_sha256"] == integrity["selection_sha256"]
        and launch["no_long_outcomes_used_for_choice"] is True,
        "Existing diagnostic belongs to another selection",
    )
    for path, key in [
        (RESULTS / "selection-integrity.json", "selection_integrity_sha256"),
        (RUNS / name / "plan.json", "candidate_plan_sha256"),
        (Path(__file__), "launcher_sha256"),
    ]:
        verify_hash(path, launch[key])


def report_long(selection: dict, integrity: dict) -> dict:
    """Keep all 30 cells and compare only controls with completed long suites."""
    verify_launch(selection, integrity)
    name = selection["selected_name"]
    controls = {
        "parent": RUNS / read(STUDY / "plan.json")["parent"],
        "quarter": RUNS / "peri-rms-scaled-ssmax-gain-quarter-lr2",
        "baseline": ROOT / "runs/early99/baseline",
        "lr2": RUNS / "adamw-lr2",
        "noqk": RUNS / "peri-rms-scaled-ssmax-noqk-lr2",
    }
    expected_pairs = {(t["name"], seed) for t in selected("long") for seed in SEEDS}
    reports = {}
    with tempfile.TemporaryDirectory(prefix="early99-muon-long-") as temporary:
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
                    RESULTS / f"{name}-vs-{label}-long{suffix}",
                    (directory / f"comparison-long{suffix}").read_bytes(),
                )
            reports[label] = verify_comparison(
                RESULTS / f"{name}-vs-{label}-long.json",
                Path(temporary) / "verification",
            )
    audit_path = RESULTS / f"audit-{name}-complete.json"
    audit = read(audit_path)
    verify_hash(Path(__file__).with_name("audit.py"), audit["auditor_sha256"])
    verify_hash(
        Path(__file__).with_name("checkpoint.py"), audit["checkpoint_checker_sha256"]
    )
    record = audit["candidates"][name]
    short_count = len(selected("short")) * len(SEEDS)
    winner = long_measurement(reports["lr2"], "candidate")
    require(
        record["completed"] == short_count + len(expected_pairs)
        and record["passed"] == short_count + winner["passed"]
        and record["execution_errors"] == 0
        and audit["durable_update_journals_and_all_replayed_work_verified"],
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
        "winner_geometric_mean_epochs": reports["lr2"][
            "candidate_geometric_mean_epochs"
        ],
        "winner_geometric_mean_epochs_lower": reports["lr2"][
            "candidate_geometric_mean_epochs_lower"
        ],
        "winner_geometric_mean_epochs_upper": reports["lr2"][
            "candidate_geometric_mean_epochs_upper"
        ],
        "controls": {
            label: long_measurement(report, "baseline")
            for label, report in reports.items()
        },
        "comparisons": {label: long_contrasts(r) for label, r in reports.items()},
        "qk_parent_has_reused_long_measurement": True,
        "checkpoint_audit_sha256": digest(audit_path),
        "long_launch_sha256": digest(RESULTS / "long-launch.json"),
        "long_outcomes_used_for_selection_or_tuning": False,
    }
    publish(
        RESULTS / "long-summary.json", (json.dumps(summary, indent=2) + "\n").encode()
    )
    # Reconstruct short selection again: reporting the diagnostic cannot change it.
    frozen_choice()
    files = [
        audit_path,
        RESULTS / "long-launch.json",
        RESULTS / "long-summary.json",
        *(
            RESULTS / f"{name}-vs-{label}-long{suffix}"
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
        "ready_for_final_report": True,
    }
    publish(
        RESULTS / "complete-audit.json",
        (json.dumps(complete, indent=2) + "\n").encode(),
    )
    return summary


def main() -> None:
    """Train only after immutable selection; refuse duplicate live dispatchers."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()
    lock_path = PROGRESS.with_name("long.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection, integrity = frozen_choice()
        name = selection["selected_name"]
        launch_path = RESULTS / "long-launch.json"
        if args.run:
            require(
                not (RESULTS / "long-summary.json").exists(),
                "Long diagnostic is complete",
            )
            if launch_path.exists():
                verify_launch(selection, integrity)
            else:
                require(
                    not any(
                        (RUNS / name / t["name"]).exists() for t in selected("long")
                    ),
                    "Long cells already exist without their frozen launch record",
                )
                write_json(
                    launch_path,
                    {
                        "launched_at_utc": datetime.now(UTC).isoformat(),
                        "selected_name": name,
                        "selection_sha256": integrity["selection_sha256"],
                        "selection_integrity_sha256": digest(
                            RESULTS / "selection-integrity.json"
                        ),
                        "candidate_plan_sha256": digest(RUNS / name / "plan.json"),
                        "launcher_sha256": digest(Path(__file__)),
                        "git_head": subprocess.run(
                            ["/usr/bin/git", "rev-parse", "HEAD"],
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout.strip(),
                        "no_long_outcomes_used_for_choice": True,
                    },
                )
            write_json(PROGRESS, {"phase": "selected_long_running", "candidate": name})
            command = [
                sys.executable,
                "-m",
                "research.early99_muon.run",
                "--run",
                "--configuration",
                str(STUDY / "configs" / f"{name}.json"),
                "--tier",
                "long",
            ]
            with (RUNS / name / "long-dispatcher.log").open("a") as handle:
                subprocess.run(  # noqa: S603 -- Fixed runner and frozen configuration.
                    command, check=True, stdout=handle, stderr=subprocess.STDOUT
                )
        require(
            launch_path.is_file(), "The selected long diagnostic has not been launched"
        )
        audit_path = RESULTS / f"audit-{name}-complete.json"
        if not audit_path.exists():
            with (RUNS / name / "complete-audit.log").open("a") as handle:
                subprocess.run(  # noqa: S603 -- Fixed checkpoint auditor.
                    [
                        sys.executable,
                        "-m",
                        "research.early99_muon_reporting.audit",
                        "--name",
                        name,
                        "--full",
                        "--long",
                        "--output",
                        str(audit_path),
                    ],
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
        summary = report_long(selection, integrity)
        write_json(
            PROGRESS,
            {
                "phase": "complete_and_audited",
                "selected": name,
                "long_passed": summary["winner"]["passed"],
                "long_expected": 30,
            },
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
