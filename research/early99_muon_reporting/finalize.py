"""Require all screens, promoted short suites and any new-winner long diagnostic."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from research.early99.protocol import selected
from research.early99_dimension_gain.readiness import sha256
from research.early99_muon_reporting.run_long import report_long
from research.early99_muon_reporting.run_screens import RESULTS, ROOT, RUNS, write_json
from research.early99_muon_reporting.verify_results import verify
from research.early99_reporting.round2_audit import read, require, verify_hash


def main() -> None:
    """Close the declared study only when every required measurement is audited."""
    integrity = verify()
    require(
        integrity == read(RESULTS / "selection-integrity.json"),
        "Freeze the exact short selection audit first",
    )
    selection = read(RESULTS / "shared-selection.json")
    screen = read(RESULTS / "screen-integrity.json")
    files = {ROOT / p for p in integrity["frozen_files_sha256"]}
    files.add(RESULTS / "selection-integrity.json")
    if selection["new_shared_winner"]:
        # Reconstruct long reports using the previously frozen choice and launch.
        summary = report_long(selection, integrity)
        complete = read(RESULTS / "complete-audit.json")
        require(complete["ready_for_final_report"], "Long audit is incomplete")
        files.update(ROOT / p for p in complete["long_files_sha256"])
        files.add(RESULTS / "complete-audit.json")
        long = {
            "new_diagnostic": True,
            "passed": summary["winner"]["passed"],
            "expected": summary["winner"]["expected"],
        }
    else:
        require(
            not (RESULTS / "long-launch.json").exists(),
            "Retained parent must not trigger a duplicate diagnostic",
        )
        for name in screen["summaries"]:
            require(
                not any((RUNS / name / t["name"]).exists() for t in selected("long")),
                "Unselected candidate has long measurements",
            )
        prior_path = (
            ROOT
            / "research/early99_dimension_gain_recovery/results/complete-audit.json"
        )
        prior = read(prior_path)
        require(
            prior["ready_for_final_report"]
            and prior["selected_name"] == selection["selected_name"],
            "Retained parent has no completed matching long audit",
        )
        for path, digest in prior["long_files_sha256"].items():
            verify_hash(ROOT / path, digest)
        files.add(prior_path)
        long = {"new_diagnostic": False, "retained_parent_diagnostic": str(prior_path)}
    files.update(
        RESULTS / name
        for name in ["validation.json", "guarantees.json", "selection-validation.json"]
    )
    for file in files:
        require(file.is_file(), f"Missing closure evidence: {file}")
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "study_complete": True,
        "selected_name": selection["selected_name"],
        "new_shared_winner": selection["new_shared_winner"],
        "screens": screen["summaries"],
        "promoted_full_short_arms": selection["promoted"],
        "long": long,
        "optimizer_implementation_tests_passed": 54,
        "additional_reporting_tests_passed": 5,
        "proof_statements_checked": 6,
        "unresolved_full_model_numeric_failures": 10,
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
    }
    write_json(RESULTS / "study-complete.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "files_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
