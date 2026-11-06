"""Independently reconstruct short-only selection and preserve every failure."""

import argparse
import json
import tempfile
from pathlib import Path

from research.early99_dimension_gain.readiness import sha256
from research.early99_dimension_gain_reporting.verify_results import (
    verify as verify_parent,
)
from research.early99_reporting.round2_audit import (
    geometric_mean_epochs,
    read,
    require,
    verify_comparison,
    verify_hash,
)
from research.early99_schedule_free_reporting.run_full import promoted
from research.early99_schedule_free_reporting.run_screens import (
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    write_json,
)
from research.early99_schedule_free_reporting.verify_screen import controls
from research.early99_schedule_free_reporting.verify_screen import (
    verify as verify_screen,
)


def verify() -> dict:
    """Reconstruct full means for all promoted arms; never select using long scores."""
    screen = verify_screen()
    require(
        screen == read(RESULTS / "screen-integrity.json"), "Screen integrity changed"
    )
    previous = verify_parent()
    plan = read(STUDY / "plan.json")
    choice_path = RESULTS / "shared-selection.json"
    choice = read(choice_path)
    verify_hash(STUDY / "plan.json", choice["search_plan_sha256"])
    require(
        choice["previous_shared_selection_sha256"] == previous["selection_sha256"]
        and choice["policy"] == plan["full_policy"]
        and choice["long_outcomes_used_for_selection"] is False,
        "Changed previous choice, policy or short-only scope",
    )
    names = [c["name"] for c in plan["configurations"] if promoted(c["name"], plan)]
    require(names == choice["promoted"], "Promotion changed")
    parent_choice = read(
        ROOT / "research/early99_dimension_gain/results/shared-selection.json"
    )
    require(
        parent_choice["eligible_full_short_gm_epochs"][plan["parent"]]
        == plan["parent_short_gm_epochs"],
        "Parent short measurement changed",
    )
    files = {choice_path, STUDY / "plan.json", RESULTS / "screen-integrity.json"}
    files.update(ROOT / p for p in screen["evidence_sha256"])
    eligible = {plan["parent"]: plan["parent_short_gm_epochs"]}
    measurements = read(RESULTS / "full-short-summary.json")
    files.add(RESULTS / "full-short-summary.json")
    require(set(measurements) == set(names), "Missing promoted full suite")
    comparisons = 0
    with tempfile.TemporaryDirectory(prefix="schedule-free-short-audit-") as temporary:
        for name in names:
            audit_path = RESULTS / f"audit-{name}-full-short.json"
            audit = read(audit_path)
            verify_hash(Path(__file__).with_name("audit.py"), audit["auditor_sha256"])
            checked = audit["candidates"][name]
            reports = {}
            for label, control in controls().items():
                path = RESULTS / f"{name}-vs-{label}-short.json"
                report = verify_comparison(path, Path(temporary) / name / label)
                require(Path(report["baseline_directory"]) == control, "Wrong control")
                require(
                    Path(report["candidate_directory"]) == RUNS / name,
                    "Wrong candidate",
                )
                reports[label] = report
                files.update(
                    [
                        path,
                        path.with_suffix(".csv"),
                        path.with_name(path.stem + "-tasks.csv"),
                    ]
                )
                comparisons += 1
            parent = reports["parent"]
            require(
                checked["completed"] == parent["expected_cells"]
                and checked["passed"] == parent["candidate_passed"],
                "Incomplete full checkpoint audit",
            )
            gm = geometric_mean_epochs(parent, "candidate")
            saved = measurements[name]
            require(
                saved["geometric_mean_epochs"] == gm
                and saved["passed"] == checked["passed"]
                and saved["expected"] == checked["completed"]
                and saved["ratios"]
                == {
                    label: r["geometric_mean_epoch_ratio"]
                    for label, r in reports.items()
                }
                and saved["failures"] == parent["failures_and_pending"],
                "Full measurement differs from complete raw comparison",
            )
            if gm is not None:
                eligible[name] = gm
            files.add(audit_path)
    require(
        choice["eligible_full_short_gm_epochs"] == eligible,
        "Filtered or changed ranking",
    )
    winner = min(eligible, key=eligible.get)
    require(
        choice["selected_name"] == winner
        and choice["new_shared_winner"] == (winner != plan["parent"]),
        "Winner differs from complete-suite ranking",
    )
    for name, digest in choice["candidate_plan_sha256"].items():
        verify_hash(RUNS / name / "plan.json", digest)
    for configuration in plan["configurations"]:
        root = RUNS / configuration["name"]
        source = read(root / "plan.json")
        require(
            source["configuration"] == configuration, "Executed configuration changed"
        )
        for path, digest in source["source_sha256"].items():
            verify_hash(ROOT / path, digest)
            verify_hash(root / "experiment_source" / path, digest)
        files.add(root / "plan.json")
    files.update(Path(__file__).parent.glob("*.py"))
    return {
        "selected_name": winner,
        "new_shared_winner": choice["new_shared_winner"],
        "selection_sha256": sha256(choice_path),
        "full_short_comparisons_reconstructed": comparisons,
        "both_screens_verified": True,
        "ready_for_long_or_final_report": True,
        "frozen_files_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = verify()
    if args.write:
        write_json(RESULTS / "selection-integrity.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "frozen_files_sha256"}, indent=2
        )
    )
