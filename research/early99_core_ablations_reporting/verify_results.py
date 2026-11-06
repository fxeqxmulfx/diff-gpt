"""Independently reconstruct short-only selection and preserve every failure."""

import argparse
import json
import tempfile
from pathlib import Path

from research.early99_core_ablations_reporting.run_full import promoted
from research.early99_core_ablations_reporting.run_screens import (
    RESULTS,
    ROOT,
    RUNS,
    STUDY,
    write_json,
)
from research.early99_core_ablations_reporting.selection import (
    measurement,
    select_winner,
)
from research.early99_core_ablations_reporting.verify_screen import (
    verify as verify_screen,
)
from research.early99_dimension_gain.readiness import sha256
from research.early99_dimension_gain_reporting.verify_results import (
    verify as verify_parent,
)
from research.early99_optim_reporting.metrics import comparison
from research.early99_reporting.round2_audit import (
    read,
    require,
    verify_csv,
    verify_hash,
)


def controls() -> dict:
    """Full-suite ranking uses the shared incumbent, while screens use old LR2."""
    plan = read(STUDY / "plan.json")
    return {
        "parent": RUNS / plan["shared_incumbent"],
        "baseline": ROOT / "runs/early99/baseline",
        "combined_peri": RUNS / "peri-rms-scaled-lr2",
        "lr2": RUNS / "adamw-lr2",
        "combined_ssmax": RUNS / "peri-rms-scaled-ssmax-lr2",
    }


def verify_comparison(path: Path, output: Path) -> dict:
    """Rebuild all actual-exposure values and CSV rows from the original cell files."""
    saved = read(path)
    actual = comparison(
        Path(saved["baseline_directory"]),
        Path(saved["candidate_directory"]),
        saved["tier"],
        output=output,
    )
    require(actual == saved, "Actual-exposure comparison changed")
    verify_csv(path.with_suffix(".csv"), actual["runs"])
    verify_csv(path.with_name(path.stem + "-tasks.csv"), actual["tasks"])
    return actual


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
        parent_choice["eligible_full_short_gm_epochs"][plan["shared_incumbent"]]
        == plan["shared_incumbent_short_gm_epochs"],
        "Parent short measurement changed",
    )
    files = {choice_path, STUDY / "plan.json", RESULTS / "screen-integrity.json"}
    files.update(ROOT / p for p in screen["evidence_sha256"])
    measurements = read(RESULTS / "full-short-summary.json")
    files.add(RESULTS / "full-short-summary.json")
    require(set(measurements) == set(names), "Missing promoted full suite")
    comparisons = 0
    with tempfile.TemporaryDirectory(prefix="core-short-audit-") as temporary:
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
            require(
                measurements[name] == measurement(reports),
                "Full measurement differs from every complete comparison",
            )
            files.add(audit_path)
    winner, eligible = select_winner(
        plan["shared_incumbent"], plan["shared_incumbent_short_gm_epochs"], measurements
    )
    require(
        choice["eligible_full_short_gm_epoch_intervals"] == eligible,
        "Filtered or changed actual-exposure ranking",
    )
    require(
        choice["selected_name"] == winner
        and choice["new_shared_winner"] == (winner != plan["shared_incumbent"]),
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
        "all_three_screens_verified": True,
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
