"""Short-only winner selection requiring the whole replay-cost interval to agree."""

from __future__ import annotations

import math

from research.early99_reporting.round2_audit import require

SHORT_CELLS = 75


def select_winner(
    parent: str, parent_gm: float, measurements: dict
) -> tuple[str, dict]:
    """Choose a complete-suite winner only when all admissible costs support it."""
    require(math.isfinite(parent_gm) and parent_gm > 0, "Invalid parent mean")
    intervals = {parent: [parent_gm, parent_gm]}
    for name, measured in measurements.items():
        require(measured["expected"] == SHORT_CELLS, "Incomplete short grid")
        low = measured["geometric_mean_epochs_lower"]
        high = measured["geometric_mean_epochs_upper"]
        if measured["passed"] != SHORT_CELLS:
            require(low is None and high is None, "Failed cells must not have a mean")
            continue
        require(
            low is not None
            and high is not None
            and math.isfinite(low)
            and math.isfinite(high)
            and 0 < low <= high,
            "Complete winner selection requires finite valid replay bounds",
        )
        intervals[name] = [low, high]
    winner = min(intervals, key=lambda name: intervals[name][1])
    high = intervals[winner][1]
    for name, (other_low, _) in intervals.items():
        if name == winner:
            continue
        require(
            high <= other_low,
            "Overlapping winner intervals require a separately declared repeat",
        )
    require(
        winner == parent or high < parent_gm,
        "A new winner requires its upper mean below the parent",
    )
    return winner, intervals


def measurement(reports: dict) -> dict:
    """Retain full coverage, both mean bounds, all paired regressions and time."""
    parent = reports["parent"]
    worst_upper = [
        r["epoch_ratio_upper"]
        for r in parent["runs"]
        if r["epoch_ratio_upper"] is not None
    ]
    return {
        "passed": parent["candidate_passed"],
        "expected": parent["expected_cells"],
        "geometric_mean_epochs": parent["candidate_geometric_mean_epochs"],
        "geometric_mean_epochs_lower": parent["candidate_geometric_mean_epochs_lower"],
        "geometric_mean_epochs_upper": parent["candidate_geometric_mean_epochs_upper"],
        "ratios": {
            label: [
                r["geometric_mean_epoch_ratio_lower"],
                r["geometric_mean_epoch_ratio_upper"],
            ]
            for label, r in reports.items()
        },
        "parent_earlier": parent["paired_earlier"],
        "parent_tied": parent["paired_tied"],
        "parent_later": parent["paired_later"],
        "worst_observed_parent_ratio_upper": max(worst_upper, default=None),
        "failures": parent["failures_and_pending"],
        "trainer_seconds_last_recorded_segments": sum(
            r["candidate_seconds_last_run_segment"] or 0 for r in parent["runs"]
        ),
    }
