"""Adversarial selection cases for actual-example replay intervals."""

import pytest

from research.early99_muon_reporting.long_metrics import (
    long_contrasts,
    long_measurement,
)
from research.early99_muon_reporting.selection import select_winner


def measured(low: float | None, high: float | None, passed: int = 75) -> dict:
    """Build a complete fixed short grid with explicit aggregate bounds."""
    return {
        "passed": passed,
        "expected": 75,
        "geometric_mean_epochs_lower": low,
        "geometric_mean_epochs_upper": high,
    }


def test_complete_improvement_must_survive_all_replayed_work() -> None:
    """A lower bound alone cannot justify displacing the parent."""
    with pytest.raises(ValueError, match="Overlapping"):
        select_winner("parent", 16, {"new": measured(15, 17)})
    assert select_winner("parent", 16, {"new": measured(14, 15)})[0] == "new"


def test_two_new_winners_with_overlapping_intervals_need_repetition() -> None:
    """Being better than the parent does not settle ranking between new arms."""
    with pytest.raises(ValueError, match="Overlapping"):
        select_winner("parent", 16, {"a": measured(13, 14), "b": measured(13.5, 15)})
    assert (
        select_winner("parent", 16, {"a": measured(13, 14), "b": measured(15, 15.5)})[0]
        == "a"
    )


def test_ties_retain_parent_and_failures_have_no_filtered_mean() -> None:
    """Coverage takes priority; an exact tie retains the first declared incumbent."""
    assert select_winner("parent", 16, {"new": measured(16, 16)})[0] == "parent"
    assert select_winner("parent", 16, {"new": measured(16, 17)})[0] == "parent"
    assert (
        select_winner("parent", 16, {"fast_but_missing": measured(None, None, 74)})[0]
        == "parent"
    )
    with pytest.raises(ValueError, match="must not have a mean"):
        select_winner("parent", 16, {"filtered": measured(1, 1, 74)})


def test_partial_grid_cannot_establish_winner() -> None:
    """A quick screen is not a 75-cell aggregate even if every screen cell passes."""
    partial = {**measured(1, 1, 18), "expected": 18}
    with pytest.raises(ValueError, match="Incomplete short grid"):
        select_winner("parent", 16, {"screen": partial})


def test_long_counts_keep_uncertain_successes_and_censor_bounds() -> None:
    """An uncertain cost is still a success; a miss prevents an aggregate mean."""
    rows = [
        {
            "case": "a",
            "seed": 0,
            "baseline_status": "passed",
            "candidate_status": "passed",
            "epoch_ratio": None,
            "epoch_ratio_lower": 0.8,
            "epoch_ratio_upper": 1.2,
            "candidate_seconds_last_run_segment": 1,
        },
        {
            "case": "b",
            "seed": 0,
            "baseline_status": "passed",
            "candidate_status": "budget_exhausted",
            "epoch_ratio": None,
            "candidate_seconds_last_run_segment": 2,
            "candidate_canonical_censor_epochs": 10,
            "candidate_censor_epochs_lower": 11,
            "candidate_censor_epochs_upper": 12,
            "candidate_final_heldout_accuracy": 0.98,
        },
    ]
    report = {
        "runs": rows,
        "expected_cells": 2,
        "paired_earlier": 0,
        "paired_tied": 0,
        "paired_later": 0,
        "geometric_mean_epoch_ratio": None,
        "geometric_mean_epoch_ratio_lower": None,
        "geometric_mean_epoch_ratio_upper": None,
    }
    contrasts = long_contrasts(report)
    assert contrasts["common_passes"] == contrasts["uncertain_order"] == 1
    assert contrasts["geometric_mean_epoch_ratio"] is None
    assert contrasts["newly_censored"] == [rows[1]]
    measured_long = long_measurement(report, "candidate")
    assert measured_long["passed"] == 1
    assert [
        measured_long["censored"][0]["candidate_censor_epochs_lower"],
        measured_long["censored"][0]["candidate_censor_epochs_upper"],
    ] == [11, 12]
