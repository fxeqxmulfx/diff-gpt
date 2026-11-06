"""Verify complete-suite and replay-interval selection without training a model."""

from pathlib import Path

import pytest

from research.early99_optim_reporting.metrics import (
    comparison,
    geometric_mean_bounds,
    screen_decision,
)

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "runs/early99/candidates/peri-rms-scaled-ssmax-inverse-width-lr2"
MIN_TASKS = 3


@pytest.mark.parametrize("improved", [2, 3])
def test_complete_screen_uses_the_declared_three_task_gate(improved: int) -> None:
    """Retain all18cells and require the declared task-improvement threshold."""
    rows = [{"candidate_status": "passed"}] * 18
    tasks = [
        {
            "median_paired_epoch_ratio_lower": ratio,
            "median_paired_epoch_ratio_upper": ratio,
        }
        for ratio in [0.5] * improved + [2.0] * (6 - improved)
    ]
    assert screen_decision(rows, tasks) == (18, improved, improved >= MIN_TASKS)


def test_replay_interval_crossing_the_gate_requires_resolution() -> None:
    """Require resolution when one uncertain task can change promotion."""
    rows = [{"candidate_status": "passed"}] * 18
    tasks = [
        {
            "median_paired_epoch_ratio_lower": 0.5,
            "median_paired_epoch_ratio_upper": 0.5,
        },
        {
            "median_paired_epoch_ratio_lower": 0.5,
            "median_paired_epoch_ratio_upper": 0.5,
        },
        {
            "median_paired_epoch_ratio_lower": 0.9,
            "median_paired_epoch_ratio_upper": 1.1,
        },
    ]
    with pytest.raises(ValueError, match="Replay uncertainty"):
        screen_decision(rows, tasks)


def test_geometric_mean_bounds_keep_every_failure() -> None:
    """A missing cell prevents both means; complete endpoints give a known interval."""
    rows = [{"lower": 100.0, "upper": 100.0}, {"lower": 4.0, "upper": 9.0}]
    assert geometric_mean_bounds(rows, "lower", "upper") == pytest.approx((20, 30))
    rows.append({"lower": None, "upper": None})
    assert geometric_mean_bounds(rows, "lower", "upper") == (None, None)


@pytest.mark.skipif(not PARENT.exists(), reason="Completed research control required")
def test_actual_75cell_parent_compared_with_itself_has_exact_unit_ratios(
    tmp_path: Path,
) -> None:
    """Replay all complete frozen histories through the new physical-work comparator."""
    result = comparison(PARENT, PARENT, "short", output=tmp_path)
    assert (
        result["expected_cells"],
        result["baseline_passed"],
        result["candidate_passed"],
    ) == (75, 75, 75)
    assert result["geometric_mean_epoch_ratio"] == 1
    assert result["candidate_geometric_mean_epochs"] == pytest.approx(
        16.266050125508265
    )
    assert all(row["epoch_ratio"] == 1 for row in result["runs"])
