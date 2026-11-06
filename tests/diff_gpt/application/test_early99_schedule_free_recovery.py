"""Check that unobserved replay work cannot manufacture a promotion decision."""

import pytest

from research.early99_schedule_free_recovery.report import bounded_tasks


@pytest.mark.parametrize(
    ("known", "unknown_lower", "expected", "guaranteed", "possible"),
    [
        ((0.5, 0.75), 0.2, (0.5, 0.75), True, True),
        ((0.5, 1.5), 0.2, (0.5, 1.5), False, True),
        ((1.2, 2.0), 0.9, (1.2, 2.0), False, False),
        ((0.5, 0.75), 3.0, (0.75, 0.75), True, True),
        ((0.7, 1.0), 0.5, (0.7, 1.0), False, True),
    ],
)
def test_unknown_extra_work_bounds_every_possible_three_seed_median(
    known: tuple[float, float],
    unknown_lower: float,
    expected: tuple[float, float],
    *,
    guaranteed: bool,
    possible: bool,
) -> None:
    """Check endpoints and explicit realizations on both sides of each ordering."""
    rows = [
        {
            "case": "task",
            "epoch_ratio_lower_bound": value,
            "epoch_ratio_upper_bound": value,
        }
        for value in known
    ]
    rows.append(
        {
            "case": "task",
            "epoch_ratio_lower_bound": unknown_lower,
            "epoch_ratio_upper_bound": None,
        }
    )
    result = bounded_tasks(rows)[0]
    assert (
        result["median_epoch_ratio_lower_bound"],
        result["median_epoch_ratio_upper_bound"],
    ) == expected
    assert result["guaranteed_improved"] is guaranteed
    assert result["possibly_improved"] is possible
    for actual in [unknown_lower, *known, 1.0, 100.0]:
        if actual >= unknown_lower:
            median = sorted([*known, actual])[1]
            assert expected[0] <= median <= expected[1]
            if guaranteed:
                assert median < 1
            if not possible:
                assert median >= 1


def test_missing_seed_cannot_establish_a_replay_robust_decision() -> None:
    """Reject an incomplete task group before deriving its median bounds."""
    rows = [
        {"case": "task", "epoch_ratio_lower_bound": 0.1, "epoch_ratio_upper_bound": 0.1}
    ] * 2
    with pytest.raises(ValueError, match="Incomplete"):
        bounded_tasks(rows)
