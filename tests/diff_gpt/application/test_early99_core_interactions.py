"""Check interaction bounds without dropping misses or replay uncertainty."""

import itertools

import pytest

from research.early99_core_ablations_reporting.interactions import interaction_bounds


def costs(base: float, embedding: float, peri: float, both: float) -> dict:
    """Make an exact four-arm measurement."""
    return {
        name: {"status": "passed", "lower": value, "upper": value}
        for name, value in zip(
            ("base", "embedding", "peri", "both"),
            (base, embedding, peri, both),
            strict=True,
        )
    }


def test_combination_is_compared_with_multiplicative_single_effects() -> None:
    """Two individually slower arms can interact favorably when combined."""
    assert interaction_bounds(costs(10, 20, 20, 10)) == (0.25, 0.25)
    assert interaction_bounds(costs(10, 20, 20, 40)) == (1, 1)
    assert interaction_bounds(costs(10, 20, 20, 80)) == (2, 2)


def test_replay_intervals_bound_every_cost_corner() -> None:
    """Uncertain numerator and denominator costs must move opposite bound endpoints."""
    measured = costs(10, 20, 15, 8)
    measured["embedding"]["upper"] = 22
    measured["peri"]["upper"] = 16
    measured["both"]["upper"] = 12
    low, high = interaction_bounds(measured)
    corners = [
        both * base / (embedding * peri)
        for base, embedding, peri, both in itertools.product(
            (10,), (20, 22), (15, 16), (8, 12)
        )
    ]
    assert low == pytest.approx(min(corners))
    assert high == pytest.approx(max(corners))
    measured["embedding"]["upper"] = None
    assert interaction_bounds(measured) == (0, high)
    measured["both"]["upper"] = None
    assert interaction_bounds(measured) == (0, None)


def test_missing_failed_or_zero_crossing_has_no_ratio() -> None:
    """A censored arm cannot produce a success-only interaction statistic."""
    measured = costs(10, 20, 20, 10)
    measured["peri"]["status"] = "budget_exhausted"
    assert interaction_bounds(measured) == (None, None)
    measured["peri"]["status"] = "passed"
    measured["peri"]["lower"] = 0
    assert interaction_bounds(measured) == (None, None)
    with pytest.raises(ValueError, match="Incomplete factorial"):
        interaction_bounds(
            {name: value for name, value in measured.items() if name != "base"}
        )
