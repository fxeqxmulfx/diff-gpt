"""
Check the numpy seasonal naive against hand-computed forecasts.

tests/diff_gpt/infrastructure/leaderboards checks it against statsforecast
through the leaderboard's own evaluation; these cases pin each step without it.
"""

from statistics import NormalDist

import numpy as np
import pytest

from diff_gpt.application.gift_eval.files import QUANTILE_LEVELS, Windows
from diff_gpt.application.gift_eval.seasonal_naive import (
    LEVELS,
    TOL,
    Z,
    forecast,
    impute,
    seasonal_naive,
    tails,
)

NAN = np.nan


@pytest.mark.parametrize(
    ("context", "filled"),
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, NAN, NAN, 4, NAN], [1, 1, 1, 4, 4]),
        ([NAN, NAN, 3, NAN, 5], [3, 3, 3, 3, 5]),
        ([NAN, NAN], [0, 0]),
        ([NAN], [0]),
    ],
)
def test_impute_is_last_value_imputation(
    context: list[float], filled: list[float]
) -> None:
    """NaN take the last value before, leading NaN the first, all-NaN zero."""
    values = np.array(context, dtype=np.float32)
    result = impute(values)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, filled)
    np.testing.assert_array_equal(values, context)  # The input is left alone.


def test_z_is_the_normal_quantile() -> None:
    """Z holds Φ⁻¹ of the levels the predictor asks statsforecast for."""
    expected = [NormalDist().inv_cdf(level) for level in LEVELS]
    np.testing.assert_allclose(Z, expected, rtol=0, atol=1e-15)


def test_seasonal_naive_by_hand() -> None:
    """Tile the last season; widen by the seasonal differences per season ahead."""
    context = np.array([1, 2, 3, 5, 4, 8], dtype=np.float32)
    mean, quantiles = seasonal_naive(context, horizon=5, season_length=2)
    assert mean.dtype == quantiles.dtype == np.float32
    np.testing.assert_array_equal(mean, [4, 8, 4, 8, 4])
    sigma = np.sqrt((2**2 + 3**2 + 1**2 + 3**2) / 4)  # Differences 2, 3, 1, 3.
    sigma_h = sigma * np.sqrt([1, 1, 2, 2, 3])
    np.testing.assert_allclose(quantiles, mean + Z[:, None] * sigma_h, rtol=1e-6)
    np.testing.assert_array_equal(quantiles[LEVELS.index(0.5)], mean)


def test_seasonal_naive_without_a_full_season_more() -> None:
    """Sigma is 0 without a second season; positions never observed are NaN."""
    mean, quantiles = seasonal_naive(np.array([1, 2, 3], np.float32), 4, 3)
    np.testing.assert_array_equal(mean, [1, 2, 3, 1])
    np.testing.assert_array_equal(quantiles, np.tile(mean, (len(LEVELS), 1)))
    mean, quantiles = seasonal_naive(np.array([1, 2], np.float32), 6, 4)
    np.testing.assert_array_equal(mean, [NAN, NAN, 1, 2, NAN, NAN])
    np.testing.assert_array_equal(quantiles[0], mean)


def test_tails_extrapolate_exponentially() -> None:
    """0.025 and 0.975 continue the outer quantile pairs on log scales."""
    quantiles = (100 + Z[:, None] * [5.0, 10.0]).astype(np.float32)
    left, right = tails(quantiles)
    assert left.dtype == right.dtype == np.float32

    def log(ratio_top: float, ratio_bottom: float) -> float:
        return np.log((ratio_top + TOL) / (ratio_bottom + TOL) + TOL)

    beta_left = (quantiles[1] - quantiles[0]) / log(0.2, 0.1)
    beta_right = (quantiles[-1] - quantiles[-2]) / log(1 - 0.8, 1 - 0.9)
    expected_left = quantiles[1] + beta_left * log(0.025, 0.2)
    expected_right = quantiles[-2] + beta_right * log(1 - 0.8, 1 - 0.975)
    np.testing.assert_allclose(left, expected_left, rtol=1e-6)
    np.testing.assert_allclose(right, expected_right, rtol=1e-6)
    assert (left < quantiles[0]).all()
    assert (right > quantiles[-1]).all()


def test_forecast_lays_out_every_level() -> None:
    """Each window gets 0.025, the predictor's levels and 0.975, in order."""
    rng = np.random.default_rng(0)
    hours = np.arange(100)
    first = (10 + np.sin(2 * np.pi * hours / 24) + rng.normal(size=100)).astype(
        np.float32
    )
    second = np.r_[np.float32(NAN), first[1:60] + 1]
    windows = Windows(
        config="tiny/H/short",
        split="test",
        prediction_length=6,
        freq="H",
        season_length=24,
        values=np.concatenate([first, second]),
        offsets=np.array([0, 100, 160]),
        item_id=np.array(["a", "b"]),
        forecast_start=np.array(["2020-01-05 04:00", "2020-01-03 12:00"]),
    )
    quantiles, mean = forecast(windows)
    assert quantiles.shape == (2, len(QUANTILE_LEVELS), 6)
    assert mean.shape == (2, 6)
    assert (0.025, *LEVELS, 0.975) == QUANTILE_LEVELS
    np.testing.assert_array_equal(quantiles[:, QUANTILE_LEVELS.index(0.5)], mean)
    np.testing.assert_array_equal(mean[0], first[-24:-18])
    assert (np.diff(quantiles, axis=1) > 0).all()
    _, inner = seasonal_naive(impute(second), 6, 24)
    np.testing.assert_array_equal(quantiles[1, 1:-1], inner)
