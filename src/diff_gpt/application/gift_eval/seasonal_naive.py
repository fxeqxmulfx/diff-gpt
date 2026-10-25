"""
Seasonal naive in numpy, as GIFT-Eval's Seasonal_Naive forecasts.

Run from the repository root:
``uv run python -m diff_gpt.interfaces.gift_eval.seasonal_naive --split test``. It
forecasts every exported window of the split into
RUNS/forecasts/seasonal_naive/<split>/, for leaderboards' score.

The leaderboard's Seasonal_Naive is statsforecast's SeasonalNaive behind
gift-eval's notebook predictor
(diff_gpt.infrastructure.leaderboards.gift.official), and this follows it
step by step:

- A context with NaN is filled by gluonts' LastValueImputation: a NaN takes
  the last value before it, and leading NaN the first value; a context that
  is all NaN, or a lone NaN, becomes 0.
- The point forecast repeats the last season. A context shorter than a
  season leaves NaN where no observation falls (statsforecast's partial
  season), which the scorer rejects, as it does the predictor's forecast;
  GIFT-Eval has no such context.
- Quantile q of step j is the point forecast plus Φ⁻¹(q)·sigma·√(⌊j/s⌋ + 1),
  sigma the root mean square of the seasonal differences y[t] - y[t - s], or
  0 if there are none. statsforecast stores the quantiles as float32.
- The predictor gives the quantiles 0.1 to 0.9 only, and gluonts reads MSIS's
  0.025 and 0.975 from them by its exponential tail approximation; those are
  the 0.025 and 0.975 written here.
"""

import time
from typing import TYPE_CHECKING

import numpy as np

from diff_gpt.application.gift_eval.files import (
    QUANTILE_LEVELS,
    forecast_path,
    read_windows,
    windows_paths,
    write_forecasts,
)
from diff_gpt.application.ports import UsageError

if TYPE_CHECKING:
    import argparse

    from diff_gpt.application.gift_eval.files import Windows

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# scipy.stats.norm.ppf(LEVELS): statsforecast's bounds at its levels 20, 40,
# 60 and 80 are the point forecast ∓ norm.ppf(0.5 + level / 200)·sigma_h.
Z = np.array(
    [
        -1.2815515655446004,
        -0.8416212335729143,
        -0.5244005127080407,
        -0.2533471031357997,
        0.0,
        0.2533471031357997,
        0.5244005127080407,
        0.8416212335729143,
        1.2815515655446004,
    ]
)
TOL = 1e-8  # gluonts' ExponentialTailApproximation tol.


def impute(context: np.ndarray) -> np.ndarray:
    """Return `context` with NaN filled as gluonts' LastValueImputation fills it."""
    values = context.copy()
    missing = np.isnan(values)
    if not missing.any():
        return values
    if len(values) == 1 or missing.all():
        values[missing] = 0.0
        return values
    last = np.where(missing, 0, np.arange(len(values)))
    np.maximum.accumulate(last, out=last)
    values = values[last]
    missing = np.isnan(values)  # Only leading NaN are left.
    values[missing] = np.interp(
        np.flatnonzero(missing), np.flatnonzero(~missing), values[~missing]
    )
    return values


def seasonal_naive(
    context: np.ndarray, horizon: int, season_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return statsforecast's SeasonalNaive forecast of a context without NaN.

    That is the point forecast (horizon,) and the quantiles at LEVELS
    (len(LEVELS), horizon), float32 both. sigma follows statsforecast's
    arithmetic under NumPy 1: a float32 sum of squares divided in float64.
    """
    y = np.asarray(context, dtype=np.float32)
    n, s = len(y), season_length
    season = np.full(s, np.nan, dtype=np.float32)
    season[max(s - n, 0) :] = y[-min(s, n) :]
    mean = np.tile(season, -(-horizon // s))[:horizon]
    sigma = 0.0
    if n > s:
        residuals = np.full(n, np.nan, dtype=np.float32)
        residuals[s:] = y[s:] - y[:-s]
        sigma = np.sqrt(np.float64(np.nansum(residuals**2)) / (n - s))
    sigma_h = sigma * np.sqrt(np.floor(np.arange(horizon) / s) + 1)
    return mean, (mean + Z[:, None] * sigma_h).astype(np.float32)


def tails(quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the 0.025 and 0.975 quantiles gluonts extrapolates from `quantiles`.

    `quantiles` are at LEVELS; this is ExponentialTailApproximation's left
    and right tail, in float32 as gluonts computes them from float32
    forecasts under NumPy 1 casting.
    """
    lower, low, high, upper = LEVELS[0], LEVELS[1], LEVELS[-2], LEVELS[-1]
    beta_left = (quantiles[1] - quantiles[0]) / np.float32(
        np.log((low + TOL) / (lower + TOL) + TOL)
    )
    beta_right = (quantiles[-1] - quantiles[-2]) / np.float32(
        np.log((1 - high + TOL) / (1 - upper + TOL) + TOL)
    )
    left = (
        beta_left * np.float32(np.log((QUANTILE_LEVELS[0] + TOL) / (low + TOL) + TOL))
        + quantiles[1]
    )
    right = (
        beta_right
        * np.float32(np.log((1 - high + TOL) / (1 - QUANTILE_LEVELS[-1] + TOL) + TOL))
        + quantiles[-2]
    )
    return left, right


def forecast(windows: Windows) -> tuple[np.ndarray, np.ndarray]:
    """Return the quantiles (N, 11, H) at QUANTILE_LEVELS and the means (N, H)."""
    horizon = windows.prediction_length
    quantiles = np.empty((len(windows), len(QUANTILE_LEVELS), horizon), np.float32)
    mean = np.empty((len(windows), horizon), np.float32)
    for index in range(len(windows)):
        context = impute(windows.context(index))
        mean[index], inner = seasonal_naive(context, horizon, windows.season_length)
        left, right = tails(inner)
        quantiles[index] = np.vstack([left, inner, right])
    return quantiles, mean


def forecast_split(args: argparse.Namespace) -> None:
    """Forecast every window of the split with the seasonal naive model."""
    paths = windows_paths(args.split, args.windows)
    if not paths:
        msg = f"no windows in {args.windows / args.split}"
        raise UsageError(msg)
    for path in paths:
        started = time.perf_counter()
        windows = read_windows(path)
        quantiles, mean = forecast(windows)
        write_forecasts(
            forecast_path(windows, args.model, args.forecasts),
            windows,
            quantiles,
            mean,
        )
        seconds = time.perf_counter() - started
        print(f"{windows.config}: {len(windows)} windows, {seconds:.1f} s", flush=True)
