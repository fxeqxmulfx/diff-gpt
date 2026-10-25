"""
The seasonal naive forecast, and the normal intervals around a forecast.

This is statsforecast 2.1.1's `statsforecast/utils.py` (Apache-2.0, a copy of
the license in leaderboards/LICENSE-statsforecast), cut to `_repeat_val_seas`,
`_ensure_float`, `_seasonal_naive`, `_quantiles`, `_calculate_intervals` and
`_calculate_sigma`, changed in two places besides the style of
leaderboards.vendored:
- `_quantiles` and `_calculate_intervals` take no `distribution` and
  `dist_params`: they keep the default, the normal distribution, the one
  SeasonalNaive's forecast uses. Its `frozen_error_distribution(1.0,
  "normal")` of statsforecast/distributions.py is scipy's `norm(scale=1.0)`,
  written in its place.
- `_calculate_sigma` makes the sum of the squared residuals a float64 before
  it divides it by `n`, as gift-eval's and timebench's environment does: both
  require numpy ~=1.26.0, which divided a float32 sum by the Python int `n`
  in float64. numpy 2's promotion (NEP 50) keeps that division in float32,
  which made sigma, and so every interval of the forecast, a float32.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats


def _repeat_val_seas(season_vals: np.ndarray, h: int) -> np.ndarray:
    repeats = math.ceil(h / season_vals.size)
    return np.tile(season_vals, repeats)[:h]


def _ensure_float(x: np.ndarray) -> np.ndarray:
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float32)
    return x


def _seasonal_naive(
    y: np.ndarray,  # time series
    h: int,  # forecasting horizon
    *,
    fitted: bool,  # fitted values
    season_length: int,  # season length
) -> dict[str, np.ndarray]:
    y = _ensure_float(y)
    n = y.size
    season_vals = np.full(season_length, np.nan, dtype=y.dtype)
    n_available = min(season_length, n)
    if n < season_length:
        # Partial season: align observations to their seasonal positions
        # so that the forecast cycles correctly.
        warnings.warn(
            f"Historical data ({n}) is shorter than season_length "
            f"({season_length}). Forecasts for positions without "
            f"observations will be filled with NaN.",
            stacklevel=1,
        )
        start_idx = season_length - n_available
        season_vals[start_idx:] = y[-n_available:]
    else:
        season_vals[:n_available] = y[-n_available:]
    out = _repeat_val_seas(season_vals=season_vals, h=h)
    fcst = {"mean": out}
    if fitted:
        fitted_vals = np.empty_like(y)
        fitted_vals[:season_length] = np.nan
        if n > season_length:
            fitted_vals[season_length:] = y[: n - season_length]
        fcst["fitted"] = fitted_vals
    return fcst


# Functions used for calculating prediction intervals
def _quantiles(level: np.ndarray) -> np.ndarray:
    p = 0.5 + np.asarray(level) / 200
    return stats.norm(scale=1.0).ppf(p)


def _calculate_intervals(
    out: dict[str, np.ndarray], level: list[int], h: int, sigmah: np.ndarray
) -> dict[str, np.ndarray]:
    z = _quantiles(np.asarray(level))
    zz = np.repeat(z, h)
    zz = zz.reshape(z.shape[0], h)
    lower = out["mean"] - zz * sigmah
    upper = out["mean"] + zz * sigmah
    return {
        **{f"lo-{lv}": lower[i] for i, lv in enumerate(level)},
        **{f"hi-{lv}": upper[i] for i, lv in enumerate(level)},
    }


def _calculate_sigma(residuals: np.ndarray, n: int) -> float:
    if n > 0:
        sigma = np.float64(np.nansum(residuals**2))
        sigma = sigma / n
        sigma = np.sqrt(sigma)
    else:
        sigma = 0
    return sigma
