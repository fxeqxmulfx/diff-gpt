"""
statsforecast's SeasonalNaive: the last season repeated, with normal intervals.

This is statsforecast 2.1.1's `statsforecast/models.py` (Apache-2.0, a copy of
the license in src/diff_gpt/infrastructure/leaderboards/LICENSE-statsforecast),
cut to SeasonalNaive's `__init__` and `forecast` and the `__repr__` of its base
_TS, which names the columns of its forecasts; changed in three places besides
the style of diff_gpt.infrastructure.leaderboards.vendored:
- SeasonalNaive has no base class and holds `__repr__` itself: of _TS, the
  base of statsforecast's models, only `__repr__` runs for these calls. Its
  `uses_exog`, `new` and conformal methods serve StatsForecast's checks, the
  fits and `prediction_intervals`, all left out.
- SeasonalNaive takes no `prediction_intervals`, and its forecast always has
  the native intervals: statsforecast computes conformal ones in their place
  only when `prediction_intervals` is set, which leaderboards never sets.
- `forecast` takes no `X` and `X_future`, which SeasonalNaive ignores, and no
  `fitted`, the in-sample values with their intervals, which leaderboards
  never asks for: statsforecast's GroupedArray passes fitted=False, so the
  fitted values are computed for the intervals only, as `fitted or (level is
  not None)` has it.
"""

from __future__ import annotations

import numpy as np

from diff_gpt.infrastructure.leaderboards.vendored.statsforecast.utils import (
    _calculate_intervals,
    _calculate_sigma,
    _ensure_float,
    _seasonal_naive,
)


class SeasonalNaive:
    """
    Seasonal naive model.

    A method similar to the naive, but uses the last known observation of the
    same period (e.g. the same month of the previous year) in order to capture
    seasonal variations.

    References:
        - [Rob J. Hyndman and George Athanasopoulos (2018). "forecasting
          principles and practice, Simple
          Methods"](https://otexts.com/fpp3/simple-methods.html#seasonal-na%C3%AFve-method).

    """

    def __init__(self, season_length: int, alias: str = "SeasonalNaive") -> None:
        """
        Repeat the last `season_length` observations.

        Args:
            season_length (int): Number of observations per unit of time. Ex: 24
                Hourly data.
            alias (str): Custom name of the model.

        """
        self.season_length = season_length
        self.alias = alias

    def __repr__(self) -> str:
        """Return the alias: the name of the forecast's columns."""
        return self.alias

    def forecast(
        self,
        y: np.ndarray,
        h: int,
        level: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Memory Efficient SeasonalNaive predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            level (List[float]): Confidence levels (0-100) for prediction
                intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and
                `level_*` for probabilistic predictions.

        """
        y = _ensure_float(y)
        out = _seasonal_naive(
            y=y,
            h=h,
            fitted=level is not None,
            season_length=self.season_length,
        )
        res = {"mean": out["mean"]}
        if level is not None:
            level = sorted(level)
            k = np.floor(np.arange(h) / self.season_length)
            residuals = y - out["fitted"]
            sigma = _calculate_sigma(residuals, len(y) - self.season_length)
            sigmah = sigma * np.sqrt(k + 1)
            pred_int = _calculate_intervals(out, level, h, sigmah)
            res = {**res, **pred_int}
        return res
