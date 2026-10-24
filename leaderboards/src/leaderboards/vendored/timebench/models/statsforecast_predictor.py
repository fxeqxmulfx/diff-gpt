"""
StatsForecast model wrappers for TIME benchmark.

Adapted from gift_eval implementation.

This is timebench c633e23's `src/timebench/models/statsforecast_predictor.py`
(MIT, a copy of the license in leaderboards/LICENSE-timebench), cut to
SeasonalNaivePredictor and the samples of its SeasonalNaiveForecast (its
`mean` and `median`, and the Naive classes, are left out), changed in one
place besides the style of leaderboards.vendored:
- `_forecast_single_variate` forecasts the series with statsforecast's
  GroupedArray (leaderboards.vendored.statsforecast.core), and
  `_extract_quantile_forecasts` reads the forecasts by column name from the
  array it returns. timebench hands the series to StatsForecast.forecast as a
  DataFrame (`_create_dataframe`: the series as float32, dated from
  2000-01-01 at the dataset's frequency under the id "series_<i>"), which
  makes the same GroupedArray of it, forecasts it the same way and returns
  the same float32 values in a DataFrame. With the DataFrame go
  `_create_dataframe` and the id it took, and StatsForecast's checks of the
  dates; SeasonalNaivePredictor still takes and keeps `freq`, which only
  those dates read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from leaderboards.vendored.statsforecast.core import GroupedArray
from leaderboards.vendored.statsforecast.models import SeasonalNaive

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from leaderboards.vendored.gluonts.dataset import DataEntry

# Default quantile levels aligned with GluonTS/gift_eval
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def quantile_levels_to_intervals(quantile_levels: list[float]) -> list[int]:
    """
    Convert quantile levels to statsforecast interval levels.

    statsforecast uses symmetric prediction intervals specified by 'level'.
    For example, level=80 gives the 10th and 90th percentiles (lo-80, hi-80).

    Args:
        quantile_levels: List of quantile levels (e.g., [0.1, 0.2, ..., 0.9])

    Returns:
        List of unique interval levels for statsforecast

    """
    intervals = set()
    for q in quantile_levels:
        # Convert quantile to interval: q=0.1 or q=0.9 -> interval=80
        interval = round(200 * (max(q, 1 - q) - 0.5))
        if interval > 0:  # Skip 0 (mean)
            intervals.add(interval)
    return sorted(intervals)


def get_quantile_column_name(model_name: str, quantile_level: float) -> str:
    """
    Get the statsforecast column name for a given quantile level.

    Args:
        model_name: Name of the model (e.g., 'SeasonalNaive')
        quantile_level: Quantile level (e.g., 0.1, 0.9)

    Returns:
        Column name (e.g., 'SeasonalNaive-lo-80' for q=0.1)

    """
    if quantile_level == 0.5:  # noqa: PLR2004 - the median's level.
        return model_name  # Mean/median

    interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
    side = "hi" if quantile_level > 0.5 else "lo"  # noqa: PLR2004 - the median's level.
    return f"{model_name}-{side}-{interval}"


class SeasonalNaiveForecast:
    """
    Wrapper for SeasonalNaive forecast results with true probabilistic output.

    Uses statsforecast's prediction intervals to generate samples that
    reflect actual forecast uncertainty, rather than replicating point forecasts.

    Attributes:
        samples: Forecast samples with shape (num_samples, num_variates, pred_len)

    """

    def __init__(
        self,
        quantile_forecasts: dict,
        quantile_levels: list[float],
        num_samples: int = 100,
    ) -> None:
        """
        Initialize forecast wrapper from quantile predictions.

        Args:
            quantile_forecasts: Dict mapping quantile levels to forecast arrays
                               Each array has shape (num_variates, pred_len)
            quantile_levels: List of quantile levels used
            num_samples: Number of samples to generate

        """
        self.quantile_levels = sorted(quantile_levels)
        self.quantile_forecasts = quantile_forecasts

        # Get shape from any quantile forecast
        sample_forecast = next(iter(quantile_forecasts.values()))
        if sample_forecast.ndim == 1:
            sample_forecast = sample_forecast[np.newaxis, :]
        num_variates, pred_len = sample_forecast.shape

        # Generate samples from quantile forecasts
        self._samples = self._generate_samples_from_quantiles(
            quantile_forecasts, quantile_levels, num_samples, num_variates, pred_len
        )

    def _generate_samples_from_quantiles(
        self,
        quantile_forecasts: dict,
        quantile_levels: list[float],
        num_samples: int,
        num_variates: int,
        pred_len: int,
    ) -> np.ndarray:
        """
        Generate samples from quantile forecasts using inverse CDF sampling.

        This creates samples that match the distribution implied by the quantile
        forecasts, providing true probabilistic diversity.
        """
        samples = np.zeros((num_samples, num_variates, pred_len))

        # Sort quantile levels and get corresponding forecasts
        sorted_levels = sorted(quantile_levels)

        # Build quantile values array: (num_quantiles, num_variates, pred_len)
        quantile_values = []
        for q in sorted_levels:
            qf = quantile_forecasts[q]
            if qf.ndim == 1:
                qf = qf[np.newaxis, :]
            quantile_values.append(qf)
        quantile_values = np.stack(quantile_values, axis=0)

        # Generate uniform random values for inverse CDF sampling
        uniform_samples = np.random.uniform(0, 1, size=(num_samples,))  # noqa: NPY002 - TIME draws from numpy's global generator.

        for i, u in enumerate(uniform_samples):
            # For each sample, interpolate between quantiles
            # Find which quantile interval this uniform value falls into
            for v in range(num_variates):
                for t in range(pred_len):
                    qvals = quantile_values[:, v, t]
                    samples[i, v, t] = np.interp(u, sorted_levels, qvals)

        return samples

    @property
    def samples(self) -> np.ndarray:
        """Return forecast samples with shape (num_samples, num_variates, pred_len)."""
        return self._samples


class SeasonalNaivePredictor:
    """
    Predictor wrapping statsforecast SeasonalNaive model with prediction intervals.

    Unlike the simple point forecast replication, this predictor uses
    statsforecast's built-in prediction interval estimation to generate
    true probabilistic forecasts.

    The Seasonal Naive method forecasts the value from the same season
    in the previous seasonal cycle. Prediction intervals are estimated
    based on historical seasonal errors.
    """

    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        freq: str,
        num_samples: int = 100,
        quantile_levels: list[float] | None = None,
    ) -> None:
        """
        Initialize SeasonalNaive predictor.

        Args:
            prediction_length: Number of time steps to forecast
            season_length: Seasonal period length (e.g., 7 for weekly, 24 for daily)
            freq: Frequency string (e.g., 'D', 'H', '15T')
            num_samples: Number of samples to generate
            quantile_levels: List of quantile levels for prediction intervals.
                           Defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        """
        self.prediction_length = prediction_length
        self.season_length = season_length
        self.freq = freq
        self.num_samples = num_samples
        self.quantile_levels = quantile_levels or DEFAULT_QUANTILE_LEVELS

        # Convert quantile levels to statsforecast intervals
        self.intervals = quantile_levels_to_intervals(self.quantile_levels)

        # Model name for column lookup
        self.model_name = "SeasonalNaive"

    def _extract_quantile_forecasts(
        self,
        result: dict,
    ) -> dict:
        """
        Extract quantile forecasts from statsforecast's result.

        Args:
            result: The forecasts and their column names, from statsforecast

        Returns:
            Dict mapping quantile levels to forecast arrays

        """
        quantile_forecasts = {}

        for q in self.quantile_levels:
            col_name = get_quantile_column_name(self.model_name, q)
            if col_name in result["cols"]:
                quantile_forecasts[q] = result["forecasts"][
                    :, result["cols"].index(col_name)
                ]
            elif q == 0.5 and self.model_name in result["cols"]:  # noqa: PLR2004 - the median's level.
                # Use mean as 0.5 quantile
                quantile_forecasts[q] = result["forecasts"][
                    :, result["cols"].index(self.model_name)
                ]

        return quantile_forecasts

    def _forecast_single_variate(self, history: np.ndarray) -> dict:
        """
        Generate quantile forecasts for a single univariate time series.

        Args:
            history: Historical values with shape (context_len,)

        Returns:
            Dict mapping quantile levels to forecast arrays of shape (pred_len,)

        """
        # Handle NaN values by forward filling
        history = history.copy()
        if np.isnan(history).any():
            # Forward fill NaN values
            mask = np.isnan(history)
            idx = np.where(~mask, np.arange(len(mask)), 0)
            np.maximum.accumulate(idx, out=idx)
            history = history[idx]
            # If still has NaN (e.g., leading NaN), fill with first valid value
            if np.isnan(history).any():
                first_valid = (
                    history[~np.isnan(history)][0] if (~np.isnan(history)).any() else 0
                )
                history = np.nan_to_num(history, nan=first_valid)

        # The series as float32, as StatsForecast makes it of the DataFrame
        ga = GroupedArray(history.astype(np.float32), np.array([0, len(history)]))

        # Generate forecasts with prediction intervals
        result = ga.forecast(
            models=[SeasonalNaive(season_length=self.season_length)],
            h=self.prediction_length,
            level=self.intervals,
        )

        # Extract quantile forecasts
        return self._extract_quantile_forecasts(result)

    def predict(
        self, dataset_input: Iterable[DataEntry]
    ) -> Iterator[SeasonalNaiveForecast]:
        """
        Generate forecasts for all series in the dataset.

        Args:
            dataset_input: Iterable of data entries, each containing:
                - "target": Historical values with shape (context_len,) for
                  univariate or (num_variates, context_len) for multivariate

        Yields:
            SeasonalNaiveForecast objects with probabilistic predictions

        """
        for entry in dataset_input:
            target = entry["target"]

            # Handle univariate case
            if target.ndim == 1:
                quantile_forecasts = self._forecast_single_variate(target)
                yield SeasonalNaiveForecast(
                    quantile_forecasts, self.quantile_levels, self.num_samples
                )

            # Handle multivariate case - process each variate independently
            else:
                num_variates = target.shape[0]

                # Collect quantile forecasts for all variates
                all_quantile_forecasts = {q: [] for q in self.quantile_levels}

                for v in range(num_variates):
                    qf = self._forecast_single_variate(target[v])
                    for q in self.quantile_levels:
                        if q in qf:
                            all_quantile_forecasts[q].append(qf[q])

                # Stack variates: (num_variates, pred_len)
                combined_qf = {}
                for q in self.quantile_levels:
                    if all_quantile_forecasts[q]:
                        combined_qf[q] = np.stack(all_quantile_forecasts[q], axis=0)

                yield SeasonalNaiveForecast(
                    combined_qf, self.quantile_levels, self.num_samples
                )
