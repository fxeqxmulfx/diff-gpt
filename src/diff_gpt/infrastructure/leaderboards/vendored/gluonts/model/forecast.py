# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
A forecast as quantiles and mean, and the quantile between or beyond them.

This is gluonts 0.15.1's `gluonts/model/forecast.py` (Apache-2.0, a copy of
the license in src/diff_gpt/infrastructure/leaderboards/LICENSE-gluonts), cut
to `_linear_interpolation`, ExponentialTailApproximation, Quantile, Forecast's
`mean`, `quantile`, `median` and `[]` (its `index` and the `_index` that caches
it left out), and QuantileForecast's `__init__`, `quantile` and `mean`, changed
in two places besides the style of
diff_gpt.infrastructure.leaderboards.vendored:
- Quantile is a plain dataclass, where gluonts makes it a pydantic one.
  `__post_init__` keeps pydantic's one check, that the value is in [0, 1],
  and raises a ValueError, as pydantic's ValidationError is one, so that
  `from_str` still turns to the "p10" form when a name is no level. pydantic
  would also convert the value to a float: left out, since `from_float` and
  `from_str`, the only callers, pass a float already.
- ExponentialTailApproximation takes its logarithms as Python floats, so that
  the tails of float32 quantiles are float32, as gift-eval's and timebench's
  environment computes them: both require numpy ~=1.26.0, which cast the
  numpy float64 of `np.log` to the quantiles' float32. numpy 2's promotion
  (NEP 50) makes those tails float64, which moved the quantiles extrapolated
  beyond the forecast's levels, MSIS's 0.025 and 0.975.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _linear_interpolation(
    xs: list[float], ys: list[np.ndarray], x: float
) -> np.ndarray:
    if sorted(xs) != xs:
        raise AssertionError
    if len(xs) != len(ys):
        raise AssertionError
    if not len(xs) >= 2:  # noqa: PLR2004 - two points to interpolate between.
        raise AssertionError

    if x < xs[0]:
        idx = 1
    elif x > xs[-1]:
        idx = len(xs) - 1
    else:
        idx = next(i for i, x_ in enumerate(xs) if x <= x_)

    x_low, x_high = xs[idx - 1 : idx + 1]
    y_low, y_high = ys[idx - 1 : idx + 1]

    dx = x_high - x_low
    dy = y_high - y_low

    return y_low + (x - x_low) / dx * dy


class ExponentialTailApproximation:
    """
    Approximate function on tails based on knots, and infer at a query point.

    Can be used for either interpolation or extrapolation on tails.

    Parameters
    ----------
    x_coord
        x-coordinates of the data points must be in increasing order.
    y_coord
        y-coordinates of the data points - may be a higher numpy array.
    tol
        tolerance when performing the division and computing the log in the
        exponential extrapolation.

    """

    def __init__(
        self,
        x_coord: list[float],
        y_coord: list[np.ndarray],
        tol: float = 1e-8,
    ) -> None:
        """Fit the two tails to the knots."""
        self.x_coord = x_coord
        if sorted(self.x_coord) != self.x_coord:
            raise AssertionError
        self.y_coord = y_coord
        self.num_points = len(self.x_coord)
        if not self.num_points >= 2:  # noqa: PLR2004 - two knots per tail.
            msg = "Need at least two points for exponential approximation."
            raise AssertionError(msg)
        self.tol = tol
        (
            self.beta_inv_left,
            self.beta_inv_right,
        ) = self.init_exponential_tail_weights()

    def init_exponential_tail_weights(self) -> tuple[float, float]:
        """
        Initialize the weight of exponentially decaying tail functions.

        Based on two extreme points on the left and right, respectively.

        Returns
        -------
        Tuple
            beta coefficient for left and right tails.

        """
        q_log_diff = float(
            np.log(
                (self.x_coord[1] + self.tol) / (self.x_coord[0] + self.tol) + self.tol
            )
        )
        y_diff_left = self.y_coord[1] - self.y_coord[0]
        beta_inv_left = y_diff_left / q_log_diff

        z_log_diff = float(
            np.log(
                (1 - self.x_coord[-2] + self.tol) / (1 - self.x_coord[-1] + self.tol)
                + self.tol
            )
        )  # where z = 1/(1-q)
        y_diff_right = self.y_coord[-1] - self.y_coord[-2]
        beta_inv_right = y_diff_right / z_log_diff

        return beta_inv_left, beta_inv_right

    def left(self, x: float) -> np.ndarray:
        """
        Return the inference made on exponentially decaying tail functions.

        For left tail, x = exp(beta * (q - alpha))
        For right tail, x = 1 - exp(-beta * (q - alpha))

        E.g. for x = self.x_coord[0] or self.x_coord[1], return value is
        exactly self.y_coord[0] or self.y_coord[1], respectively.

        Parameters
        ----------
        x
            x-coordinate to evaluate the right tail.

        """
        return (
            self.beta_inv_left
            * float(np.log((x + self.tol) / (self.x_coord[1] + self.tol) + self.tol))
        ) + self.y_coord[1]

    def right(self, x: float) -> np.ndarray:
        """
        Return the inference made on exponentially decaying tail functions.

        For left tail, x = exp(beta * (q - alpha))
        For right tail, x = 1 - exp(-beta * (q - alpha))

        E.g. for x = self.x_coord[-1] or self.x_coord[-2] ,
        return value is exactly self.y_coord[-1]
        or self.y_coord[-2] respectively.

        Parameters
        ----------
        x
            x-coordinate to evaluate the right tail.

        """
        return (
            self.beta_inv_right
            * float(
                np.log(
                    (1 - self.x_coord[-2] + self.tol) / (1 - x + self.tol) + self.tol
                )
            )
        ) + self.y_coord[-2]

    def tail_range(
        self, default_left_tail: float = 0.1, default_right_tail: float = 0.9
    ) -> tuple[float, float]:
        """Return an effective range of left and right tails."""
        left_tail = max(
            self.x_coord[0],
            min(self.x_coord[1], default_left_tail),
        )
        right_tail = min(
            self.x_coord[-1],
            max(self.x_coord[-2], default_right_tail),
        )
        return left_tail, right_tail


@dataclass
class Quantile:
    """A quantile level: its value in [0, 1], and its name."""

    value: float
    name: str

    def __post_init__(self) -> None:
        """Check that the value is in [0, 1]."""
        if not 0.0 <= self.value <= 1.0:
            msg = f"quantile level {self.value} is not in [0, 1]"
            raise ValueError(msg)

    @classmethod
    def from_float(cls, quantile: float) -> Quantile:
        """Return level `quantile`, named by its string."""
        if not isinstance(quantile, float):
            raise AssertionError  # noqa: TRY004 - upstream's assert.
        return cls(value=quantile, name=str(quantile))

    @classmethod
    def from_str(cls, quantile: str) -> Quantile:
        """Return the level named `quantile`: "0.1", say, or "p10"."""
        if not isinstance(quantile, str):
            raise AssertionError  # noqa: TRY004 - upstream's assert.

        try:
            return cls(value=float(quantile), name=quantile)
        except ValueError as err:
            m = re.match(r"^p(\d+)$", quantile)

            if m is None:
                msg = (
                    'Quantile string should be of the form "p10", "p50", ...'
                    f' or "0.1", "0.5", ... but found {quantile}'
                )
                raise ValueError(msg) from err

            return cls.from_float(float(m.group(1)) / 100)

    @classmethod
    def parse(cls, quantile: Quantile | float | str) -> Quantile:
        """
        Produce equivalent float and string representation of a quantile level.

        >>> Quantile.parse(0.1)
        Quantile(value=0.1, name='0.1')

        >>> Quantile.parse('0.2')
        Quantile(value=0.2, name='0.2')

        >>> Quantile.parse('0.20')
        Quantile(value=0.2, name='0.20')

        >>> Quantile.parse('p99')
        Quantile(value=0.99, name='0.99')

        Parameters
        ----------
        quantile
            Quantile, can be a float a str representing a float e.g. '0.1' or a
            quantile string of the form 'p10'.

        Returns
        -------
        Quantile
            A tuple containing both a float and a string representation of the
            input quantile level.

        """
        if isinstance(quantile, Quantile):
            return quantile

        if isinstance(quantile, float):
            return cls.from_float(quantile)

        return cls.from_str(quantile)

    def __str__(self) -> str:
        """Return the name."""
        return self.name


class Forecast:
    """Abstract class representing predictions."""

    start_date: pd.Period
    item_id: str | None
    info: dict | None
    prediction_length: int

    @property
    def mean(self) -> np.ndarray:
        """The mean across the prediction range."""
        raise NotImplementedError

    def quantile(self, q: float | str) -> np.ndarray:
        """
        Compute a quantile from the predicted distribution.

        Parameters
        ----------
        q
            Quantile to compute.

        Returns
        -------
        numpy.ndarray
            Value of the quantile across the prediction range.

        """
        raise NotImplementedError

    @property
    def median(self) -> np.ndarray:
        """The median across the prediction range."""
        return self.quantile(0.5)

    def __getitem__(self, name: float | str) -> np.ndarray:
        """Return the mean, the median, or the quantile `name`."""
        if name == "mean":
            return self.mean
        if name == "median":
            return self.median

        return self.quantile(name)


class QuantileForecast(Forecast):
    """
    A Forecast that contains arrays (i.e. time series) for quantiles and mean.

    Parameters
    ----------
    forecast_arrays
        An array of forecasts
    start_date
        start of the forecast
    forecast_keys
        A list of quantiles of the form '0.1', '0.9', etc.,
        and potentially 'mean'. Each entry corresponds to one array in
        forecast_arrays.
    item_id
        Identifier of the item being forecasted.
    info
        Additional information that the forecaster may provide e.g. estimated
        parameters, number of iterations ran etc.

    """

    def __init__(
        self,
        forecast_arrays: np.ndarray,
        start_date: pd.Period,
        forecast_keys: list[str],
        item_id: str | None = None,
        info: dict | None = None,
    ) -> None:
        """Hold one array of `forecast_arrays` per key of `forecast_keys`."""
        self.forecast_array = forecast_arrays
        if not isinstance(start_date, pd.Period):
            msg = "start_date should be a pandas Period object"
            raise AssertionError(msg)  # noqa: TRY004 - upstream's assert.
        self.start_date = start_date

        # normalize keys
        self.forecast_keys = [
            Quantile.from_str(key).name if key != "mean" else key
            for key in forecast_keys
        ]
        self.item_id = item_id
        self.info = info
        self._dim: int | None = None

        shape = self.forecast_array.shape
        if shape[0] != len(self.forecast_keys):
            msg = (
                f"The forecast_array (shape={shape} should have the same "
                f"length as the forecast_keys (len={len(self.forecast_keys)})."
            )
            raise AssertionError(msg)
        self.prediction_length = shape[1]
        self._forecast_dict = {
            k: self.forecast_array[i] for i, k in enumerate(self.forecast_keys)
        }
        self._nan_out = np.array([np.nan] * self.prediction_length)

    def quantile(self, inference_quantile: float | str) -> np.ndarray:
        """Return a stored quantile, or one between or beyond the stored ones."""
        sorted_forecast_dict = dict(sorted(self._forecast_dict.items()))
        sorted_forecast_dict.pop("mean", None)
        quantiles = [float(q) for q in sorted_forecast_dict]
        quantile_predictions = list(sorted_forecast_dict.values())

        inference_quantile = Quantile.parse(inference_quantile).value

        if len(quantiles) < 2 or inference_quantile in quantiles:  # noqa: PLR2004 - two to interpolate or extrapolate from.
            q_str = Quantile.parse(inference_quantile).name
            return self._forecast_dict.get(q_str, self._nan_out)

        exp_tail_approximation = ExponentialTailApproximation(
            quantiles, quantile_predictions
        )
        # The effective range of left, right tails varies over tail
        # approximation class
        (
            left_tail_quantile,
            right_tail_quantile,
        ) = exp_tail_approximation.tail_range()

        if inference_quantile <= left_tail_quantile:
            return exp_tail_approximation.left(inference_quantile)
        if inference_quantile >= right_tail_quantile:
            return exp_tail_approximation.right(inference_quantile)
        return _linear_interpolation(
            quantiles,
            quantile_predictions,
            inference_quantile,
        )

    @property
    def mean(self) -> np.ndarray:
        """Forecast mean."""
        if "mean" in self._forecast_dict:
            return self._forecast_dict["mean"]
        logger.warning(
            "The mean prediction is not stored in the forecast data; "
            "the median is being returned instead. "
            "This behaviour may change in the future."
        )
        return self.quantile("p50")
