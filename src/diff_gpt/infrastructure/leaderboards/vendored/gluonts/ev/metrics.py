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
gluonts.ev's metrics: a statistic aggregated, or metrics combined.

This is gluonts 0.15.1's `gluonts/ev/metrics.py` (Apache-2.0, a copy of the
license in src/diff_gpt/infrastructure/leaderboards/LICENSE-gluonts), cut to
the metrics leaderboards computes (MSE, MAE, MASE, MAPE, sMAPE, MSIS, RMSE,
NRMSE, ND and the mean weighted quantile loss) and the ones they are made of:
Metric without `update_all`, BaseMetricDefinition without `+` and `add`.
Unchanged but for the style of diff_gpt.infrastructure.leaderboards.vendored.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Self

import numpy as np

from diff_gpt.infrastructure.leaderboards.vendored.gluonts.ev.aggregations import (
    Mean,
    Sum,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.ev.stats import (
    absolute_error,
    absolute_label,
    absolute_percentage_error,
    absolute_scaled_error,
    quantile_loss,
    scaled_interval_score,
    squared_error,
    symmetric_absolute_percentage_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.ev.aggregations import (
        Aggregation,
    )


@dataclass
class Metric:
    """A metric, updated batch by batch."""

    name: str

    def update(self, data: Mapping[str, np.ndarray]) -> Self:
        """Update metric using a single data instance."""
        raise NotImplementedError

    def get(self) -> np.ndarray:
        """Return the metric."""
        raise NotImplementedError


@dataclass
class DirectMetric(Metric):
    """A Metric which uses a single function and aggregation strategy."""

    stat: Callable
    aggregate: Aggregation

    def update(self, data: Mapping[str, np.ndarray]) -> Self:
        """Aggregate the statistic of `data`."""
        self.aggregate.step(self.stat(data))

        return self

    def get(self) -> np.ndarray:
        """Return the aggregate."""
        return self.aggregate.get()


@dataclass
class DerivedMetric(Metric):
    """
    A Metric that is computed using other metrics.

    A derived metric updates multiple, simpler metrics independently and in the
    end combines their results as defined in `post_process`.
    """

    metrics: dict[str, Metric]
    post_process: Callable

    def update(self, data: Mapping[str, np.ndarray]) -> Self:
        """Update each of the metrics."""
        for evaluator in self.metrics.values():
            evaluator.update(data)

        return self

    def get(self) -> np.ndarray:
        """Return `post_process` of the metrics."""
        return self.post_process(
            **{name: evaluator.get() for name, evaluator in self.metrics.items()}
        )


class BaseMetricDefinition:
    """The definition of a metric: called with an axis, it makes the metric."""

    def __call__(self, axis: int | tuple | None) -> Metric:
        """Make the metric, aggregated over `axis`."""
        raise NotImplementedError


class MeanAbsoluteLabel(BaseMetricDefinition):
    """The mean absolute label."""

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name="mean_absolute_label",
            stat=absolute_label,
            aggregate=Mean(axis=axis),
        )


mean_absolute_label = MeanAbsoluteLabel()


class SumAbsoluteLabel(BaseMetricDefinition):
    """The sum of the absolute labels."""

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name="sum_absolute_label",
            stat=absolute_label,
            aggregate=Sum(axis=axis),
        )


sum_absolute_label = SumAbsoluteLabel()


@dataclass
class SumAbsoluteError(BaseMetricDefinition):
    """The sum of the absolute errors."""

    forecast_type: str = "0.5"

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name=f"sum_absolute_error[{self.forecast_type}]",
            stat=partial(absolute_error, forecast_type=self.forecast_type),
            aggregate=Sum(axis=axis),
        )


@dataclass
class MAE(BaseMetricDefinition):
    """Mean Absolute Error."""

    forecast_type: str = "0.5"

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name=f"MAE[{self.forecast_type}]",
            stat=partial(absolute_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MSE(BaseMetricDefinition):
    """Mean Squared Error."""

    forecast_type: str = "mean"

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name=f"MSE[{self.forecast_type}]",
            stat=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SumQuantileLoss(BaseMetricDefinition):
    """The sum of the quantile losses of quantile `q`."""

    q: float

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name=f"sum_quantile_loss[{self.q}]",
            stat=partial(quantile_loss, q=self.q),
            aggregate=Sum(axis=axis),
        )


@dataclass
class MAPE(BaseMetricDefinition):
    """Mean Absolute Percentage Error."""

    forecast_type: str = "0.5"

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name=f"MAPE[{self.forecast_type}]",
            stat=partial(absolute_percentage_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class SMAPE(BaseMetricDefinition):
    """Symmetric Mean Absolute Percentage Error."""

    forecast_type: str = "0.5"

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name=f"sMAPE[{self.forecast_type}]",
            stat=partial(
                symmetric_absolute_percentage_error,
                forecast_type=self.forecast_type,
            ),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MSIS(BaseMetricDefinition):
    """Mean Scaled Interval Score."""

    alpha: float = 0.05

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name="MSIS",
            stat=partial(scaled_interval_score, alpha=self.alpha),
            aggregate=Mean(axis=axis),
        )


@dataclass
class MASE(BaseMetricDefinition):
    """Mean Absolute Scaled Error."""

    forecast_type: str = "0.5"

    def __call__(self, axis: int | tuple | None = None) -> DirectMetric:
        """Make the metric, aggregated over `axis`."""
        return DirectMetric(
            name=f"MASE[{self.forecast_type}]",
            stat=partial(absolute_scaled_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class ND(BaseMetricDefinition):
    """Normalized Deviation."""

    forecast_type: str = "0.5"

    @staticmethod
    def normalized_deviation(
        sum_absolute_error: np.ndarray, sum_absolute_label: np.ndarray
    ) -> np.ndarray:
        """Return the sum of the absolute errors over that of the labels."""
        return sum_absolute_error / sum_absolute_label

    def __call__(self, axis: int | tuple | None = None) -> DerivedMetric:
        """Make the metric, aggregated over `axis`."""
        return DerivedMetric(
            name=f"ND[{self.forecast_type}]",
            metrics={
                "sum_absolute_error": SumAbsoluteError(
                    forecast_type=self.forecast_type
                )(axis=axis),
                "sum_absolute_label": sum_absolute_label(axis=axis),
            },
            post_process=self.normalized_deviation,
        )


@dataclass
class RMSE(BaseMetricDefinition):
    """Root Mean Squared Error."""

    forecast_type: str = "mean"

    @staticmethod
    def root_mean_squared_error(mean_squared_error: np.ndarray) -> np.ndarray:
        """Return the root of the mean squared error."""
        return np.sqrt(mean_squared_error)

    def __call__(self, axis: int | tuple | None = None) -> DerivedMetric:
        """Make the metric, aggregated over `axis`."""
        return DerivedMetric(
            name=f"RMSE[{self.forecast_type}]",
            metrics={
                "mean_squared_error": MSE(forecast_type=self.forecast_type)(axis=axis)
            },
            post_process=self.root_mean_squared_error,
        )


@dataclass
class NRMSE(BaseMetricDefinition):
    """RMSE, normalized by the mean absolute label."""

    forecast_type: str = "mean"

    @staticmethod
    def normalize_root_mean_squared_error(
        root_mean_squared_error: np.ndarray, mean_absolute_label: np.ndarray
    ) -> np.ndarray:
        """Return the RMSE over the mean absolute label."""
        return root_mean_squared_error / mean_absolute_label

    def __call__(self, axis: int | tuple | None = None) -> DerivedMetric:
        """Make the metric, aggregated over `axis`."""
        return DerivedMetric(
            name=f"NRMSE[{self.forecast_type}]",
            metrics={
                "root_mean_squared_error": RMSE(forecast_type=self.forecast_type)(
                    axis=axis
                ),
                "mean_absolute_label": mean_absolute_label(axis=axis),
            },
            post_process=self.normalize_root_mean_squared_error,
        )


@dataclass
class WeightedSumQuantileLoss(BaseMetricDefinition):
    """The sum of the quantile losses of `q` over that of the absolute labels."""

    q: float

    @staticmethod
    def weight_sum_quantile_loss(
        sum_quantile_loss: np.ndarray, sum_absolute_label: np.ndarray
    ) -> np.ndarray:
        """Return the sum of the quantile losses over that of the labels."""
        return sum_quantile_loss / sum_absolute_label

    def __call__(self, axis: int | tuple | None = None) -> DerivedMetric:
        """Make the metric, aggregated over `axis`."""
        return DerivedMetric(
            name=f"weighted_sum_quantile_loss[{self.q}]",
            metrics={
                "sum_quantile_loss": SumQuantileLoss(q=self.q)(axis=axis),
                "sum_absolute_label": sum_absolute_label(axis=axis),
            },
            post_process=self.weight_sum_quantile_loss,
        )


@dataclass
class MeanWeightedSumQuantileLoss(BaseMetricDefinition):
    """The mean over `quantile_levels` of the weighted sum quantile loss."""

    quantile_levels: Collection[float]

    @staticmethod
    def mean(**quantile_losses: np.ndarray) -> np.ndarray:
        """Return the mean of the losses."""
        stacked_quantile_losses = np.stack(
            list(quantile_losses.values()),
            axis=0,
        )
        return np.mean(stacked_quantile_losses, axis=0)

    def __call__(self, axis: int | tuple | None = None) -> DerivedMetric:
        """Make the metric, aggregated over `axis`."""
        return DerivedMetric(
            name="mean_weighted_sum_quantile_loss",
            metrics={
                f"quantile_loss[{q}]": WeightedSumQuantileLoss(q=q)(axis=axis)
                for q in self.quantile_levels
            },
            post_process=self.mean,
        )
