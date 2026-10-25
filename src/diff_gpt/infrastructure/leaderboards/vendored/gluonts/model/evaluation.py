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
The evaluation of a predictor's forecasts on test data, by gluonts.ev metrics.

This is gluonts 0.15.1's `gluonts/model/evaluation.py` (Apache-2.0, a copy of
the license in src/diff_gpt/infrastructure/leaderboards/LICENSE-gluonts),
changed in one place besides the style of
diff_gpt.infrastructure.leaderboards.vendored:
- `evaluate_forecasts_raw` shows no tqdm progress bar, and leaderboards no
  longer needs tqdm. The bar only counted the forecasts on stderr.
toolz's `first` and `valmap` are written out: `next(iter(...))`, as `first`
is, and a dict comprehension, as `valmap` builds its dict.
"""

from __future__ import annotations

import logging
from collections import ChainMap
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from diff_gpt.infrastructure.leaderboards.vendored.gluonts.ev.ts_stats import (
    seasonal_error,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.itertools import (
    batcher,
    prod,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.time_feature.seasonality import (  # noqa: E501 - the module's name alone is wider than the line.
    get_seasonality,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset import DataEntry
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.split import (
        TestData,
    )
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.ev.metrics import (
        BaseMetricDefinition,
    )
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.model.forecast import (
        Forecast,
    )
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.model.predictor import (
        Predictor,
    )

logger = logging.getLogger(__name__)


@dataclass
class BatchForecast:
    """
    Forecasts whose arrays come with a batch dimension in front.

    Wrapper around ``Forecast`` objects, that adds a batch dimension to arrays
    returned by ``__getitem__``, for compatibility with ``gluonts.ev``.
    """

    forecasts: list[Forecast]
    allow_nan: bool = False

    def __getitem__(self, name: float | str) -> np.ndarray:
        """Return the forecasts' arrays `name`, stacked."""
        values = [forecast[name].T for forecast in self.forecasts]
        res = np.stack(values, axis=0)

        if np.isnan(res).any():
            if not self.allow_nan:
                msg = "Forecast contains NaN values"
                raise ValueError(msg)

            logger.warning("Forecast contains NaN values. Metrics may be incorrect.")

        return res


def _get_data_batch(
    input_batch: list[DataEntry],
    label_batch: list[DataEntry],
    forecast_batch: list[Forecast],
    *,
    seasonality: int | None = None,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
) -> ChainMap:
    label_target = np.stack([label["target"] for label in label_batch], axis=0)
    if mask_invalid_label:
        label_target = np.ma.masked_invalid(label_target)

    other_data = {
        "label": label_target,
    }

    seasonal_error_values = []
    for input_ in input_batch:
        seasonality_entry = seasonality
        if seasonality_entry is None:
            seasonality_entry = get_seasonality(input_["start"].freqstr)
        input_target = input_["target"]
        if mask_invalid_label:
            input_target = np.ma.masked_invalid(input_target)
        seasonal_error_values.append(
            seasonal_error(
                input_target,
                seasonality=seasonality_entry,
                time_axis=-1,
            )
        )
    other_data["seasonal_error"] = np.array(seasonal_error_values)

    return ChainMap(
        other_data, BatchForecast(forecast_batch, allow_nan=allow_nan_forecast)
    )


def evaluate_forecasts_raw(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics: Iterable[BaseMetricDefinition],
    axis: int | tuple | None = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: int | None = None,
) -> dict:
    """
    Evaluate ``forecasts`` against ``test_data``, according to ``metrics``.

    .. note:: This feature is experimental and may be subject to changes.

    The optional ``axis`` arguments controls aggregation of the metrics:
    - ``None`` (default) aggregates across all dimensions
    - ``0`` aggregates across the dataset
    - ``1`` aggregates across the first data dimension (time, in the univariate
      setting)
    - ``2`` aggregates across the second data dimension (time, in the
      multivariate setting)

    Return results as a dictionary.
    """
    label_ndim = next(iter(test_data.label))["target"].ndim

    if label_ndim not in [1, 2]:
        raise AssertionError

    if axis is None:
        axis = tuple(range(label_ndim + 1))
    if isinstance(axis, int):
        axis = (axis,)

    if not all(ax in range(3) for ax in axis):
        raise AssertionError

    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)
        evaluators[evaluator.name] = evaluator

    index_data = []

    input_batches = batcher(test_data.input, batch_size=batch_size)
    label_batches = batcher(test_data.label, batch_size=batch_size)
    forecast_batches = batcher(forecasts, batch_size=batch_size)

    for input_batch, label_batch, forecast_batch in zip(
        input_batches, label_batches, forecast_batches, strict=False
    ):
        if 0 not in axis:
            index_data.extend(
                [(forecast.item_id, forecast.start_date) for forecast in forecast_batch]
            )

        data_batch = _get_data_batch(
            input_batch,
            label_batch,
            forecast_batch,
            seasonality=seasonality,
            mask_invalid_label=mask_invalid_label,
            allow_nan_forecast=allow_nan_forecast,
        )

        for evaluator in evaluators.values():
            evaluator.update(data_batch)

    metrics_values = {
        metric_name: evaluator.get() for metric_name, evaluator in evaluators.items()
    }

    if index_data:
        metrics_values["__index_0"] = index_data

    return metrics_values


def evaluate_forecasts(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics: Iterable[BaseMetricDefinition],
    axis: int | tuple | None = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate ``forecasts`` against ``test_data``, according to ``metrics``.

    .. note:: This feature is experimental and may be subject to changes.

    The optional ``axis`` arguments controls aggregation of the metrics:
    - ``None`` (default) aggregates across all dimensions
    - ``0`` aggregates across the dataset
    - ``1`` aggregates across the first data dimension (time, in the univariate
      setting)
    - ``2`` aggregates across the second data dimension (time, in the
      multivariate setting)

    Return results as a Pandas ``DataFrame``.
    """
    metrics_values = evaluate_forecasts_raw(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=axis,
        batch_size=batch_size,
        mask_invalid_label=mask_invalid_label,
        allow_nan_forecast=allow_nan_forecast,
        seasonality=seasonality,
    )
    index0 = metrics_values.pop("__index_0", None)

    metric_shape = metrics_values[next(iter(metrics_values))].shape
    if metric_shape == ():
        index = [None]
    else:
        index_arrays = np.unravel_index(range(prod(metric_shape)), metric_shape)
        if index0 is not None:
            index0_repeated = np.take(index0, indices=index_arrays[0], axis=0)
            index_arrays = (*zip(*index0_repeated, strict=False), *index_arrays[1:])
        index = pd.MultiIndex.from_arrays(index_arrays)

    flattened_metrics = {
        name: np.ravel(values) for name, values in metrics_values.items()
    }

    return pd.DataFrame(flattened_metrics, index=index)


def evaluate_model(
    model: Predictor,
    *,
    test_data: TestData,
    metrics: Iterable[BaseMetricDefinition],
    axis: int | tuple | None = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate ``model`` when applied to ``test_data``, according to ``metrics``.

    .. note:: This feature is experimental and may be subject to changes.

    The optional ``axis`` arguments controls aggregation of the metrics:
    - ``None`` (default) aggregates across all dimensions
    - ``0`` aggregates across the dataset
    - ``1`` aggregates across the first data dimension (time, in the univariate
      setting)
    - ``2`` aggregates across the second data dimension (time, in the
      multivariate setting)

    Return results as a Pandas ``DataFrame``.
    """
    forecasts = model.predict(test_data.input)

    return evaluate_forecasts(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=axis,
        batch_size=batch_size,
        mask_invalid_label=mask_invalid_label,
        allow_nan_forecast=allow_nan_forecast,
        seasonality=seasonality,
    )
