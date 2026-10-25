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
The base class of a predictor: a forecast for each entry of a dataset.

This is gluonts 0.15.1's `gluonts/model/predictor.py` (Apache-2.0, a copy of
the license in leaderboards/LICENSE-gluonts), cut to Predictor's `__init__`
and `predict`: its class attribute `__version__` (gluonts' version) and its
serialization are left out. Unchanged but for the style of
leaderboards.vendored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from leaderboards.vendored.gluonts.dataset import Dataset
    from leaderboards.vendored.gluonts.model.forecast import Forecast


class Predictor:
    """
    Abstract class representing predictor objects.

    Parameters
    ----------
    prediction_length
        Prediction horizon.

    """

    def __init__(self, prediction_length: int, lead_time: int = 0) -> None:
        """Forecast `prediction_length` steps, `lead_time` steps ahead."""
        if not prediction_length > 0:
            msg = "The value of `prediction_length` should be > 0"
            raise AssertionError(msg)
        if not lead_time >= 0:
            msg = "The value of `lead_time` should be >= 0"
            raise AssertionError(msg)

        self.prediction_length = prediction_length
        self.lead_time = lead_time

    def predict(self, dataset: Dataset, **kwargs: object) -> Iterator[Forecast]:
        """
        Compute forecasts for the time series in the provided dataset.

        This method is not implemented in this abstract class; please
        use one of the subclasses.

        Parameters
        ----------
        dataset
            The dataset containing the time series to predict.
        kwargs
            The subclass' options.

        Returns
        -------
        Iterator[Forecast]
            Iterator over the forecasts, in the same order as the dataset
            iterable was provided.

        """
        raise NotImplementedError
