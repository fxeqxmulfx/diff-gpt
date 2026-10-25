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
The processing of a dataset entry: its start a period, its series numpy arrays.

This is gluonts 0.15.1's `gluonts/dataset/common.py` (Apache-2.0, a copy of the
license in src/diff_gpt/infrastructure/leaderboards/LICENSE-gluonts), cut to
ProcessDataEntry, the fields it processes and `_as_period`, changed in one
place besides the style of diff_gpt.infrastructure.leaderboards.vendored:
- ProcessStartField is a plain class, where gluonts makes it a pydantic
  model. pydantic only checks and converts its arguments, which
  ProcessDataEntry passes as their own types already: the frequency string of
  a dataset, False and "start".
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.field_names import (
    FieldName,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.exceptions import (
    GluonTSDataError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset import DataEntry


@functools.lru_cache(10_000)
def _as_period(val: object, freq: str | pd.DateOffset) -> pd.Period:
    return pd.Period(val, freq)


class ProcessStartField:
    """
    Transform the start field into a Period with the given frequency.

    Parameters
    ----------
    name
        Name of the field to transform.
    freq
        Frequency to use. This must be a valid Pandas frequency string.

    """

    def __init__(
        self,
        *,
        freq: str | pd.DateOffset,
        use_timestamp: bool = False,
        name: str = FieldName.START,
    ) -> None:
        """Process field `name` at frequency `freq`."""
        self.freq = freq
        self.use_timestamp = use_timestamp
        self.name = name

    def __call__(self, data: DataEntry) -> DataEntry:
        """Return `data` with its start a period, or a timestamp."""
        try:
            if self.use_timestamp:
                data[self.name] = pd.Timestamp(data[self.name])
            else:
                data[self.name] = _as_period(data[self.name], self.freq)
        except (TypeError, ValueError) as e:
            msg = f'Error "{e}" occurred, when reading field "{self.name}"'
            raise GluonTSDataError(msg) from e

        return data


class ProcessTimeSeriesField:
    """
    Convert a time series field identified by `name` into a numpy array.

    Constructor parameters modify the conversion logic in the following way:

    If `is_required=True`, throws a `GluonTSDataError` if the field is not
    present in the `Data` dictionary.

    If `is_cat=True`, the array type is `np.int32`, otherwise it is
    `np.float32`.

    If `is_static=True`, asserts that the resulting array is 1D,
    otherwise asserts that the resulting array is 2D. 2D dynamic arrays of
    shape (T) are automatically expanded to shape (1,T).

    Parameters
    ----------
    name
        Name of the field to process.
    is_required
        Whether the field must be present.
    is_cat
        Whether the field refers to categorical (i.e. integer) values.
    is_static
        Whether the field is supposed to have a time dimension.

    """

    def __init__(
        self, name: str, *, is_required: bool, is_static: bool, is_cat: bool
    ) -> None:
        """Process field `name`."""
        self.name = name
        self.is_required = is_required
        self.req_ndim = 1 if is_static else 2
        self.dtype = np.int32 if is_cat else np.float32

    def __call__(self, data: DataEntry) -> DataEntry:
        """Return `data` with the field an array."""
        value = data.get(self.name, None)
        if value is not None:
            value = np.asarray(value, dtype=self.dtype)

            if self.req_ndim != value.ndim:
                msg = (
                    f"Array '{self.name}' has bad shape - expected "
                    f"{self.req_ndim} dimensions, got {value.ndim}."
                )
                raise GluonTSDataError(msg)

            data[self.name] = value

            return data
        if not self.is_required:
            return data
        msg = f"Object is missing a required field `{self.name}`"
        raise GluonTSDataError(msg)


class ProcessDataEntry:
    """The start of an entry a period, its series numpy arrays."""

    def __init__(
        self,
        freq: str,
        *,
        one_dim_target: bool = True,
        use_timestamp: bool = False,
    ) -> None:
        """Process the entries of a dataset of frequency `freq`."""
        self.trans = cast(
            "list[Callable[[DataEntry], DataEntry]]",
            [
                ProcessStartField(freq=freq, use_timestamp=use_timestamp),
                # The next line abuses is_static=True in case of 1D targets.
                ProcessTimeSeriesField(
                    FieldName.TARGET,
                    is_required=True,
                    is_cat=False,
                    is_static=one_dim_target,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_DYNAMIC_CAT,
                    is_required=False,
                    is_cat=True,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_DYNAMIC_REAL_LEGACY,  # backwards compatible
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_DYNAMIC_REAL,
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    FieldName.PAST_FEAT_DYNAMIC_REAL,
                    is_required=False,
                    is_cat=False,
                    is_static=False,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_STATIC_CAT,
                    is_required=False,
                    is_cat=True,
                    is_static=True,
                ),
                ProcessTimeSeriesField(
                    FieldName.FEAT_STATIC_REAL,
                    is_required=False,
                    is_cat=False,
                    is_static=True,
                ),
            ],
        )

    def __call__(self, data: DataEntry) -> DataEntry:
        """Return `data` processed field by field."""
        for t in self.trans:
            data = t(data)
        return data
