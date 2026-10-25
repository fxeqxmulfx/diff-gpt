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
The imputation of missing values that gift-eval's notebook applies to a context.

This is gluonts 0.15.1's `gluonts/transform/feature.py` (Apache-2.0, a copy of
the license in leaderboards/LICENSE-gluonts), cut to MissingValueImputation,
DummyValueImputation and LastValueImputation, changed in one place besides
the style of leaderboards.vendored:
- the `__init__` of MissingValueImputation and of DummyValueImputation is
  not wrapped in gluonts' pydantic `@validated()`, which only checks and
  converts the arguments: LastValueImputation() takes none, and
  DummyValueImputation() gets the default dummy_value, the float 0.0.
"""

from __future__ import annotations

import numpy as np


class MissingValueImputation:
    """
    The parent class for all the missing value imputation classes.

    You can just implement your own inheriting this class.
    """

    def __init__(self) -> None:
        """Take no parameters."""

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Return `values` with the NaNs replaced according to the method used."""
        raise NotImplementedError


class DummyValueImputation(MissingValueImputation):
    """Replace all the missing values with the same dummy value given in advance."""

    def __init__(self, dummy_value: float = 0.0) -> None:
        """Impute `dummy_value`."""
        self.dummy_value = dummy_value

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Return `values` with each NaN the dummy value."""
        nan_indices = np.where(np.isnan(values))
        values[nan_indices] = self.dummy_value
        return values


class LastValueImputation(MissingValueImputation):
    """
    Replace each missing value with the last value that was not missing.

    (If the first values are missing, they are replaced by the closest non
    missing value.)
    """

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Return `values` with each NaN the last value before it."""
        if len(values) == 1 or np.isnan(values).all():
            return DummyValueImputation()(values)
        values = np.expand_dims(values, axis=0)

        mask = np.isnan(values)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = values[np.arange(idx.shape[0])[:, None], idx]

        values = np.squeeze(out)
        # in case we need to replace nan at the start of the array
        mask = np.isnan(values)
        values[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), values[~mask]
        )

        return values
