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
The period at which the forecast of an entry starts.

This is gluonts 0.15.1's `gluonts/dataset/util.py` (Apache-2.0, a copy of the
license in src/diff_gpt/infrastructure/leaderboards/LICENSE-gluonts), cut to
`forecast_start`. Unchanged but for the style of
diff_gpt.infrastructure.leaderboards.vendored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.field_names import (
    FieldName,
)

if TYPE_CHECKING:
    import pandas as pd

    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset import DataEntry


def forecast_start(entry: DataEntry, time_axis: int = -1) -> pd.Period:
    """Return the period right after the last step of `entry`'s target."""
    return entry[FieldName.START] + entry[FieldName.TARGET].shape[time_axis]
