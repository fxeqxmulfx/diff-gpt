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
The seasonality of a frequency, by which MASE scales its errors.

This is gluonts 0.15.1's `gluonts/time_feature/seasonality.py` (Apache-2.0, a
copy of the license in leaderboards/LICENSE-gluonts). Unchanged but for the
style of leaderboards.vendored.
"""

import logging

import pandas as pd

from leaderboards.vendored.gluonts.time_feature._base import norm_freq_str

logger = logging.getLogger(__name__)


DEFAULT_SEASONALITIES = {
    "S": 3600,  # 1 hour
    "s": 3600,  # 1 hour
    "T": 1440,  # 1 day
    "min": 1440,  # 1 day
    "H": 24,  # 1 day
    "h": 24,  # 1 day
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,
    "ME": 12,
    "B": 5,
    "Q": 4,
    "QE": 4,
}


def get_seasonality(
    freq: str, seasonalities: dict[str, int] = DEFAULT_SEASONALITIES
) -> int:
    """
    Return the seasonality of a given frequency.

    >>> get_seasonality("2h")
    12
    """
    offset = pd.tseries.frequencies.to_offset(freq)

    base_seasonality = seasonalities.get(norm_freq_str(offset.name), 1)

    seasonality, remainder = divmod(base_seasonality, offset.n)
    if not remainder:
        return seasonality

    logger.warning(
        "Multiple %s does not divide base seasonality %s. "
        "Falling back to seasonality 1.",
        offset.n,
        base_seasonality,
    )
    return 1
