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
copy of the license in leaderboards/LICENSE-gluonts), changed in one place
besides the style of leaderboards.vendored:
- for pandas 3, `freq` is read as a period's frequency, in the codes of
  `period_freq` and by `to_offset(..., is_period=True)`, and the offset's
  `rule_code` is read for its `name`, the same string, which pandas 3.1
  deprecates. `freq` is a dataset's frequency in the codes of pandas 2.1, or
  a period's `freqstr`; pandas 3 reads neither as an offset's ("H", "5T",
  "M", "Q-DEC"), and `to_offset` made the same offsets of them in pandas 2.3,
  with a FutureWarning.
"""

import logging

import pandas as pd

from leaderboards.vendored.gluonts.time_feature._base import norm_freq_str, period_freq

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
    offset = pd.tseries.frequencies.to_offset(period_freq(freq), is_period=True)

    base_seasonality = seasonalities.get(norm_freq_str(offset.rule_code), 1)

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
