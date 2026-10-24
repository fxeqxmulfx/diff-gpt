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
The base of a pandas frequency string, without its anchor or start marker.

This is gluonts 0.15.1's `gluonts/time_feature/_base.py` (Apache-2.0, a copy
of the license in leaderboards/LICENSE-gluonts), cut to `norm_freq_str`,
changed in one place besides the style of leaderboards.vendored:
- a start frequency ("MS", "QS", ...) becomes its end frequency with the "E"
  of pandas 2.2 and later ("ME", "QE", ...) without a check of pandas'
  version: leaderboards requires pandas 2.2 or later, so the copy needs no
  `packaging` for gluonts' check.
"""


def norm_freq_str(freq_str: str) -> str:
    """Return `freq_str` without its anchor, a start frequency as its end."""
    base_freq = freq_str.split("-", maxsplit=1)[0]

    # Pandas has start and end frequencies, e.g `AS` and `A` for yearly start
    # and yearly end frequencies. We don't make that difference and instead
    # rely only on the end frequencies which don't have the `S` prefix.
    # Note: Secondly ("S") frequency exists, where we don't want to remove the
    # "S"!
    if len(base_freq) >= 2 and base_freq.endswith("S"):  # noqa: PLR2004 - a code and "S".
        base_freq = base_freq[:-1]
        # In pandas >= 2.2, period end frequencies have been renamed, e.g. "M" -> "ME"
        base_freq += "E"

    return base_freq
