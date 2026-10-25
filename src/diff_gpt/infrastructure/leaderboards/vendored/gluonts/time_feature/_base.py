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
Pandas frequency strings: their base, and the old codes in pandas 3's.

This is gluonts 0.15.1's `gluonts/time_feature/_base.py` (Apache-2.0, a copy
of the license in src/diff_gpt/infrastructure/leaderboards/LICENSE-gluonts),
cut to `norm_freq_str`, changed in two places besides the style of
diff_gpt.infrastructure.leaderboards.vendored:
- a start frequency ("MS", "QS", ...) becomes its end frequency with the "E"
  of pandas 2.2 and later ("ME", "QE", ...) without a check of pandas'
  version: leaderboards requires pandas 2.2 or later, so the copy needs no
  `packaging` for gluonts' check;
- `period_freq` is added, for pandas 3. The datasets of GIFT-Eval and TIME
  store their frequencies in the codes of pandas 2.1 ("H", "5T", "10S",
  "A-DEC"), which pandas 3 no longer reads; `period_freq` writes them in the
  codes pandas 3 reads for a period ("h", "5min", "10s", "Y-DEC"), the ones
  pandas 2.2 and 2.3 turned them into with a FutureWarning. pandas'
  `to_offset(..., is_period=True)` reads those as the offsets pandas 2.3 made
  of the old codes: "M" a month end, "Q-DEC" a quarter end, "Y-DEC" a year
  end.
"""

import re

#: The codes of pandas 2.1 that pandas 3 renamed, as it names them for a
#: period: those that the datasets of the two boards use.
RENAMED_CODES = {"A": "Y", "H": "h", "S": "s", "T": "min"}


def period_freq(freq: str) -> str:
    """
    Return `freq`, a pandas frequency string, in pandas 3's codes for a period.

    >>> period_freq("5T"), period_freq("A-DEC"), period_freq("W-SUN")
    ('5min', 'Y-DEC', 'W-SUN')
    """
    match = re.fullmatch(r"(\d*)([A-Za-z]+)(-[A-Za-z]+)?", freq)
    if match is None:
        return freq
    multiple, code, anchor = match.groups()
    return multiple + RENAMED_CODES.get(code, code) + (anchor or "")


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
