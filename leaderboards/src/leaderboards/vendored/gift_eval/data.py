# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A GIFT-Eval dataset: its series as gluonts entries, its horizon and test windows.

This is gift-eval 3c7be6d's `src/gift_eval/data.py` (Apache-2.0, a copy of the
license in leaderboards/LICENSE-gift-eval), cut to the test split of Dataset
(its `past_feat_dynamic_real_dim`, `sum_series_length`, `training_dataset` and
`validation_dataset` and the TFB horizons are left out), changed in one place
besides the style of leaderboards.vendored:
- Dataset takes the folder of the datasets as `storage_path`. gift-eval reads
  it from the environment variable `storage_env_var` (GIFT_EVAL), after
  python-dotenv loads a .env file into the environment: leaderboards set
  GIFT_EVAL to that folder right before each Dataset, and python-dotenv,
  which keeps a variable already set, found no .env file on its path.
toolz's `compose(process, itemize_start)` is written as two Maps, the inner
one itemizing the start: the same calls in the same order, entry by entry.
"""

from __future__ import annotations

import math
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import pyarrow.compute as pc
from pandas.tseries.frequencies import to_offset

from leaderboards.vendored.gluonts.dataset.common import ProcessDataEntry
from leaderboards.vendored.gluonts.dataset.split import split
from leaderboards.vendored.gluonts.itertools import Map
from leaderboards.vendored.gluonts.time_feature._base import norm_freq_str
from leaderboards.vendored.gluonts.transform._base import Transformation

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from leaderboards.vendored.gluonts.dataset import DataEntry
    from leaderboards.vendored.gluonts.dataset.split import TestData

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}


class Term(Enum):
    """The term of a forecast: its horizon a multiple of the frequency's."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        """The multiple of the frequency's horizon."""
        if self == Term.SHORT:
            return 1
        if self == Term.MEDIUM:
            return 10
        if self == Term.LONG:
            return 15
        return None


def itemize_start(data_entry: DataEntry) -> DataEntry:
    """Return `data_entry` with its start a Python scalar, the `.item()` of numpy's."""
    data_entry["start"] = data_entry["start"].item()
    return data_entry


def maybe_reconvert_freq(freq: str) -> str:
    """If the freq is one of the newest pandas freqs, convert it to the old freq."""
    deprecated_map = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }
    if freq in deprecated_map:
        return deprecated_map[freq]
    return freq


class MultivariateToUnivariate(Transformation):
    """Split each entry into one entry per variate of field `field`."""

    def __init__(self, field: str) -> None:
        """Split field `field`."""
        self.field = field

    def __call__(
        self,
        data_it: Iterable[DataEntry],
        *,
        is_train: bool = False,  # noqa: ARG002 - Transformation's signature.
    ) -> Iterator:
        """Yield an entry per variate, its item id suffixed with "_dim<i>"."""
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id_, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id_)
                yield univariate_entry


class Dataset:
    """A GIFT-Eval dataset at one term."""

    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        *,
        to_univariate: bool = False,
        storage_path: str | Path,
    ) -> None:
        """Open dataset `name` from the folder `storage_path`."""
        storage_path = Path(storage_path)
        self.hf_dataset = datasets.load_from_disk(str(storage_path / name)).with_format(
            "numpy"
        )
        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(process, Map(itemize_start, self.hf_dataset))
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )

        self.term = Term(term)
        self.name = name

    @cached_property
    def prediction_length(self) -> int:
        """The horizon: the frequency's, times the term's multiplier."""
        freq = norm_freq_str(to_offset(self.freq).name)
        freq = maybe_reconvert_freq(freq)
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        """The frequency of the series."""
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        """The variates of a series."""
        return (
            target.shape[0]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    @cached_property
    def windows(self) -> int:
        """The test windows per series: a tenth of the shortest, 1 to 20."""
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(
                    pc.list_slice(self.hf_dataset.data.column("target"), 0, 1)
                )
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return min(lengths.to_numpy())

    @property
    def test_data(self) -> TestData:
        """The test windows: the last `windows` horizons of each series."""
        _, test_template = split(
            self.gluonts_dataset, offset=-self.prediction_length * self.windows
        )
        return test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
