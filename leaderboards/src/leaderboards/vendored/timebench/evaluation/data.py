# Copyright (c) 2023, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2
# Modified by TIME for flexible prediction lengths and YAML config support.

"""
A TIME dataset: its series as gluonts entries, its test and validation windows.

This is timebench c633e23's `src/timebench/evaluation/data.py`: gift-eval's
`src/gift_eval/data.py` modified by TIME (Apache-2.0 by the notice above, a
copy of the license in leaderboards/LICENSE-gift-eval; timebench is MIT, a
copy of the license in leaderboards/LICENSE-timebench), cut to `Term`,
`load_dataset_config` and the test and validation windows of `Dataset` (its
`get_dataset_settings`, `past_feat_dynamic_real_dim`, `get_variate_names`,
`_max_series_length`, `_avg_series_length`, `sum_series_length`,
`training_dataset` and `validation_dataset` are left out), changed in three
places besides the style of leaderboards.vendored:
- `itemize_start` and `MultivariateToUnivariate` are imported from the copy
  of gift-eval's data.py: TIME's are the same code.
- The configuration is the copy's config/datasets.yaml, a byte copy of
  timebench's src/timebench/config/datasets.yaml at the same place relative
  to this file. timebench looks there first and then in a config/ folder of
  a checkout of its repository; an installed timebench has the first.
- Dataset takes the folder of the datasets as `storage_path`, always given by
  leaderboards. timebench lets it be left out and then reads it from the
  environment variable `storage_env_var` (TIME_DATASET), after python-dotenv
  loads a .env file into the environment.
toolz's `compose(process, itemize_start)` is written as two Maps, the inner
one itemizing the start: the same calls in the same order, entry by entry.
"""

from __future__ import annotations

import math
import warnings
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets
import pyarrow.compute as pc
import yaml
from pandas.tseries.frequencies import to_offset

from leaderboards.vendored.gift_eval.data import (
    MultivariateToUnivariate,
    itemize_start,
)
from leaderboards.vendored.gluonts.dataset.common import ProcessDataEntry
from leaderboards.vendored.gluonts.dataset.split import split
from leaderboards.vendored.gluonts.itertools import Map
from leaderboards.vendored.gluonts.time_feature._base import norm_freq_str

if TYPE_CHECKING:
    import numpy as np

    from leaderboards.vendored.gluonts.dataset.split import TestData

# --- Constants from Gift-Eval---
# Default prediction length by frequency (used when no config provided)
M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 12,
    "W": 8,
    "B": 21,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

# The configuration inside the timebench package.
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"


class Term(Enum):
    """The term of a forecast: its default horizon a multiple of the frequency's."""

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
        return 1


def load_dataset_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load dataset configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return {}

    with config_path.open() as f:
        return yaml.safe_load(f) or {}


class Dataset:
    """A TIME dataset at one term, with its test and validation lengths."""

    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        *,
        to_univariate: bool = False,
        prediction_length: int | None = None,
        test_length: int | None = None,
        val_length: int = 0,
        storage_path: str | Path,
    ) -> None:
        """
        Initialize a TimeBench Dataset.

        Parameters
        ----------
        name : str
            Dataset name (path relative to storage, e.g., "bitbrains_rnd/5T")
        term : Term or str
            Forecast horizon term: "short", "medium", or "long"
        to_univariate : bool
            Convert multivariate to univariate
        prediction_length : int, optional
            Prediction length. If None, use default calculation based on freq and
            term.
        test_length : int
            Length of test set (in time steps). Required parameter.
            Windows is auto-calculated: ceil(test_length / prediction_length)
        val_length : int
            Length of validation set (in time steps). Required parameter.
            Val windows is auto-calculated: ceil(val_length / prediction_length)
        storage_path : str or Path
            Direct path to dataset storage.

        """
        resolved_storage_path = Path(storage_path)

        dataset_path = resolved_storage_path / name

        if not dataset_path.exists():
            msg = f"Dataset not found at: {dataset_path}"
            raise FileNotFoundError(msg)

        self.hf_dataset = datasets.load_from_disk(str(dataset_path)).with_format(
            "numpy"
        )

        self.term = Term(term) if isinstance(term, str) else term
        self.name = name
        self._custom_prediction_length = prediction_length
        self._test_length = test_length
        self._val_length = val_length

        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(process, Map(itemize_start, self.hf_dataset))
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )

    @cached_property
    def prediction_length(self) -> int:
        """Get prediction length: the explicit value if given, else the default."""
        if self._custom_prediction_length is not None:
            return self._custom_prediction_length

        # Default calculation based on frequency and term
        freq = norm_freq_str(to_offset(self.freq).name)
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
        target = self.hf_dataset[0]["target"]
        return target.shape[0] if len(target.shape) > 1 else 1

    def _validate_lengths(self) -> None:
        """Validate test_length and val_length against min_series_length."""
        min_len = self._min_series_length

        # Check test_length
        if self._test_length > min_len:
            msg = (
                f"test_length ({self._test_length}) exceeds min_series_length"
                f" ({min_len}). Please reduce test_length."
            )
            raise ValueError(msg)
        if self._test_length > 0.5 * min_len:
            warnings.warn(
                f"test_length ({self._test_length}) exceeds 50% of"
                f" min_series_length ({min_len}). This may leave insufficient"
                " data for training.",
                UserWarning,
                stacklevel=1,
            )

        # Check val_length
        if self._val_length > min_len:
            msg = (
                f"val_length ({self._val_length}) exceeds min_series_length"
                f" ({min_len}). Please reduce val_length."
            )
            raise ValueError(msg)
        if self._val_length > 0.5 * min_len:
            warnings.warn(
                f"val_length ({self._val_length}) exceeds 50% of"
                f" min_series_length ({min_len}). This may leave insufficient"
                " data for training.",
                UserWarning,
                stacklevel=1,
            )

    @cached_property
    def windows(self) -> int:
        """Get number of test windows: floor(test_length / prediction_length)."""
        self._validate_lengths()
        w = math.floor(self._test_length / self.prediction_length)
        return max(1, w)

    @cached_property
    def val_windows(self) -> int:
        """Get number of validation windows: floor(val_length / prediction_length)."""
        self._validate_lengths()
        w = math.floor(self._val_length / self.prediction_length)
        return max(1, w)

    @cached_property
    def _series_lengths(self) -> np.ndarray:
        """Get array of all series lengths."""
        target_col = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            # Multivariate: get length of inner list
            lengths = pc.list_value_length(
                pc.list_flatten(pc.list_slice(target_col, 0, 1))
            )
        else:
            lengths = pc.list_value_length(target_col)
        return lengths.to_numpy()

    @cached_property
    def _min_series_length(self) -> int:
        return int(min(self._series_lengths))

    @property
    def test_data(self) -> TestData:
        """The test windows: the last test_length steps of each series."""
        _, test_template = split(self.gluonts_dataset, offset=-self._test_length)
        return test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )

    @property
    def val_data(self) -> TestData:
        """
        Get validation data in the same format as test_data.

        Returns TestData with val_windows instances for evaluation.
        """
        # Split at offset that includes both test and val windows
        # Validation set is before test set, so we need to split further back
        _, val_template = split(
            self.gluonts_dataset, offset=-(self._test_length + self._val_length)
        )
        # Generate validation instances starting from the split point
        return val_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.val_windows,
            distance=self.prediction_length,
        )
