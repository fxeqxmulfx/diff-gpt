"""
The windows of a configuration: the leaderboard's test split and ours for validation.

Test is gift-eval's Dataset.test_data: the last `windows` horizons of each
series, as consecutive windows. Validation is one window per series, the
horizon right before the test split, with everything before it as context:
gift-eval's training_dataset ends there and its validation_dataset ends where
the test split starts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from diff_gpt.infrastructure.leaderboards.vendored.gift_eval.data import Dataset
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.split import (
    split as split_dataset,
)

if TYPE_CHECKING:
    from pathlib import Path

    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.split import (
        TestData,
    )


def load(name: str, term: str, data: Path) -> Dataset:
    """
    Open gift-eval dataset `name` at `term` from `data`, as the notebooks do.

    A multivariate dataset is split into univariate series (item ids get a
    "_dim<i>" suffix).
    """
    multivariate = Dataset(name=name, term=term, storage_path=data).target_dim > 1
    return Dataset(name=name, term=term, to_univariate=multivariate, storage_path=data)


def windows(dataset: Dataset, split: str) -> TestData:
    """Return the windows of `split` ("test" or "val"), in the scorer's order."""
    if split == "test":
        return dataset.test_data
    if split == "val":
        horizon = dataset.prediction_length
        _, template = split_dataset(
            dataset.gluonts_dataset, offset=-horizon * (dataset.windows + 1)
        )
        return template.generate_instances(
            prediction_length=horizon, windows=1, distance=horizon
        )
    msg = f"unknown split {split!r}"
    raise ValueError(msg)
