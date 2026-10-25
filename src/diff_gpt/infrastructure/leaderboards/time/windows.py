"""
The windows of a TIME task: the leaderboard's test split and TIME's validation split.

Test is timebench's Dataset.test_data: the last test_length steps of each
series, cut into floor(test_length / H) consecutive windows. Validation is
its val_data: the val_length steps right before them, cut the same way. Every
window has the whole history before it as context, earlier windows included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from diff_gpt.infrastructure.leaderboards.vendored.timebench.evaluation.data import (
    Dataset,
)

if TYPE_CHECKING:
    from pathlib import Path

    from diff_gpt.infrastructure.leaderboards.time.configs import Task
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.split import (
        TestData,
    )


def load(task: Task, data: Path) -> Dataset:
    """
    Open `task` from `data` as TIME's experiments do.

    A multivariate dataset is split into univariate series (item ids get a
    "_dim<i>" suffix).
    """
    multivariate = (
        Dataset(name=task.name, term=task.term, storage_path=data).target_dim > 1
    )
    return Dataset(
        name=task.name,
        term=task.term,
        to_univariate=multivariate,
        prediction_length=task.prediction_length,
        test_length=task.test_length,
        val_length=task.val_length,
        storage_path=data,
    )


@dataclass(frozen=True)
class Split:
    """The attributes of a dataset that save_window_predictions reads."""

    test_data: TestData  # The windows it scores, whichever split they are.
    windows: int  # Windows per series.
    prediction_length: int
    target_dim: int  # The variates of a series before the univariate split.
    freq: str


def split_of(dataset: Dataset, split: str) -> Split:
    """
    Return the windows of `split` ("test" or "val") for save_window_predictions.

    The saver scores dataset.test_data, dataset.windows windows per series;
    for "val" these are the validation windows.
    """
    if split == "test":
        test_data, windows = dataset.test_data, dataset.windows
    elif split == "val":
        test_data, windows = dataset.val_data, dataset.val_windows
    else:
        msg = f"unknown split {split!r}"
        raise ValueError(msg)
    return Split(
        test_data, windows, dataset.prediction_length, dataset.target_dim, dataset.freq
    )
