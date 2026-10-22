"""
Tiny GIFT-Eval datasets, written in the format of the Hugging Face download.

"tiny" holds three hourly series and "tiny_multi" two hourly rows of two
variates; both have NaN in the places the seasonal naive treats specially.
Hourly data has a horizon of 48 and a season of 24, and the shortest series
(520 steps) gives two test windows.

The same series serve as the TIME datasets "tiny/H" and "tiny_multi/H",
whose tasks (TIME_CONFIG, in the format of timebench's datasets.yaml) have
two test windows and the two validation windows before them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import datasets
import numpy as np
import pandas as pd

from leaderboards.gift.configs import Config

if TYPE_CHECKING:
    from pathlib import Path

HORIZON, SEASON, WINDOWS = 48, 24, 2
CONFIGS = (
    Config("tiny", "short", "tiny/H/short", "Test", 1),
    Config("tiny_multi", "short", "tiny_multi/H/short", "Test", 2),
)
TIME_CONFIG = {
    "datasets": {
        name: {
            "test_length": HORIZON * WINDOWS,
            "val_length": HORIZON * WINDOWS,
            "short": {"prediction_length": HORIZON},
        }
        for name in ("tiny/H", "tiny_multi/H")
    }
}


def series(length: int, seed: int) -> np.ndarray:
    """Return a positive daily-seasonal random walk of `length` hours."""
    rng = np.random.default_rng(seed)
    hours = np.arange(length)
    walk = np.cumsum(rng.normal(size=length))
    return (100 + 10 * np.sin(2 * np.pi * hours / SEASON) + walk).astype(np.float32)


def tiny() -> list[np.ndarray]:
    """
    Return the targets of "tiny".

    T0 has NaN at its start and in its last label; T1 a run of NaN that
    starts the 200-step context of the first test window; T2 NaN that end
    that context and one run inside the validation context.
    """
    targets = [series(520, 0), series(610, 1), series(700, 2)]
    targets[0][:5] = np.nan
    targets[0][-10] = np.nan
    first = len(targets[1]) - HORIZON * WINDOWS
    targets[1][first - 200 : first - 190] = np.nan
    first = len(targets[2]) - HORIZON * WINDOWS
    targets[2][first - 5 : first] = np.nan
    targets[2][first - HORIZON - 50 : first - HORIZON - 40] = np.nan
    return targets


def tiny_multi() -> list[np.ndarray]:
    """Return the targets of "tiny_multi": rows of two variates, one with NaN."""
    targets = [
        np.stack([series(530, 3), series(530, 4)]),
        np.stack([series(560, 5), series(560, 6)]),
    ]
    targets[1][1, -HORIZON * WINDOWS - 3 :] = np.nan
    return targets


def write(root: Path, name: str, targets: list[np.ndarray]) -> None:
    """Save `targets` as gift-eval dataset `name` under `root`."""
    value = datasets.Value("float32")
    target = datasets.Sequence(value)
    if targets[0].ndim > 1:
        target = datasets.Sequence(target)
    features = datasets.Features(
        {
            "item_id": datasets.Value("string"),
            "start": datasets.Value("timestamp[s]"),
            "freq": datasets.Value("string"),
            "target": target,
        }
    )
    prefix = "T" if targets[0].ndim == 1 else "M"
    table = {
        "item_id": [f"{prefix}{index}" for index in range(len(targets))],
        "start": list(pd.date_range("2020-01-01", periods=len(targets), freq="D")),
        "freq": ["H"] * len(targets),
        "target": [values.tolist() for values in targets],
    }
    datasets.Dataset.from_dict(table, features=features).save_to_disk(root / name)
