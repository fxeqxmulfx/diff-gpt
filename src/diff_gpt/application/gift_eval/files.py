"""
The file exchange with leaderboards/, the GIFT-Eval adapter diff-gpt never imports.

leaderboards' export writes the windows of each configuration and split to
RUNS/windows/<split>/<file key>.npz. A forecaster reads them with
read_windows and writes its forecasts with write_forecasts to
RUNS/forecasts/<model>/<split>/<file key>.npz, which leaderboards' score
replays through gluonts. The file key is the leaderboard's name of the
configuration, e.g. "m4_hourly/H/short", with "/" as "__".
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from diff_gpt.application.paths import ROOT

if TYPE_CHECKING:
    from pathlib import Path

RUNS = ROOT / "runs" / "leaderboards" / "gift-eval"
QUANTILE_LEVELS = (0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975)


@dataclass(frozen=True)
class Windows:
    """The windows of one configuration and split, in the scorer's order."""

    config: str  # The leaderboard's name, e.g. "m4_hourly/H/short".
    split: str  # "test" or "val".
    prediction_length: int
    freq: str  # The pandas frequency of the series, e.g. "H" or "W-SUN".
    season_length: int  # gluonts' seasonality of freq: MASE's lag.
    values: np.ndarray  # float32, the contexts concatenated; NaN is missing.
    offsets: np.ndarray  # int64 (N+1,): context i is values[offsets[i]:offsets[i+1]].
    item_id: np.ndarray  # str (N,).
    forecast_start: np.ndarray  # str (N,): the period of the first forecast step.

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.item_id)

    @property
    def file_key(self) -> str:
        """Return the file name of this configuration, without ".npz"."""
        return self.config.replace("/", "__")

    def context(self, index: int) -> np.ndarray:
        """Return the context of window `index`."""
        return self.values[self.offsets[index] : self.offsets[index + 1]]


def read_windows(path: Path) -> Windows:
    """Read the windows the export wrote to `path`."""
    with np.load(path) as file:
        return Windows(
            config=str(file["config"]),
            split=str(file["split"]),
            prediction_length=int(file["prediction_length"]),
            freq=str(file["freq"]),
            season_length=int(file["season_length"]),
            values=file["values"],
            offsets=file["offsets"],
            item_id=file["item_id"],
            forecast_start=file["forecast_start"],
        )


def windows_paths(split: str, windows: Path = RUNS / "windows") -> list[Path]:
    """Return the window files of `split`, one per configuration."""
    return sorted((windows / split).glob("*.npz"))


def forecast_path(
    windows: Windows, model: str, forecasts: Path = RUNS / "forecasts"
) -> Path:
    """Return the file of `model`'s forecasts of `windows`."""
    return forecasts / model / windows.split / f"{windows.file_key}.npz"


def write_forecasts(
    path: Path, windows: Windows, quantiles: np.ndarray, mean: np.ndarray
) -> None:
    """
    Write forecasts of `windows` to `path`.

    `quantiles` (N, len(QUANTILE_LEVELS), H) holds each window's quantiles at
    QUANTILE_LEVELS and `mean` (N, H) its mean; both are stored as float32.
    """
    count, horizon = len(windows), windows.prediction_length
    if mean.shape != (count, horizon):
        msg = f"{windows.config}: mean {mean.shape}, expected {(count, horizon)}"
        raise ValueError(msg)
    if quantiles.shape != (count, len(QUANTILE_LEVELS), horizon):
        msg = f"{windows.config}: quantiles {quantiles.shape}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez(
        partial,
        quantile_levels=np.array(QUANTILE_LEVELS, dtype=np.float64),
        quantiles=quantiles.astype(np.float32),
        mean=mean.astype(np.float32),
        item_id=windows.item_id,
    )
    partial.replace(path)
