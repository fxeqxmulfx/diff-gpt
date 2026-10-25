"""Check the window and forecast files against the format the adapter writes."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks.gift_eval.files import (
    QUANTILE_LEVELS,
    forecast_path,
    read_windows,
    windows_paths,
    write_forecasts,
)

if TYPE_CHECKING:
    from pathlib import Path

CONTEXTS = [
    np.array([1.0, np.nan, 3.0], dtype=np.float32),
    np.array([4.0, 5.0], dtype=np.float32),
]


def write_windows(path: Path) -> None:
    """Write CONTEXTS as leaderboards' export writes windows."""
    np.savez(
        path,
        values=np.concatenate(CONTEXTS),
        offsets=np.array([0, 3, 5], dtype=np.int64),
        item_id=np.array(["a", "b"]),
        forecast_start=np.array(["2020-01-01 03:00", "2020-01-02 02:00"]),
        config=np.array("tiny/H/short"),
        split=np.array("val"),
        prediction_length=np.array(4, dtype=np.int64),
        freq=np.array("H"),
        season_length=np.array(24, dtype=np.int64),
    )


def test_read_windows(tmp_path: Path) -> None:
    """The fields come back typed, the contexts sliced by the offsets."""
    (tmp_path / "val").mkdir()
    path = tmp_path / "val" / "tiny__H__short.npz"
    write_windows(path)
    windows = read_windows(path)
    assert windows_paths("val", tmp_path) == [path]
    assert (windows.config, windows.split, windows.freq) == ("tiny/H/short", "val", "H")
    assert (windows.prediction_length, windows.season_length) == (4, 24)
    assert len(windows) == len(CONTEXTS)
    assert windows.file_key == "tiny__H__short"
    for index, context in enumerate(CONTEXTS):
        np.testing.assert_array_equal(windows.context(index), context)
    assert windows.item_id.tolist() == ["a", "b"]
    assert windows.forecast_start.tolist() == ["2020-01-01 03:00", "2020-01-02 02:00"]


def test_write_forecasts(tmp_path: Path) -> None:
    """Forecasts are stored as float32 with the levels and the windows' item ids."""
    write_windows(tmp_path / "windows.npz")
    windows = read_windows(tmp_path / "windows.npz")
    path = forecast_path(windows, "model", tmp_path)
    assert path == tmp_path / "model" / "val" / "tiny__H__short.npz"
    quantiles = np.zeros((2, len(QUANTILE_LEVELS), 4))
    write_forecasts(path, windows, quantiles, np.ones((2, 4)))
    with np.load(path) as file:
        assert file["quantile_levels"].dtype == np.float64
        assert file["quantile_levels"].tolist() == list(QUANTILE_LEVELS)
        assert file["quantiles"].dtype == file["mean"].dtype == np.float32
        assert file["quantiles"].shape == (2, len(QUANTILE_LEVELS), 4)
        assert file["mean"].tolist() == [[1.0] * 4] * 2
        assert file["item_id"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="mean"):
        write_forecasts(path, windows, quantiles, np.ones((2, 5)))
    with pytest.raises(ValueError, match="quantiles"):
        write_forecasts(path, windows, quantiles[:, 1:-1], np.ones((2, 4)))
