"""
Export GIFT-Eval windows for a forecaster that reads them from files.

``uv run python -m diff_gpt.interfaces.leaderboards.gift.export``. For each
configuration and split it writes RUNS/windows/<split>/<file key>.npz:

- values: float32, the contexts concatenated, each cut to its last
  --max-context steps; NaN stays NaN;
- offsets: int64 (N+1,), context i is values[offsets[i]:offsets[i+1]];
- item_id, forecast_start: str (N,), the series and the period its forecast
  starts at;
- 0-d arrays: config (the leaderboard's key), split, prediction_length, freq,
  season_length (gluonts' get_seasonality, as the scorer's MASE uses it).

The windows are univariate and in the order the scorer iterates them. No
labels are written, for either split.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from diff_gpt.infrastructure.leaderboards.gift.configs import configurations, select
from diff_gpt.infrastructure.leaderboards.gift.windows import load, windows
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.util import (
    forecast_start,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.time_feature.seasonality import (  # noqa: E501 - the module's name alone is wider than the line.
    get_seasonality,
)

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.split import (
        TestData,
    )

MAX_CONTEXT = 4096


def export(
    test_data: TestData,
    path: Path,
    *,
    key: str,
    split: str,
    freq: str,
    season_length: int,
    max_context: int = MAX_CONTEXT,
) -> int:
    """Write the windows of `test_data` to `path`; return how many there are."""
    contexts, item_ids, starts = [], [], []
    for entry in test_data.input:
        target = entry["target"]
        if target.ndim != 1:
            msg = f"{key}: {entry['item_id']} is not univariate"
            raise ValueError(msg)
        contexts.append(target[-max_context:])
        item_ids.append(str(entry["item_id"]))
        starts.append(str(forecast_start(entry)))
    offsets = np.zeros(len(contexts) + 1, dtype=np.int64)
    np.cumsum([len(context) for context in contexts], out=offsets[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez(
        partial,
        values=np.concatenate(contexts).astype(np.float32, copy=False),
        offsets=offsets,
        item_id=np.array(item_ids, dtype=str),
        forecast_start=np.array(starts, dtype=str),
        config=np.array(key),
        split=np.array(split),
        prediction_length=np.array(test_data.prediction_length, dtype=np.int64),
        freq=np.array(freq),
        season_length=np.array(season_length, dtype=np.int64),
    )
    partial.replace(path)
    return len(item_ids)


def export_windows(args: argparse.Namespace) -> None:
    """Export the windows of the configurations and splits that `args` choose."""
    for config in select(configurations(args.gift_repo), args.only):
        dataset = load(config.name, config.term, args.data)
        for split in args.split:
            started = time.perf_counter()
            count = export(
                windows(dataset, split),
                args.out / split / f"{config.file_key}.npz",
                key=config.key,
                split=split,
                freq=dataset.freq,
                season_length=get_seasonality(dataset.freq),
                max_context=args.max_context,
            )
            seconds = time.perf_counter() - started
            print(f"{split} {config.key}: {count} windows, {seconds:.1f} s", flush=True)
