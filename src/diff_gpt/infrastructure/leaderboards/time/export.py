"""
Export TIME windows for a forecaster that reads them from files.

``uv run python -m diff_gpt.interfaces.leaderboards.time.export``. For each
task and split it writes RUNS/windows/<split>/<file key>.npz in the format of
diff_gpt.infrastructure.leaderboards.gift.export, which
src/diff_gpt/application/gift_eval/files.py reads: the contexts, each cut to
its last --max-context steps, their item ids and forecast starts, the task's
key, H, frequency and season length (gluonts' get_seasonality, the MASE lag
TIME's experiments pass to the saver).

The windows are univariate and in the order save_window_predictions reads
them: series by series, window by window, the variates of a multivariate
series one after another. No labels are written, for either split.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from diff_gpt.infrastructure.leaderboards.gift.export import export
from diff_gpt.infrastructure.leaderboards.time.configs import select, tasks
from diff_gpt.infrastructure.leaderboards.time.windows import load, split_of
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.time_feature.seasonality import (  # noqa: E501 - the module's name alone is wider than the line.
    get_seasonality,
)

if TYPE_CHECKING:
    import argparse


def export_windows(args: argparse.Namespace) -> None:
    """Export the windows of the tasks and splits that `args` choose."""
    for task in select(tasks(), args.only):
        dataset = load(task, args.data)
        for split in args.split:
            started = time.perf_counter()
            count = export(
                split_of(dataset, split).test_data,
                args.out / split / f"{task.file_key}.npz",
                key=task.key,
                split=split,
                freq=dataset.freq,
                season_length=get_seasonality(dataset.freq),
                max_context=args.max_context,
            )
            seconds = time.perf_counter() - started
            print(f"{split} {task.key}: {count} windows, {seconds:.1f} s", flush=True)
