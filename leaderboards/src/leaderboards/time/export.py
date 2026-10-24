"""
Export TIME windows for a forecaster that reads them from files.

Run in leaderboards/: ``uv run python -m leaderboards.time.export``. For each
task and split it writes RUNS/windows/<split>/<file key>.npz in the format of
leaderboards.gift.export, which benchmarks/gift_eval/files.py reads: the
contexts, each cut to its last --max-context steps, their item ids and
forecast starts, the task's key, H, frequency and season length (gluonts'
get_seasonality, the MASE lag TIME's experiments pass to the saver).

The windows are univariate and in the order save_window_predictions reads
them: series by series, window by window, the variates of a multivariate
series one after another. No labels are written, for either split.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from leaderboards.gift.export import MAX_CONTEXT, export
from leaderboards.time.configs import DATA, RUNS, SPLITS, select, tasks
from leaderboards.time.windows import load, split_of
from leaderboards.vendored.gluonts.time_feature.seasonality import get_seasonality


def main() -> None:
    """Export the windows of the chosen tasks and splits."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    parser.add_argument("--only", nargs="*", help="keys (default: all 98)")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RUNS / "windows")
    args = parser.parse_args()
    if args.max_context < 1:
        parser.error("--max-context must be positive")
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


if __name__ == "__main__":
    main()
