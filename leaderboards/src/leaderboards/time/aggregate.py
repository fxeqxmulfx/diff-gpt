"""
Aggregate TIME results into the leaderboard's MASE and CRPS.

Run in leaderboards/: ``uv run python -m leaderboards.time.aggregate FOLDER...``,
each FOLDER a model's results on one split, with
<dataset>/<freq>/<term>/metrics.npz for every task. As the TIME Space does
(src/utils.py get_all_datasets_results, src/leaderboard.py
get_overall_leaderboard), a task's MASE and CRPS are the nanmeans of its
metrics.npz over series, windows and variates, divided by seasonal_naive's,
and each score is the geometric mean of these ratios over the 98 tasks.
Test is divided by the hosted seasonal_naive (HOSTED/seasonal_naive), val by
our numpy seasonal naive (RUNS/results/numpy_seasonal_naive/val: a local
seasonal naive never goes in a folder named seasonal_naive, which TIME's
compute_local_leaderboard.py prefers to the hosted one). The Space leaves
a task whose ratio is NaN or infinite out of the mean; here that is an
error, as is a missing task.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from leaderboards.gift.aggregate import aggregate
from leaderboards.time.configs import HOSTED, SPLITS, tasks
from leaderboards.time.score import RESULTS, task_scores

if TYPE_CHECKING:
    from leaderboards.time.configs import Task


def baseline_path(split: str) -> Path:
    """Return the seasonal naive results `split` is divided by."""
    if split == "test":
        return HOSTED / "seasonal_naive"
    return RESULTS / "numpy_seasonal_naive" / split


def read_scores(folder: Path, chosen: list[Task]) -> pd.DataFrame:
    """Return the MASE and CRPS of each task of `chosen` in `folder`."""
    missing = [t.key for t in chosen if not (folder / t.key / "metrics.npz").exists()]
    if missing:
        msg = f"{folder}: no results for {missing}"
        raise ValueError(msg)
    scores = {task.key: task_scores(folder / task.key) for task in chosen}
    return pd.DataFrame.from_dict(scores, orient="index")


def main() -> None:
    """Print the leaderboard's scores of each results folder."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, nargs="+", help="results folders")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--baseline", type=Path, help="default: by --split")
    args = parser.parse_args()
    chosen = tasks()
    baseline = read_scores(args.baseline or baseline_path(args.split), chosen)
    for folder in args.results:
        scores = aggregate(read_scores(folder, chosen), baseline)
        print(f"MASE {scores['MASE']:.4f} CRPS {scores['CRPS']:.4f} ({folder})")


if __name__ == "__main__":
    main()
