"""
Compare two TIME results folders task by task, as gate G1 does.

``uv run python -m diff_gpt.interfaces.leaderboards.time.compare A B``. For
each metric of metrics.npz it prints the median and the largest relative
difference |A - B| / |B| of the tasks' nanmeans, and where the largest is,
then each task whose difference exceeds --rtol. Both folders must hold every
task (or those of --only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from diff_gpt.infrastructure.leaderboards.time.configs import select, tasks

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from diff_gpt.infrastructure.leaderboards.time.configs import Task

METRICS = ("MSE", "MAE", "RMSE", "MAPE", "sMAPE", "MASE", "ND", "CRPS")


def means(folder: Path, chosen: list[Task]) -> pd.DataFrame:
    """Return the nanmean of each metric of each task of `chosen` in `folder`."""
    rows = {}
    for task in chosen:
        path = folder / task.key / "metrics.npz"
        if not path.exists():
            msg = f"{folder}: no results for {task.key}"
            raise ValueError(msg)
        with np.load(path) as metrics:
            rows[task.key] = {
                name: float(np.nanmean(metrics[name])) for name in METRICS
            }
    return pd.DataFrame.from_dict(rows, orient="index")


def differences(results: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return |results - reference| / |reference| per task and metric, 0 where equal."""
    gap = (results - reference.loc[results.index]).abs()
    return (gap / reference.loc[results.index].abs()).where(gap > 0, 0.0)


def print_differences(args: argparse.Namespace) -> None:
    """Print the relative differences of the results folder from the reference."""
    chosen = select(tasks(), args.only)
    relative = differences(means(args.results, chosen), means(args.reference, chosen))
    print(f"{len(relative)} tasks")
    for name in METRICS:
        worst = relative[name].idxmax()
        print(
            f"{name:6s} median {relative[name].median():.2e}"
            f" max {relative[name].max():.2e} ({worst})"
        )
    above = relative[(relative > args.rtol).any(axis=1)]
    print(f"{len(above)} tasks differ by more than {args.rtol:g}")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        if len(above):
            print(above.map(lambda value: f"{value:.1e}" if value else "0"))
    if not np.isfinite(relative.to_numpy()).all():
        print("non-finite differences: a metric is 0 or NaN in the reference")
