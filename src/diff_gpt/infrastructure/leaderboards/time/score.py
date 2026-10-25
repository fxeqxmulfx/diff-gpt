"""
Score forecasts that a forecaster wrote to files.

``uv run python -m diff_gpt.interfaces.leaderboards.time.score --model NAME``.
It reads RUNS/forecasts/<model>/<split>/<file key>.npz for every task, in the
format diff_gpt.infrastructure.leaderboards.gift.score reads, checks that they
are the windows' own and finite, and hands their quantiles 0.1 to 0.9 to
timebench's save_window_predictions. That computes the metrics of every window
and writes predictions.npz, metrics.npz and config.json (with the forecaster's
settings, from forecast_config.json if the folder has one) into
RUNS/results/<model>/<split>/<dataset>/<freq>/<term>/: for test, the folder a
submission uploads. --jobs scores that many tasks at once, each in a process of
its own that exits after it.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np

from diff_gpt.application.ports import UsageError
from diff_gpt.infrastructure.leaderboards.gift.score import (
    QUANTILE_LEVELS,
    check,
    read_forecasts,
)
from diff_gpt.infrastructure.leaderboards.time.configs import RUNS, select, tasks
from diff_gpt.infrastructure.leaderboards.time.windows import load, split_of
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.time_feature.seasonality import (  # noqa: E501 - the module's name alone is wider than the line.
    get_seasonality,
)
from diff_gpt.infrastructure.leaderboards.vendored.timebench.evaluation.saver import (
    save_window_predictions,
)

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from diff_gpt.infrastructure.leaderboards.gift.score import Forecasts
    from diff_gpt.infrastructure.leaderboards.time.configs import Task
    from diff_gpt.infrastructure.leaderboards.time.windows import Split

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # The saver's default.
RESULTS = RUNS / "results"
SCORES = ("MASE", "CRPS")


def task_scores(folder: Path) -> dict[str, float]:
    """Return the MASE and CRPS of the task in `folder`, as the Space reads them."""
    with np.load(folder / "metrics.npz") as metrics:
        return {name: float(np.nanmean(metrics[name])) for name in SCORES}


def save(
    split: Split,
    quantiles: np.ndarray,
    task: Task,
    folder: Path,
    season_length: int,
    hyperparams: dict | None = None,
) -> dict[str, float]:
    """
    Score `quantiles` (N, 9, H) of the windows of `split` as TIME does.

    save_window_predictions writes them, their metrics and config.json into
    folder/<task key>/; the task's MASE and CRPS are returned. Its progress
    notes are dropped.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        save_window_predictions(
            split,
            quantiles,
            task.key,
            str(folder),
            seasonality=season_length,
            model_hyperparams=hyperparams,
        )
    return task_scores(folder / task.key)


def score(
    split: Split,
    forecasts: Forecasts,
    task: Task,
    folder: Path,
    season_length: int,
    hyperparams: dict | None = None,
) -> dict[str, float]:
    """Score `forecasts` of the windows of `split` into folder/<task key>/."""
    check(split.test_data, forecasts)
    index = [QUANTILE_LEVELS.index(level) for level in LEVELS]
    return save(
        split, forecasts.quantiles[:, index], task, folder, season_length, hyperparams
    )


def run(
    task: Task, split: str, data: Path, path: Path, folder: Path, hyperparams: dict
) -> tuple[Task, dict[str, float], float]:
    """Score the forecasts at `path` of `split` of `task`; return the seconds too."""
    started = time.perf_counter()
    dataset = load(task, data)
    scores = score(
        split_of(dataset, split),
        read_forecasts(path),
        task,
        folder,
        get_seasonality(dataset.freq),
        hyperparams,
    )
    return task, scores, time.perf_counter() - started


def score_forecasts(args: argparse.Namespace) -> None:
    """Score the model's forecasts of the tasks `args` choose, one folder per task."""
    chosen = select(tasks(), args.only)
    source = args.forecasts / args.model / args.split
    missing = [t.key for t in chosen if not (source / f"{t.file_key}.npz").exists()]
    if missing:
        msg = f"no forecasts in {source} for {missing}"
        raise UsageError(msg)
    hyperparams = {"model": args.model}
    record = source / "forecast_config.json"
    if record.exists():
        hyperparams |= json.loads(record.read_text())["settings"]
    folder = args.results / args.model / args.split
    with ProcessPoolExecutor(args.jobs, max_tasks_per_child=1) as pool:
        runs = [
            pool.submit(
                run,
                task,
                args.split,
                args.data,
                source / f"{task.file_key}.npz",
                folder,
                hyperparams,
            )
            for task in chosen
        ]
        for done in as_completed(runs):
            task, scores, seconds = done.result()
            print(
                f"{task.key}: MASE {scores['MASE']:.4f} CRPS {scores['CRPS']:.4f},"
                f" {seconds:.1f} s",
                flush=True,
            )
    print(f"wrote {folder}")
