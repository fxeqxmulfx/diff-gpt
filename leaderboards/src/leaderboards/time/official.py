"""
TIME's seasonal_naive: timebench's predictor, as experiments/seasonal_naive.py runs it.

Run in leaderboards/: ``uv run python -m leaderboards.time.official --split test``.
For each window the predictor fills a NaN with the value before it (leading
NaN with the first value), forecasts with statsforecast's SeasonalNaive and
its intervals at levels 20 to 80, and draws --samples paths (TIME: 100)
between those quantiles by inverse-CDF sampling. The paths' quantiles 0.1 to
0.9 (np.quantile) are scored by save_window_predictions into
RUNS/results/<model>/<split>/. TIME draws them unseeded (its issue #12), so
two runs differ by about 0.6% per task. --seed makes a run repeatable: before
each task it seeds numpy's global generator with task_seed(seed, key), so
that the tasks of a run draw independently of each other. A multivariate
series is split into its variates first, each drawing its own paths, as
TIME's code does now; the hosted seasonal_naive drew a window's paths once
for all the variates of a series (README.md, TIME).

--samples 0 scores statsforecast's quantiles themselves, without the draws:
the forecasts the numpy seasonal naive of benchmarks/gift_eval replays.
--max-length cuts each context to its last steps first, as
leaderboards.time.export does. --jobs scores that many tasks at once, each
in a process of its own that exits after it.
"""

from __future__ import annotations

import argparse
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from gluonts.time_feature import get_seasonality
from timebench.models import SeasonalNaivePredictor

from leaderboards.time.configs import DATA, SPLITS, select, tasks
from leaderboards.time.score import LEVELS, RESULTS, save
from leaderboards.time.windows import load, split_of

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from gluonts.dataset import DataEntry

    from leaderboards.time.configs import Task

SAMPLES = 100  # experiments/seasonal_naive.py's num_samples.


def task_seed(seed: int, key: str) -> int:
    """
    Return the seed of task `key`'s draws in a run seeded with `seed`.

    It is the CRC-32 of "<seed>:<key>": the same in every process, whatever
    PYTHONHASHSEED, and different for the tasks of one run.
    """
    return zlib.crc32(f"{seed}:{key}".encode())


def cut(inputs: Iterable[DataEntry], max_length: int) -> Iterator[DataEntry]:
    """Yield the entries of `inputs`, each target cut to its last `max_length` steps."""
    for entry in inputs:
        yield {**entry, "target": entry["target"][..., -max_length:]}


def forecast(
    predictor: SeasonalNaivePredictor, inputs: Iterable[DataEntry], samples: int
) -> np.ndarray:
    """
    Return the quantiles (N, 9, H) of `predictor`'s forecasts of `inputs`.

    They are the quantiles of the drawn paths if `samples`, else
    statsforecast's own.
    """
    rows = []
    for prediction in predictor.predict(inputs):
        if samples:
            rows.append(np.quantile(prediction.samples, LEVELS, axis=0)[:, 0])
        else:
            quantiles = prediction.quantile_forecasts
            rows.append(np.stack([quantiles[level] for level in LEVELS]))
    return np.stack(rows)


def run(
    task: Task,
    split: str,
    data: Path,
    folder: Path,
    *,
    samples: int = SAMPLES,
    seed: int | None = None,
    max_length: int | None = None,
) -> tuple[Task, dict[str, float], float]:
    """Score the seasonal naive on `split` of `task`; return the seconds too."""
    started = time.perf_counter()
    dataset = load(task, data)
    windows = split_of(dataset, split)
    season_length = get_seasonality(dataset.freq)
    predictor = SeasonalNaivePredictor(
        dataset.prediction_length,
        season_length=season_length,
        freq=dataset.freq,
        num_samples=samples or 1,
    )
    inputs = windows.test_data.input
    if max_length is not None:
        inputs = cut(inputs, max_length)
    own_seed = None
    if seed is not None:
        own_seed = task_seed(seed, task.key)
        np.random.seed(own_seed)  # noqa: NPY002 - the predictor draws from numpy's global generator.
    quantiles = forecast(predictor, inputs, samples)
    hyperparams = {
        "season_length": season_length,
        "num_samples": samples,
        "seed": seed,
        "task_seed": own_seed,
        "max_length": max_length,
    }
    scores = save(windows, quantiles, task, folder, season_length, hyperparams)
    return task, scores, time.perf_counter() - started


def main() -> None:
    """Score the seasonal naive on the chosen tasks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--model",
        help="default: statsforecast_seasonal_naive[_exact][_<max length>]",
    )
    parser.add_argument("--only", nargs="*", help="keys (default: all 98)")
    parser.add_argument("--jobs", type=int, default=1, help="processes")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be 0 or more")
    model = args.model or "statsforecast_seasonal_naive" + (
        "_exact" if args.samples == 0 else ""
    ) + (f"_{args.max_length}" if args.max_length else "")
    folder = args.results / model / args.split
    with ProcessPoolExecutor(args.jobs, max_tasks_per_child=1) as pool:
        runs = [
            pool.submit(
                run,
                task,
                args.split,
                args.data,
                folder,
                samples=args.samples,
                seed=args.seed,
                max_length=args.max_length,
            )
            for task in select(tasks(), args.only)
        ]
        for done in as_completed(runs):
            task, scores, seconds = done.result()
            print(
                f"{task.key}: MASE {scores['MASE']:.4f} CRPS {scores['CRPS']:.4f},"
                f" {seconds:.1f} s",
                flush=True,
            )
    print(f"wrote {folder}")


if __name__ == "__main__":
    main()
