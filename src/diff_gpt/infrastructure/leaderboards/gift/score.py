"""
Score forecasts that a forecaster wrote to files.

``uv run python -m diff_gpt.interfaces.leaderboards.gift.score --model NAME``.
It reads RUNS/forecasts/<model>/<split>/<file key>.npz for every
configuration:

- quantile_levels: float64 (11,), QUANTILE_LEVELS;
- quantiles: float32 (N, 11, H); mean: float32 (N, H);
- item_id: str (N,), the export's item_id, in its order;

replays each window as a gluonts QuantileForecast that starts where the
window's forecast starts, and runs evaluate_model with the leaderboard's
metrics into RUNS/results/<model>/<split>/all_results.csv. --jobs scores
that many configurations at once, each in a process of its own that exits
after it; this process writes their rows as they finish.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from diff_gpt.application.ports import UsageError
from diff_gpt.infrastructure.leaderboards.gift.configs import configurations, select
from diff_gpt.infrastructure.leaderboards.gift.evaluation import (
    evaluate,
    results_path,
    save_row,
)
from diff_gpt.infrastructure.leaderboards.gift.windows import load, windows
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.util import (
    forecast_start,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.model.forecast import (
    QuantileForecast,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.model.predictor import (
    Predictor,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.time_feature.seasonality import (  # noqa: E501 - the module's name alone is wider than the line.
    get_seasonality,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from diff_gpt.infrastructure.leaderboards.gift.configs import Config
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset import DataEntry
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.split import (
        TestData,
    )

QUANTILE_LEVELS = (0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975)
FORECAST_KEYS = ["mean", *(str(level) for level in QUANTILE_LEVELS)]


@dataclass(frozen=True)
class Forecasts:
    """One configuration's forecasts, one row per window."""

    quantiles: np.ndarray  # (N, len(QUANTILE_LEVELS), H)
    mean: np.ndarray  # (N, H)
    item_id: np.ndarray  # (N,)


def read_forecasts(path: Path) -> Forecasts:
    """Read and check a forecast file."""
    with np.load(path) as file:
        levels = file["quantile_levels"]
        forecasts = Forecasts(file["quantiles"], file["mean"], file["item_id"])
    if not np.array_equal(levels, QUANTILE_LEVELS):
        msg = f"{path}: quantile_levels {levels} are not {QUANTILE_LEVELS}"
        raise ValueError(msg)
    count, horizon = forecasts.mean.shape
    if forecasts.quantiles.shape != (count, len(QUANTILE_LEVELS), horizon):
        msg = f"{path}: quantiles {forecasts.quantiles.shape}, mean {(count, horizon)}"
        raise ValueError(msg)
    if forecasts.item_id.shape != (count,):
        msg = f"{path}: item_id {forecasts.item_id.shape}, {count} forecasts"
        raise ValueError(msg)
    return forecasts


class ReplayPredictor(Predictor):
    """Yield stored forecasts, one per window of the dataset it is given."""

    def __init__(self, forecasts: Forecasts) -> None:
        """Replay `forecasts`."""
        super().__init__(prediction_length=forecasts.mean.shape[1])
        self.forecasts = forecasts

    def predict(self, dataset: Iterable[DataEntry]) -> Iterator[QuantileForecast]:
        """Yield the stored forecast of each window of `dataset`, in order."""
        for index, entry in enumerate(dataset):
            arrays = np.concatenate(
                [self.forecasts.mean[index, None], self.forecasts.quantiles[index]]
            )
            yield QuantileForecast(
                forecast_arrays=arrays.astype(np.float32),
                start_date=forecast_start(entry),
                forecast_keys=FORECAST_KEYS,
                item_id=entry["item_id"],
            )


def check(test_data: TestData, forecasts: Forecasts) -> None:
    """
    Check that `forecasts` are the windows' own, and finite.

    They must be as many as the windows of `test_data`, for the same series in
    the same order, over the same horizon.
    """
    item_ids = np.array([str(entry["item_id"]) for entry in test_data.input])
    if len(item_ids) != len(forecasts.item_id):
        msg = f"{len(forecasts.item_id)} forecasts for {len(item_ids)} windows"
        raise ValueError(msg)
    if forecasts.mean.shape[1] != test_data.prediction_length:
        msg = (
            f"forecasts over {forecasts.mean.shape[1]} steps,"
            f" the horizon is {test_data.prediction_length}"
        )
        raise ValueError(msg)
    wrong = np.flatnonzero(item_ids != forecasts.item_id.astype(str))
    if wrong.size:
        msg = (
            f"window {wrong[0]} is of {str(item_ids[wrong[0]])!r},"
            f" its forecast of {str(forecasts.item_id[wrong[0]])!r}"
        )
        raise ValueError(msg)
    finite = np.isfinite(forecasts.quantiles).all(axis=(1, 2)) & np.isfinite(
        forecasts.mean
    ).all(axis=1)
    if not finite.all():
        msg = f"{np.count_nonzero(~finite)} forecasts are not finite"
        raise ValueError(msg)


def score(
    test_data: TestData, forecasts: Forecasts, season_length: int
) -> dict[str, float]:
    """
    Score `forecasts` of the windows of `test_data`.

    evaluate_model zips the windows with the forecasts and stops at the
    shorter, so they are checked first.
    """
    check(test_data, forecasts)
    return evaluate(ReplayPredictor(forecasts), test_data, season_length)


def run(
    config: Config, split: str, data: Path, path: Path
) -> tuple[Config, dict[str, float], float]:
    """Score the forecasts at `path` of `split` of `config`; return the seconds too."""
    started = time.perf_counter()
    dataset = load(config.name, config.term, data)
    metrics = score(
        windows(dataset, split), read_forecasts(path), get_seasonality(dataset.freq)
    )
    return config, metrics, time.perf_counter() - started


def score_forecasts(args: argparse.Namespace) -> None:
    """Score the model's forecasts of the configurations `args` choose, into one CSV."""
    configs = select(configurations(args.gift_repo), args.only)
    folder = args.forecasts / args.model / args.split
    missing = [c.key for c in configs if not (folder / f"{c.file_key}.npz").exists()]
    if missing:
        msg = f"no forecasts in {folder} for {missing}"
        raise UsageError(msg)
    path = results_path(args.model, args.split, args.results)
    with ProcessPoolExecutor(args.jobs, max_tasks_per_child=1) as pool:
        runs = [
            pool.submit(
                run, config, args.split, args.data, folder / f"{config.file_key}.npz"
            )
            for config in configs
        ]
        for done in as_completed(runs):
            config, metrics, seconds = done.result()
            save_row(path, config, args.model, metrics)
            print(
                f"{config.key}: MASE {metrics['MASE[0.5]']:.4f}"
                f" CRPS {metrics['mean_weighted_sum_quantile_loss']:.4f},"
                f" {seconds:.1f} s",
                flush=True,
            )
    print(f"wrote {path}")
