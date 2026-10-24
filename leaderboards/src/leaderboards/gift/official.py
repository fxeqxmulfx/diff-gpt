"""
The leaderboard's Seasonal_Naive: statsforecast's SeasonalNaive in the notebook.

A port of StatsForecastPredictor from gift-eval's notebooks/naive.ipynb
(Apache-2.0, a copy of the license in leaderboards/LICENSE-gift-eval) with
SeasonalNaive as its model, the reference of gate G1. Run in leaderboards/:
``uv run python -m leaderboards.gift.official --split test``. --max-length
cuts each context to its last steps before imputation, as the predictor's
max_length does; the notebook leaves it unset. --jobs scores that many
configurations at once, each in a process of its own that exits after it.

Three departures from the notebook, none of which changes a forecast:

- statsforecast 2 rejects level 0, which the notebook requests for the 0.5
  quantile. That quantile is the point forecast (lo-0 = mean - Φ⁻¹(0.5)·sigma
  = mean), so it is read from the point forecast's column.
- The notebook forecasts again with SeasonalNaive the series whose forecast
  has NaN. With SeasonalNaive as the model that is the same forecast, so it
  is left out: a NaN forecast fails evaluate_model either way.
- The notebook hands statsforecast's StatsForecast a DataFrame of each
  batch's series with their dates, and reads the forecasts from the
  DataFrame it returns. Here the series go straight to statsforecast's
  GroupedArray, which StatsForecast makes of that DataFrame and forecasts
  in the same way, and the forecasts are read by row and column
  (leaderboards.vendored.statsforecast.core): the same values. Each
  forecast starts at forecast_start(entry), where the notebook takes the
  first date of statsforecast's forecast as a period; the metrics never
  read it.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from leaderboards.gift.configs import DATA, GIFT_REPO, SPLITS, configurations, select
from leaderboards.gift.evaluation import (
    BATCH_SIZE,
    RESULTS,
    evaluate,
    results_path,
    save_row,
)
from leaderboards.gift.windows import load, windows
from leaderboards.vendored.gluonts.dataset.util import forecast_start
from leaderboards.vendored.gluonts.model.forecast import QuantileForecast
from leaderboards.vendored.gluonts.model.predictor import Predictor
from leaderboards.vendored.gluonts.time_feature.seasonality import get_seasonality
from leaderboards.vendored.gluonts.transform.feature import LastValueImputation
from leaderboards.vendored.statsforecast.core import GroupedArray
from leaderboards.vendored.statsforecast.models import SeasonalNaive

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import pandas as pd

    from leaderboards.gift.configs import Config
    from leaderboards.vendored.gluonts.dataset import DataEntry

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MODEL = "SeasonalNaive"  # statsforecast's column of the point forecast.


def interval(level: float) -> int:
    """Return the statsforecast interval (percent) whose bound is quantile `level`."""
    return round(200 * (max(level, 1 - level) - 0.5))


def column(level: float) -> str:
    """Return the statsforecast column of quantile `level`."""
    if interval(level) == 0:
        return MODEL
    return f"{MODEL}-{'hi' if level > 1 - level else 'lo'}-{interval(level)}"


class SeasonalNaivePredictor(Predictor):
    """The notebook's StatsForecastPredictor with SeasonalNaive as its model."""

    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        max_length: int | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        """Forecast `prediction_length` steps, with seasons of `season_length`."""
        super().__init__(prediction_length=prediction_length)
        self.model = SeasonalNaive(season_length=season_length)
        self.fallback_model = SeasonalNaive(season_length=season_length)
        self.forecast_keys = ["mean", *(str(level) for level in QUANTILE_LEVELS)]
        self.columns = [MODEL, *(column(level) for level in QUANTILE_LEVELS)]
        self.levels = sorted({interval(level) for level in QUANTILE_LEVELS} - {0})
        self.imputation = LastValueImputation()
        self.max_length = max_length
        self.batch_size = batch_size

    def predict(self, dataset: Iterable[DataEntry]) -> Iterator[QuantileForecast]:
        """Forecast each series of `dataset`, `batch_size` series at a time."""
        batch = []
        for entry in dataset:
            target = entry["target"]
            if target.ndim != 1 or len(target) < 1:
                msg = f"{entry['item_id']}: not a univariate series"
                raise ValueError(msg)
            if self.max_length is not None:
                target = target[-self.max_length :]
            target = np.asarray(target, np.float32)
            if np.isnan(target).any():
                target = self.imputation(target.copy())
            unique_id = f"{entry['item_id']}_{forecast_start(entry)}_{len(batch)}"
            batch.append((unique_id, target, forecast_start(entry)))
            if len(batch) == self.batch_size:
                yield from self._forecast(batch)
                batch = []
        if batch:
            yield from self._forecast(batch)

    def _forecast(
        self, batch: list[tuple[str, np.ndarray, pd.Period]]
    ) -> Iterator[QuantileForecast]:
        h = self.prediction_length
        targets = [target for _, target, _ in batch]
        result = GroupedArray(
            np.concatenate(targets), np.cumsum([0, *map(len, targets)])
        ).forecast(
            models=[self.model],
            h=h,
            fallback_model=self.fallback_model,
            level=self.levels,
        )
        forecasts = result["forecasts"]
        index = [result["cols"].index(name) for name in self.columns]
        for i, (unique_id, _, start_date) in enumerate(batch):
            yield QuantileForecast(
                forecast_arrays=np.stack(
                    [forecasts[i * h : (i + 1) * h, j] for j in index]
                ),
                forecast_keys=self.forecast_keys,
                start_date=start_date,
                item_id=unique_id,
            )


def run(
    config: Config, split: str, data: Path, max_length: int | None, batch_size: int
) -> tuple[Config, dict[str, float], float]:
    """Score the seasonal naive on `split` of `config`; return the seconds too."""
    started = time.perf_counter()
    dataset = load(config.name, config.term, data)
    season_length = get_seasonality(dataset.freq)
    predictor = SeasonalNaivePredictor(
        dataset.prediction_length,
        season_length=season_length,
        max_length=max_length,
        batch_size=batch_size,
    )
    metrics = evaluate(predictor, windows(dataset, split), season_length)
    return config, metrics, time.perf_counter() - started


def main() -> None:
    """Score the seasonal naive on the chosen configurations."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--model", help="default: statsforecast_seasonal_naive[_<max length>]"
    )
    parser.add_argument("--only", nargs="*", help="keys (default: all 97)")
    parser.add_argument("--jobs", type=int, default=1, help="processes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--gift-repo", type=Path, default=GIFT_REPO)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    model = args.model or "statsforecast_seasonal_naive" + (
        f"_{args.max_length}" if args.max_length else ""
    )
    path = results_path(model, args.split, args.results)
    configs = select(configurations(args.gift_repo), args.only)
    with ProcessPoolExecutor(args.jobs, max_tasks_per_child=1) as pool:
        runs = [
            pool.submit(
                run, config, args.split, args.data, args.max_length, args.batch_size
            )
            for config in configs
        ]
        for done in as_completed(runs):
            config, metrics, seconds = done.result()
            save_row(path, config, model, metrics)
            print(
                f"{config.key}: MASE {metrics['MASE[0.5]']:.4f}"
                f" CRPS {metrics['mean_weighted_sum_quantile_loss']:.4f},"
                f" {seconds:.1f} s",
                flush=True,
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
