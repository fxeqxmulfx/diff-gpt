"""
The leaderboard's evaluation: gluonts metrics and the all_results.csv format.

METRICS and the evaluate_model arguments are notebooks/naive.ipynb's, and the
CSV has the 15 columns scripts/validate_results.py checks.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from diff_gpt.infrastructure.leaderboards.gift.configs import RUNS
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from diff_gpt.infrastructure.leaderboards.vendored.gluonts.model.evaluation import (
    evaluate_model,
)

if TYPE_CHECKING:
    from pathlib import Path

    from diff_gpt.infrastructure.leaderboards.gift.configs import Config
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.dataset.split import (
        TestData,
    )
    from diff_gpt.infrastructure.leaderboards.vendored.gluonts.model.predictor import (
        Predictor,
    )

BATCH_SIZE = 512
RESULTS = RUNS / "results"
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]
METRIC_NAMES = (
    "MSE[mean]",
    "MSE[0.5]",
    "MAE[0.5]",
    "MASE[0.5]",
    "MAPE[0.5]",
    "sMAPE[0.5]",
    "MSIS",
    "RMSE[mean]",
    "NRMSE[mean]",
    "ND[0.5]",
    "mean_weighted_sum_quantile_loss",
)
COLUMNS = (
    "dataset",
    "model",
    *(f"eval_metrics/{name}" for name in METRIC_NAMES),
    "domain",
    "num_variates",
)


def evaluate(
    predictor: Predictor, test_data: TestData, season_length: int
) -> dict[str, float]:
    """Score `predictor` on `test_data` as the leaderboard does."""
    result = evaluate_model(
        predictor,
        test_data=test_data,
        metrics=METRICS,
        batch_size=BATCH_SIZE,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    return {name: float(result[name].iloc[0]) for name in METRIC_NAMES}


def results_path(model: str, split: str, results: Path = RESULTS) -> Path:
    """Return the all_results.csv of `model` on `split` under `results`."""
    return results / model / split / "all_results.csv"


def save_row(path: Path, config: Config, model: str, metrics: dict[str, float]) -> None:
    """Write `config`'s row into the CSV at `path`, replacing an earlier one."""
    rows = {}
    if path.exists():
        with path.open(newline="") as file:
            rows = {row["dataset"]: row for row in csv.DictReader(file)}
    rows[config.key] = {
        "dataset": config.key,
        "model": model,
        **{f"eval_metrics/{name}": metrics[name] for name in METRIC_NAMES},
        "domain": config.domain,
        "num_variates": config.num_variates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with partial.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows.values())
    partial.replace(path)
