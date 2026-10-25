"""
Aggregate all_results.csv files into the leaderboard's MASE and CRPS.

Run in leaderboards/: ``uv run python -m leaderboards.gift.aggregate CSV...``.
As the GIFT-Eval Space does (src/utils.py: standardize_df, then stats.gmean
per model), MASE[0.5] and CRPS (mean_weighted_sum_quantile_loss) of each
configuration are divided by Seasonal_Naive's, and each score is the
geometric mean of these ratios over the 97 configurations. Test is divided
by the published Seasonal_Naive (results/Seasonal_Naive in the gift-eval
clone), val by our seasonal naive (RUNS/results/seasonal_naive/val). The
Space fills a missing value with the mean over all models; here it is an
error, as is a missing configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from leaderboards.gift.configs import GIFT_REPO, PRETTY_NAMES, SPLITS
from leaderboards.gift.evaluation import RESULTS, results_path

SCORES = {
    "MASE": "eval_metrics/MASE[0.5]",
    "CRPS": "eval_metrics/mean_weighted_sum_quantile_loss",
}


def baseline_path(split: str) -> Path:
    """Return the seasonal naive results `split` is divided by."""
    if split == "test":
        return GIFT_REPO / "results" / "Seasonal_Naive" / "all_results.csv"
    return results_path("seasonal_naive", split, RESULTS)


def read_scores(path: Path) -> pd.DataFrame:
    """Return the MASE and CRPS of each configuration in the CSV at `path`."""
    results = pd.read_csv(path)
    parts = results["dataset"].str.split("/", n=1, expand=True)
    results.index = parts[0].replace(PRETTY_NAMES) + "/" + parts[1]
    if results.index.has_duplicates:
        msg = f"{path}: repeated configurations"
        raise ValueError(msg)
    return results.loc[:, list(SCORES.values())].set_axis(list(SCORES), axis=1)


def ratios(scores: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Divide each configuration's scores by the baseline's."""
    if set(scores.index) != set(baseline.index):
        missing = sorted(set(baseline.index) - set(scores.index))
        extra = sorted(set(scores.index) - set(baseline.index))
        msg = f"configurations missing: {missing}, unknown: {extra}"
        raise ValueError(msg)
    divided = scores / baseline.loc[scores.index]
    invalid = ~(np.isfinite(divided) & (divided > 0)).all(axis=1)
    if invalid.any():
        msg = f"invalid ratios:\n{divided[invalid]}"
        raise ValueError(msg)
    return divided


def aggregate(scores: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float]:
    """Return the geometric means of the ratios of `scores` to `baseline`."""
    divided = ratios(scores, baseline)
    return {name: float(np.exp(np.log(divided[name]).mean())) for name in SCORES}


def main() -> None:
    """Print the leaderboard's scores of each results CSV."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, nargs="+", help="all_results.csv")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--baseline", type=Path, help="default: by --split")
    args = parser.parse_args()
    baseline = read_scores(args.baseline or baseline_path(args.split))
    for path in args.results:
        scores = aggregate(read_scores(path), baseline)
        model = pd.read_csv(path)["model"].iloc[0]
        print(f"{model}: MASE {scores['MASE']:.4f} CRPS {scores['CRPS']:.4f} ({path})")


if __name__ == "__main__":
    main()
