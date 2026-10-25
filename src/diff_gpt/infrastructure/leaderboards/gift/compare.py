"""
Compare two all_results.csv files metric by metric, as gate G1 does.

``uv run python -m diff_gpt.interfaces.leaderboards.gift.compare A B``. For
each metric it prints the largest relative difference |A - B| / |B| over the
configurations and where it is, then each configuration whose difference
exceeds --rtol. Both files must hold the same configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from diff_gpt.infrastructure.leaderboards.gift.evaluation import METRIC_NAMES

if TYPE_CHECKING:
    import argparse

COLUMNS = [f"eval_metrics/{name}" for name in METRIC_NAMES]


def differences(results: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return |results - reference| / |reference| per configuration and metric."""
    results, reference = results.set_index("dataset"), reference.set_index("dataset")
    if set(results.index) != set(reference.index):
        msg = f"other configurations: {set(results.index) ^ set(reference.index)}"
        raise ValueError(msg)
    results = results.loc[reference.index, COLUMNS]
    reference = reference.loc[:, COLUMNS]
    gap = (results - reference).abs()
    return (gap / reference.abs()).where(gap > 0, 0.0).set_axis(METRIC_NAMES, axis=1)


def print_differences(args: argparse.Namespace) -> None:
    """Print the relative differences of the results file from the reference."""
    relative = differences(pd.read_csv(args.results), pd.read_csv(args.reference))
    print(f"{len(relative)} configurations")
    for name in METRIC_NAMES:
        worst = relative[name].idxmax()
        print(f"{name:32s} max {relative[name].max():.2e} ({worst})")
    above = relative[(relative > args.rtol).any(axis=1)]
    print(f"{len(above)} configurations differ by more than {args.rtol:g}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        if len(above):
            print(above.map(lambda value: f"{value:.1e}" if value else "0"))
    if not np.isfinite(relative.to_numpy()).all():
        print("non-finite differences: a metric is 0 or NaN in the reference")
