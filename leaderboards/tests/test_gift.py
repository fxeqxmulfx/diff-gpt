"""Check the GIFT-Eval adapter: names, windows, the replayed seasonal naive, scores."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from leaderboards.gift.aggregate import aggregate, read_scores
from leaderboards.gift.compare import differences
from leaderboards.gift.configs import GIFT_REPO, ROOT, SPLITS, configurations, select
from leaderboards.gift.evaluation import (
    COLUMNS,
    METRIC_NAMES,
    evaluate,
    results_path,
    save_row,
)
from leaderboards.gift.export import export
from leaderboards.gift.official import SeasonalNaivePredictor
from leaderboards.gift.score import QUANTILE_LEVELS, Forecasts, read_forecasts, score
from leaderboards.gift.windows import load, windows
from leaderboards.vendored.gluonts.time_feature.seasonality import get_seasonality
from tests.tiny import CONFIGS, HORIZON, SEASON, WINDOWS, tiny

if TYPE_CHECKING:
    from pathlib import Path

    from leaderboards.gift.configs import Config

MAX_CONTEXT = 200  # Shorter than every tiny context, so that the cut shows.
PUBLISHED = GIFT_REPO / "results"


def test_configurations_are_the_leaderboards() -> None:
    """The 97 keys are the published ones, named as the notebooks name them."""
    configs = configurations()
    published = pd.read_csv(PUBLISHED / "Seasonal_Naive" / "all_results.csv")
    assert sorted(config.key for config in configs) == sorted(published["dataset"])
    terms = [config.term for config in configs]
    assert [terms.count(term) for term in ("short", "medium", "long")] == [55, 21, 21]
    keys = {(config.name, config.term): config.key for config in configs}
    assert keys["m4_yearly", "short"] == "m4_yearly/A/short"
    assert keys["saugeenday/D", "short"] == "saugeen/D/short"
    assert keys["temperature_rain_with_missing", "short"] == "temperature_rain/D/short"
    assert keys["LOOP_SEATTLE/5T", "long"] == "loop_seattle/5T/long"
    assert keys["bizitobs_application", "medium"] == "bizitobs_application/10S/medium"
    ett = next(config for config in configs if config.key == "ett1/H/medium")
    assert (ett.file_key, ett.domain, ett.num_variates) == (
        "ett1__H__medium",
        "Energy",
        7,
    )
    chosen = select(configs, ["ett1__H__medium", "m4_yearly/A/short"])
    assert [config.key for config in chosen] == ["m4_yearly/A/short", "ett1/H/medium"]
    with pytest.raises(ValueError, match="unknown"):
        select(configs, ["ett1/H/weekly"])


@pytest.mark.parametrize("split", SPLITS)
def test_export_writes_the_windows_in_order(
    data: Path, tmp_path: Path, split: str
) -> None:
    """Test windows run series by series, then window by window; val is one each."""
    dataset = load("tiny", "short", data)
    assert dataset.prediction_length == HORIZON
    assert dataset.windows == WINDOWS
    assert get_seasonality(dataset.freq) == SEASON
    path = tmp_path / "tiny__H__short.npz"
    count = export(
        windows(dataset, split),
        path,
        key="tiny/H/short",
        split=split,
        freq=dataset.freq,
        season_length=SEASON,
        max_context=MAX_CONTEXT,
    )
    targets = tiny()
    gaps = range(WINDOWS, 0, -1) if split == "test" else [WINDOWS + 1]
    ends = [
        (i, len(target) - HORIZON * gap)
        for i, target in enumerate(targets)
        for gap in gaps
    ]
    with np.load(path) as file:
        assert count == len(ends)
        assert file["item_id"].tolist() == [f"T{i}" for i, _ in ends]
        first = pd.Period("2020-01-01 00:00", "h")
        assert file["forecast_start"].tolist() == [
            str(first + 24 * i + end) for i, end in ends
        ]
        assert file["values"].dtype == np.float32
        assert file["offsets"].dtype == np.int64
        for index, (i, end) in enumerate(ends):
            start, stop = file["offsets"][index], file["offsets"][index + 1]
            context = targets[i][:end][-MAX_CONTEXT:]
            np.testing.assert_array_equal(file["values"][start:stop], context)
        assert np.isnan(file["values"]).any()
        fields = [file[name][()] for name in ("config", "split", "freq")]
        assert fields == ["tiny/H/short", split, "H"]
        assert int(file["prediction_length"]) == HORIZON
        assert int(file["season_length"]) == SEASON


def test_export_splits_the_variates(data: Path, tmp_path: Path) -> None:
    """A multivariate row becomes one series per variate, each with its windows."""
    dataset = load("tiny_multi", "short", data)
    path = tmp_path / "tiny_multi__H__short.npz"
    export(
        windows(dataset, "test"),
        path,
        key="tiny_multi/H/short",
        split="test",
        freq=dataset.freq,
        season_length=SEASON,
    )
    with np.load(path) as file:
        ids = [f"M{row}_dim{dim}" for row in range(2) for dim in range(2)]
        assert file["item_id"].tolist() == [i for i in ids for _ in range(WINDOWS)]


@pytest.fixture(scope="module")
def runs(data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the tiny windows and forecast them with the numpy seasonal naive."""
    runs = tmp_path_factory.mktemp("runs")
    for config in CONFIGS:
        dataset = load(config.name, config.term, data)
        for split in SPLITS:
            export(
                windows(dataset, split),
                runs / "windows" / split / f"{config.file_key}.npz",
                key=config.key,
                split=split,
                freq=dataset.freq,
                season_length=SEASON,
                max_context=MAX_CONTEXT,
            )
    for split in SPLITS:
        command = [
            ROOT / ".venv" / "bin" / "python",
            "-m",
            "diff_gpt.interfaces.gift_eval.seasonal_naive",
            "--split",
            split,
            "--windows",
            runs / "windows",
            "--forecasts",
            runs / "forecasts",
        ]
        subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603 - this repository's module, no shell.
    return runs


def forecasts_of(runs: Path, config: Config, split: str) -> Forecasts:
    """Read the seasonal naive's forecasts of `config` on `split`."""
    folder = runs / "forecasts" / "seasonal_naive" / split
    return read_forecasts(folder / f"{config.file_key}.npz")


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.name)
def test_replayed_seasonal_naive_is_the_leaderboards(
    data: Path, runs: Path, config: Config, split: str
) -> None:
    """The numpy forecasts score as statsforecast's do through the notebook's path."""
    dataset = load(config.name, config.term, data)
    forecasts = forecasts_of(runs, config, split)
    assert forecasts.quantiles.shape == (
        len(forecasts.mean),
        len(QUANTILE_LEVELS),
        HORIZON,
    )
    assert forecasts.quantiles.dtype == forecasts.mean.dtype == np.float32
    replayed = score(windows(dataset, split), forecasts, SEASON)
    predictor = SeasonalNaivePredictor(HORIZON, SEASON, max_length=MAX_CONTEXT)
    official = evaluate(predictor, windows(dataset, split), SEASON)
    assert list(replayed) == list(official) == list(METRIC_NAMES)
    for name in METRIC_NAMES:
        assert replayed[name] == pytest.approx(official[name], rel=1e-6), name


def test_score_rejects_misaligned_forecasts(
    data: Path, runs: Path, tmp_path: Path
) -> None:
    """Forecasts in another order, fewer, or over another horizon are errors."""
    config = CONFIGS[0]
    test_data = windows(load(config.name, config.term, data), "test")
    forecasts = forecasts_of(runs, config, "test")
    reordered = Forecasts(
        forecasts.quantiles[::-1], forecasts.mean[::-1], forecasts.item_id[::-1]
    )
    with pytest.raises(ValueError, match="window 0 is of 'T0'"):
        score(test_data, reordered, SEASON)
    fewer = Forecasts(
        forecasts.quantiles[1:], forecasts.mean[1:], forecasts.item_id[1:]
    )
    with pytest.raises(ValueError, match="5 forecasts for 6 windows"):
        score(test_data, fewer, SEASON)
    shorter = Forecasts(
        forecasts.quantiles[..., 1:], forecasts.mean[:, 1:], forecasts.item_id
    )
    with pytest.raises(ValueError, match="the horizon is 48"):
        score(test_data, shorter, SEASON)
    path = runs / "forecasts" / "seasonal_naive" / "test" / f"{config.file_key}.npz"
    with np.load(path) as file:
        arrays = dict(file)
    arrays["quantile_levels"] = arrays["quantile_levels"][1:-1]
    arrays["quantiles"] = arrays["quantiles"][:, 1:-1]
    np.savez(tmp_path / "nine.npz", **arrays)
    with pytest.raises(ValueError, match="quantile_levels"):
        read_forecasts(tmp_path / "nine.npz")


def test_save_row_keeps_one_row_per_configuration(tmp_path: Path) -> None:
    """A row saved again replaces the earlier one, in the board's 15 columns."""
    path = results_path("model", "val", tmp_path)
    save_row(path, CONFIGS[0], "model", dict.fromkeys(METRIC_NAMES, 1.0))
    save_row(path, CONFIGS[1], "model", dict.fromkeys(METRIC_NAMES, 1.0))
    save_row(path, CONFIGS[0], "model", dict.fromkeys(METRIC_NAMES, 2.0))
    table = pd.read_csv(path)
    assert tuple(table.columns) == COLUMNS
    assert table["dataset"].tolist() == ["tiny/H/short", "tiny_multi/H/short"]
    assert table["eval_metrics/MASE[0.5]"].tolist() == [2.0, 1.0]
    assert table["num_variates"].tolist() == [1, 2]


def test_aggregate_is_the_geometric_mean_of_ratios(tmp_path: Path) -> None:
    """Ratios pair configurations by name, after the leaderboard's renaming."""
    baseline = pd.DataFrame(
        {"MASE": [1.0, 2.0], "CRPS": [0.5, 0.25]},
        index=["saugeen/D/short", "b/H/short"],
    )
    path = tmp_path / "all_results.csv"
    pd.DataFrame(
        {
            "dataset": ["b/H/short", "saugeenday/D/short"],
            "eval_metrics/MASE[0.5]": [2.0, 1.0],
            "eval_metrics/mean_weighted_sum_quantile_loss": [1.0, 0.5],
        }
    ).to_csv(path, index=False)
    scores = read_scores(path)
    assert aggregate(scores, baseline) == pytest.approx({"MASE": 1.0, "CRPS": 2.0})
    with pytest.raises(ValueError, match="missing"):
        aggregate(scores.iloc[:1], baseline)
    with pytest.raises(ValueError, match="invalid"):
        aggregate(scores.assign(CRPS=[np.nan, 1.0]), baseline)


@pytest.mark.parametrize(
    ("model", "mase", "crps"), [("PatchTST", 0.849, 0.587), ("TiRex", 0.716, 0.488)]
)
def test_aggregate_reproduces_the_published_leaderboard(
    model: str, mase: float, crps: float
) -> None:
    """The published rows of two entries aggregate to their leaderboard scores."""
    baseline = read_scores(PUBLISHED / "Seasonal_Naive" / "all_results.csv")
    scores = aggregate(read_scores(PUBLISHED / model / "all_results.csv"), baseline)
    assert scores == pytest.approx({"MASE": mase, "CRPS": crps}, abs=6e-4)


def test_differences_are_relative_to_the_reference() -> None:
    """Rows pair by configuration; a difference is |A - B| / |B|, 0 where equal."""
    metrics = [f"eval_metrics/{name}" for name in METRIC_NAMES]
    reference = pd.DataFrame(4.0, index=range(2), columns=metrics).assign(
        dataset=["a/H/short", "b/H/short"]
    )
    results = reference.iloc[::-1].copy()
    results.loc[1, "eval_metrics/MASE[0.5]"] = 5.0
    relative = differences(results, reference)
    expected = pd.DataFrame(0.0, index=relative.index, columns=list(METRIC_NAMES))
    expected.loc["b/H/short", "MASE[0.5]"] = 0.25
    assert relative.index.tolist() == ["a/H/short", "b/H/short"]
    pd.testing.assert_frame_equal(relative, expected)
    with pytest.raises(ValueError, match="other configurations"):
        differences(results.assign(dataset=["a/H/short", "c/H/short"]), reference)
