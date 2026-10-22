"""Check the TIME adapter: tasks, windows, the replayed seasonal naive, scores."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from leaderboards.gift.export import export
from leaderboards.gift.score import QUANTILE_LEVELS, Forecasts, read_forecasts
from leaderboards.time.aggregate import aggregate, read_scores
from leaderboards.time.compare import METRICS, differences
from leaderboards.time.configs import HOSTED, ROOT, SPLITS, select, tasks
from leaderboards.time.official import run
from leaderboards.time.score import score
from leaderboards.time.windows import load, split_of
from tests.tiny import HORIZON, SEASON, TIME_CONFIG, WINDOWS, tiny, tiny_multi

if TYPE_CHECKING:
    from pathlib import Path

    from leaderboards.time.configs import Task

MAX_CONTEXT = 200  # Shorter than every tiny context, so that the cut shows.
TASKS = tuple(tasks(TIME_CONFIG))


def ends(targets: list[np.ndarray], split: str) -> list[tuple[int, int]]:
    """Return (series, end of context) of each window of `split`, in order."""
    first = WINDOWS if split == "test" else 2 * WINDOWS
    return [
        (i, target.shape[-1] - HORIZON * gap)
        for i, target in enumerate(targets)
        for gap in range(first, first - WINDOWS, -1)
    ]


def test_tasks_are_the_leaderboards() -> None:
    """The 98 keys are the hosted ones, with datasets.yaml's lengths."""
    all_tasks = tasks()
    folder = HOSTED / "seasonal_naive"
    hosted = [path.parent.relative_to(folder) for path in folder.glob("*/*/*/*.npz")]
    assert sorted(task.key for task in all_tasks) == sorted(map(str, hosted))
    terms = [task.term for task in all_tasks]
    assert [terms.count(term) for term in ("short", "medium", "long")] == [50, 24, 24]
    keyed = {task.key: task for task in all_tasks}
    weather = keyed["SG_Weather/D/long"]
    assert (weather.prediction_length, weather.test_length, weather.val_length) == (
        14,
        366,
        366,
    )
    assert weather.file_key == "SG_Weather__D__long"
    azure = keyed["azure2019_D/5T/short"]
    assert (azure.prediction_length, azure.file_key) == (288, "azure2019_D__5T__short")
    chosen = select(all_tasks, ["JOLTS/M/short", "SG_Weather__D__long"])
    assert [task.key for task in chosen] == ["SG_Weather/D/long", "JOLTS/M/short"]
    with pytest.raises(ValueError, match="unknown"):
        select(all_tasks, ["SG_Weather/D/weekly"])


@pytest.mark.parametrize("split", SPLITS)
def test_export_writes_the_windows_in_order(
    time_data: Path, tmp_path: Path, split: str
) -> None:
    """Windows go series by series, window by window; val ends where test starts."""
    task = TASKS[0]
    dataset = load(task, time_data)
    assert dataset.windows == dataset.val_windows == WINDOWS
    path = tmp_path / f"{task.file_key}.npz"
    count = export(
        split_of(dataset, split).test_data,
        path,
        key=task.key,
        split=split,
        freq=dataset.freq,
        season_length=SEASON,
        max_context=MAX_CONTEXT,
    )
    targets = tiny()
    windows = ends(targets, split)
    with np.load(path) as file:
        assert count == len(windows)
        assert file["item_id"].tolist() == [f"T{i}" for i, _ in windows]
        first = pd.Period("2020-01-01 00:00", "h")
        assert file["forecast_start"].tolist() == [
            str(first + 24 * i + end) for i, end in windows
        ]
        for index, (i, end) in enumerate(windows):
            start, stop = file["offsets"][index], file["offsets"][index + 1]
            context = targets[i][:end][-MAX_CONTEXT:]
            np.testing.assert_array_equal(file["values"][start:stop], context)
        fields = [file[name][()] for name in ("config", "split", "freq")]
        assert fields == ["tiny/H/short", split, "H"]
        assert int(file["prediction_length"]) == HORIZON


def test_export_splits_the_variates(time_data: Path, tmp_path: Path) -> None:
    """A multivariate row becomes one series per variate, each with its windows."""
    task = TASKS[1]
    path = tmp_path / f"{task.file_key}.npz"
    export(
        split_of(load(task, time_data), "test").test_data,
        path,
        key=task.key,
        split="test",
        freq="H",
        season_length=SEASON,
    )
    with np.load(path) as file:
        ids = [f"M{row}_dim{dim}" for row in range(2) for dim in range(2)]
        assert file["item_id"].tolist() == [i for i in ids for _ in range(WINDOWS)]


@pytest.fixture(scope="module")
def runs(time_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the tiny windows and forecast them with the numpy seasonal naive."""
    runs = tmp_path_factory.mktemp("runs")
    for task in TASKS:
        dataset = load(task, time_data)
        for split in SPLITS:
            export(
                split_of(dataset, split).test_data,
                runs / "windows" / split / f"{task.file_key}.npz",
                key=task.key,
                split=split,
                freq=dataset.freq,
                season_length=SEASON,
                max_context=MAX_CONTEXT,
            )
    for split in SPLITS:
        command = [
            ROOT / ".venv" / "bin" / "python",
            "-m",
            "benchmarks.gift_eval.seasonal_naive",
            "--split",
            split,
            "--windows",
            runs / "windows",
            "--forecasts",
            runs / "forecasts",
        ]
        subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603 - this repository's module, no shell.
    return runs


def forecasts_of(runs: Path, task: Task, split: str) -> Forecasts:
    """Read the seasonal naive's forecasts of `task` on `split`."""
    folder = runs / "forecasts" / "seasonal_naive" / split
    return read_forecasts(folder / f"{task.file_key}.npz")


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_replayed_seasonal_naive_is_times(
    time_data: Path, runs: Path, tmp_path: Path, task: Task, split: str
) -> None:
    """The numpy forecasts score as TIME's own quantiles do, window by window."""
    windows = split_of(load(task, time_data), split)
    replayed = score(
        windows, forecasts_of(runs, task, split), task, tmp_path / "replay", SEASON
    )
    _, official, _ = run(
        task, split, time_data, tmp_path / "official", samples=0, max_length=200
    )
    assert replayed == pytest.approx(official, rel=1e-6)
    paths = [tmp_path / name / task.key for name in ("replay", "official")]
    with np.load(paths[0] / "metrics.npz") as a, np.load(paths[1] / "metrics.npz") as b:
        assert sorted(a.files) == sorted(b.files) == sorted(METRICS)
        for name in METRICS:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-6, err_msg=name)
    config = json.loads((paths[0] / "config.json").read_text())
    variates = 1 if task.name == "tiny/H" else 2
    series = 3 if task.name == "tiny/H" else 2
    assert (config["num_series"], config["num_windows"]) == (series, WINDOWS)
    assert config["num_variates"] == variates
    assert config["quantile_levels"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.mark.parametrize("split", SPLITS)
def test_score_reads_the_labels_of_its_split(
    time_data: Path, runs: Path, tmp_path: Path, split: str
) -> None:
    """Forecasts equal to the split's labels score 0 wherever a label is known."""
    task = TASKS[1]
    targets = [variate for row in tiny_multi() for variate in row]
    labels = np.stack(
        [targets[i][end : end + HORIZON] for i, end in ends(targets, split)]
    )
    oracle = np.nan_to_num(labels)
    forecasts = Forecasts(
        np.repeat(oracle[:, None], len(QUANTILE_LEVELS), axis=1),
        oracle,
        forecasts_of(runs, task, split).item_id,
    )
    score(split_of(load(task, time_data), split), forecasts, task, tmp_path, SEASON)
    with np.load(tmp_path / task.key / "metrics.npz") as metrics:
        mae, crps = metrics["MAE"], metrics["CRPS"]
    # (series, windows, variates) against the windows' order: variates within a row.
    known = ~np.isnan(labels).all(axis=1).reshape(2, 2, WINDOWS).transpose(0, 2, 1)
    assert np.array_equal(~np.isnan(mae), known)
    assert np.nanmax(mae) == np.nanmax(crps) == 0
    # Row 1's second variate has no test label; its last val window lacks three.
    assert known.all() == (split == "val")


def test_score_rejects_misaligned_and_infinite_forecasts(
    time_data: Path, runs: Path, tmp_path: Path
) -> None:
    """Forecasts out of order, or not finite, are errors before anything is saved."""
    task = TASKS[0]
    windows = split_of(load(task, time_data), "test")
    forecasts = forecasts_of(runs, task, "test")
    reordered = Forecasts(
        forecasts.quantiles[::-1], forecasts.mean[::-1], forecasts.item_id[::-1]
    )
    with pytest.raises(ValueError, match="window 0 is of 'T0'"):
        score(windows, reordered, task, tmp_path, SEASON)
    quantiles = forecasts.quantiles.copy()
    quantiles[2, 4, 7] = np.inf
    infinite = Forecasts(quantiles, forecasts.mean, forecasts.item_id)
    with pytest.raises(ValueError, match="1 forecasts are not finite"):
        score(windows, infinite, task, tmp_path, SEASON)
    assert not (tmp_path / task.key).exists()


def test_official_draws_repeat_with_a_seed(time_data: Path, tmp_path: Path) -> None:
    """TIME's draws are unseeded; with --seed a run is the same every time."""
    task = TASKS[0]
    scores = [
        run(task, "test", time_data, tmp_path / str(index), seed=seed)[1]
        for index, seed in enumerate((0, 0, 1))
    ]
    assert scores[0] == scores[1]
    assert scores[0] != scores[2]


def test_read_scores_are_nanmeans(tmp_path: Path) -> None:
    """A task's MASE and CRPS are its metrics' nanmeans; a missing task is an error."""
    for task, value in zip(TASKS, (1.0, 3.0), strict=True):
        (tmp_path / task.key).mkdir(parents=True)
        metrics = np.full((2, WINDOWS, 1), value)
        metrics[0, 0, 0] = np.nan
        np.savez(tmp_path / task.key / "metrics.npz", MASE=metrics, CRPS=2 * metrics)
    scores = read_scores(tmp_path, list(TASKS))
    assert scores.loc["tiny/H/short"].to_dict() == {"MASE": 1.0, "CRPS": 2.0}
    baseline = scores.assign(MASE=[2.0, 1.0])
    assert aggregate(scores, baseline) == pytest.approx(
        {"MASE": np.sqrt(0.5 * 3.0), "CRPS": 1.0}
    )
    with pytest.raises(ValueError, match="no results"):
        read_scores(tmp_path / "other", list(TASKS))


@pytest.mark.parametrize(
    ("model", "mase", "crps"), [("Chronos-2", 0.662, 0.556), ("TiRex", 0.683, 0.573)]
)
def test_aggregate_reproduces_the_hosted_leaderboard(
    model: str, mase: float, crps: float
) -> None:
    """The hosted results of two entries aggregate to their leaderboard scores."""
    all_tasks = tasks()
    baseline = read_scores(HOSTED / "seasonal_naive", all_tasks)
    scores = aggregate(read_scores(HOSTED / model, all_tasks), baseline)
    assert scores == pytest.approx({"MASE": mase, "CRPS": crps}, abs=6e-4)


def test_differences_are_relative_to_the_reference() -> None:
    """Rows pair by task; a difference is |A - B| / |B|, 0 where equal."""
    reference = pd.DataFrame(4.0, index=["a/H/short", "b/H/short"], columns=METRICS)
    results = reference.iloc[::-1].copy()
    results.loc["b/H/short", "MASE"] = 5.0
    relative = differences(results, reference)
    expected = pd.DataFrame(0.0, index=results.index, columns=list(METRICS))
    expected.loc["b/H/short", "MASE"] = 0.25
    pd.testing.assert_frame_equal(relative, expected)
