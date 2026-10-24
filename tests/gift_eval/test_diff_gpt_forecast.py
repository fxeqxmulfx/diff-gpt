"""Check the DiffGPT forecaster of the GIFT-Eval windows on a tiny pretraining run."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from benchmarks.gift_eval.diff_gpt_forecast import chunk_plan, main
from benchmarks.gift_eval.files import QUANTILE_LEVELS
from benchmarks.synthetic_pretrain import main as pretrain

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
BLOCK_SIZE = 32
SHORT = 8  # one pass of the tiny model
LONG = 40  # three passes of it, from min_context = 16 rows: 17, 17 and 6 points
# The contexts without an observed value in their first pass, or with one row,
# and their constant forecasts.
CONSTANT = {3: 0.0, 4: 7.0, 5: 3.0}


def test_chunk_plan_fills_the_block() -> None:
    """Each pass fills the block, and the passes draw the whole horizon."""
    assert chunk_plan(48, 512, 1, 256) == [(465, 48)]
    assert chunk_plan(600, 512, 1, 257) == [(257, 256), (257, 256), (425, 88)]
    for horizon in (1, 17, 18, 900):
        plan = chunk_plan(horizon, BLOCK_SIZE, 1, 16)
        assert sum(points for _, points in plan) == horizon
        assert all(rows - 1 + points == BLOCK_SIZE for rows, points in plan)
    for min_context in (1, BLOCK_SIZE + 1):
        with pytest.raises(ValueError, match="min_context must lie in"):
            chunk_plan(10, BLOCK_SIZE, 1, min_context)


def _contexts() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    walk = 300.0 + 50.0 * np.cumsum(rng.normal(size=100))
    gappy = walk.copy()
    gappy[40:60] = np.nan
    gappy[70] = np.inf
    return [
        walk,
        gappy,
        walk[:3],
        np.full(20, np.nan),  # no observed value: 0
        np.array([7.0]),  # no row past the prefix: its value
        np.concatenate([[1.0, 2.0, 3.0], np.full(30, np.nan)]),  # the last one: 3
    ]


def _write_windows(folder: Path, config: str, horizon: int) -> None:
    """Write `_contexts` as leaderboards' export writes the windows of `config`."""
    contexts = _contexts()
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(
        folder / f"{config.replace('/', '__')}.npz",
        values=np.concatenate(contexts).astype(np.float32),
        offsets=np.cumsum([0, *map(len, contexts)]).astype(np.int64),
        item_id=np.array([f"item{i}" for i in range(len(contexts))]),
        forecast_start=np.array(["2020-01-01 00:00"] * len(contexts)),
        config=np.array(config),
        split=np.array("val"),
        prediction_length=np.array(horizon, dtype=np.int64),
        freq=np.array("H"),
        season_length=np.array(24, dtype=np.int64),
    )


def _tiny_run(tmp_path: Path) -> Path:
    """Pretrain a tiny model for two updates on random walks; return its folder."""
    rng = np.random.default_rng(0)
    corpus = tmp_path / "corpus"
    for split, n in (("train", 6), ("holdout", 3)):
        (corpus / split).mkdir(parents=True)
        walks = np.cumsum(rng.normal(size=(n, 80)), axis=1)
        np.save(corpus / split / "ou_process.npy", walks.astype(np.float32))
    run = tmp_path / "pilot"
    pretrain(
        [
            *("--corpus", str(corpus), "--output", str(run)),
            *("--block-size", str(BLOCK_SIZE), "--width", "16", "--layers", "1"),
            *("--heads", "2", "--vocab", "64", "--domain-windows", "64"),
            *("--batch-size", "4", "--max-iters", "2", "--eval-interval", "2"),
            *("--eval-iters", "1", "--warmup", "1", "--score-windows", "2"),
            *("--horizon", "8", "--num-samples", "4", "--device", "cpu"),
        ],
    )
    return run


def test_forecasts_of_a_tiny_run(tmp_path: Path) -> None:
    """
    Check the forecasts of one-pass and three-pass horizons, and the reruns.

    The quantiles rise with the level, constant contexts get their last
    value, a rerun skips what exists, --overwrite redraws it the same, and
    other settings must go to another --model.
    """
    run = _tiny_run(tmp_path)
    windows = tmp_path / "windows"
    _write_windows(windows / "val", "toy/H/short", SHORT)
    _write_windows(windows / "val", "toy/H/long", LONG)
    forecasts = tmp_path / "forecasts"
    argv = [
        *("--run", str(run), "--windows", str(windows)),
        *("--forecasts", str(forecasts), "--num-samples", "4", "--device", "cpu"),
    ]
    main(argv)
    folder = forecasts / "diff_gpt-pilot" / "val"
    record = json.loads((folder / "forecast_config.json").read_text())
    drawn: dict[str, dict[str, np.ndarray]] = {}
    for config, horizon, passes in (("toy/H/short", SHORT, 1), ("toy/H/long", LONG, 3)):
        assert record["configs"][config]["constant"] == len(CONSTANT)
        assert len(record["configs"][config]["plan"]) == passes
        with np.load(folder / f"{config.replace('/', '__')}.npz") as file:
            drawn[config] = {key: file[key] for key in file.files}
        quantiles, mean = drawn[config]["quantiles"], drawn[config]["mean"]
        assert quantiles.shape == (len(_contexts()), len(QUANTILE_LEVELS), horizon)
        assert mean.shape == (len(_contexts()), horizon)
        assert np.isfinite(quantiles).all()
        assert np.isfinite(mean).all()
        assert (np.diff(quantiles, axis=1) >= 0).all()
        assert drawn[config]["item_id"].tolist() == [f"item{i}" for i in range(6)]
        for i, value in CONSTANT.items():
            assert (quantiles[i] == value).all()
            assert (mean[i] == value).all()
    main(argv)  # every forecast exists: nothing is redrawn
    main([*argv, "--overwrite"])
    for config, arrays in drawn.items():
        with np.load(folder / f"{config.replace('/', '__')}.npz") as file:
            for key, array in arrays.items():
                assert np.array_equal(file[key], array), (config, key)
    with pytest.raises(ValueError, match="other settings"):
        main([*argv, "--seed", "1"])


def _forecast_files(folder: Path) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as file:
            yield path.name, {key: file[key] for key in file.files}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU's checkpoint")
def test_a_process_without_a_gpu_forecasts_with_a_gpu_runs_weights(
    tmp_path: Path,
) -> None:
    """
    The weights of a GPU run load onto the CPU of a process that sees no GPU.

    A run trained on a GPU saves CUDA tensors, which torch.load would read
    back onto the GPU. The forecasts of such a copy of the tiny run, drawn
    with CUDA_VISIBLE_DEVICES empty, equal those of the run itself.
    """
    run = _tiny_run(tmp_path)
    on_gpu = tmp_path / "gpu" / run.name
    on_gpu.mkdir(parents=True)
    shutil.copy(run / "config.json", on_gpu / "config.json")
    saved = torch.load(run / "best.bin", weights_only=True)
    saved["model"] = {name: value.cuda() for name, value in saved["model"].items()}
    torch.save(saved, on_gpu / "best.bin")
    windows = tmp_path / "windows"
    _write_windows(windows / "val", "toy/H/short", SHORT)
    argv = ["--windows", str(windows), "--num-samples", "4", "--device", "cpu"]
    main([*argv, "--run", str(run), "--forecasts", str(tmp_path / "cpu_run")])
    subprocess.run(  # noqa: S603 - this interpreter and the module, no shell.
        [
            sys.executable,
            *("-m", "benchmarks.gift_eval.diff_gpt_forecast"),
            *argv,
            *("--run", str(on_gpu), "--forecasts", str(tmp_path / "gpu_run")),
        ],
        cwd=ROOT,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )
    drawn = dict(_forecast_files(tmp_path / "cpu_run" / "diff_gpt-pilot" / "val"))
    again = dict(_forecast_files(tmp_path / "gpu_run" / "diff_gpt-pilot" / "val"))
    assert drawn
    assert drawn.keys() == again.keys()
    for name, arrays in drawn.items():
        for key, array in arrays.items():
            assert np.array_equal(again[name][key], array), (name, key)
