"""Check the synthetic pretraining script: its holdout scores and a tiny run."""

import csv
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from benchmarks.synthetic_pretrain import (
    TEMPOPFN_WEIGHTS,
    forecast_scores,
    holdout_windows,
    main,
)
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.model.gpt import GPT

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.synthetic_pretrain import Windows

CONTEXT = 24
HORIZON = 8
PER_SOURCE = 4
# A forecast halfway from the naive one to the truth halves both errors.
HALFWAY = 0.5


def _series() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "a": np.cumsum(rng.normal(size=(3, 40)), axis=1),
        "b": 5.0 + np.cumsum(rng.normal(size=(2, 40)), axis=1),
    }


def _windows() -> Windows:
    return holdout_windows(_series(), PER_SOURCE, CONTEXT, HORIZON, seed=0)


def _samples(point: np.ndarray) -> np.ndarray:
    """Return five samples of the (N, H) `point`: each quantile is the point."""
    return np.repeat(point[:, None, :], 5, axis=1)


def test_holdout_windows_are_slices_of_the_series() -> None:
    """Check that each window is a slice of a series of its source, in order."""
    series = _series()
    windows = _windows()
    assert windows.sources.tolist() == [0] * PER_SOURCE + [1] * PER_SOURCE
    assert windows.contexts.shape == (2 * PER_SOURCE, CONTEXT)
    assert windows.futures.shape == (2 * PER_SOURCE, HORIZON)
    names = list(series)
    for k, context, future in zip(*windows, strict=True):
        window = np.concatenate([context, future])
        data = series[names[k]]
        assert any(
            np.array_equal(row[o : o + len(window)], window)
            for row in data
            for o in range(data.shape[1] - len(window) + 1)
        )


def test_scores_are_ratios_to_the_naive_forecast() -> None:
    """
    Check the scores of the naive forecast and of one halfway to the truth.

    Samples that all equal a point forecast make every quantile that point,
    so the naive forecast scores 1 and the halfway one 0.5, on both scores.
    """
    windows = _windows()
    naive = np.repeat(windows.contexts[:, -1:], HORIZON, axis=1)
    halfway = naive + HALFWAY * (windows.futures - naive)
    for point, expected in ((naive, 1.0), (halfway, HALFWAY)):
        scores = forecast_scores(_samples(point), windows, ["a", "b"])
        assert scores["mase"] == pytest.approx(expected)
        assert scores["crps"] == pytest.approx(expected)
        assert set(scores["sources"]) == {"a", "b"}


def test_scores_skip_missing_values() -> None:
    """A missing future value counts for nothing; naive is the last value seen."""
    windows = _windows()
    windows.futures[0, 3] = np.nan
    windows.contexts[1, -1] = np.nan
    last = windows.contexts[:, -1].copy()
    last[1] = windows.contexts[1, -2]
    naive = np.repeat(last[:, None], HORIZON, axis=1)
    scores = forecast_scores(_samples(naive), windows, ["a", "b"])
    assert scores["mase"] == pytest.approx(1.0)
    assert scores["crps"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "precision",
    [[], ["--train-dtype", "bfloat16", "--score-dtype", "float16"]],
)
def test_a_tiny_run_writes_its_curve_and_checkpoints(
    tmp_path: Path,
    precision: list[str],
) -> None:
    """
    Check a CPU run of four updates on a corpus of two sources.

    The history holds the evaluations after 0, 2 and 4 updates with their
    FLOPs, each one scored, and best.bin loads into the model config.json
    describes. Trained in mixed precision and scored in float16 too, the
    checkpoints hold the float32 master weights.
    """
    rng = np.random.default_rng(0)
    corpus = tmp_path / "corpus"
    for split, n in (("train", 6), ("holdout", 3)):
        (corpus / split).mkdir(parents=True)
        for name in ("sinewave", "ou_process"):
            walks = np.cumsum(rng.normal(size=(n, 80)), axis=1)
            np.save(corpus / split / f"{name}.npy", walks.astype(np.float32))
    output = tmp_path / "run"
    main(
        [
            *("--corpus", str(corpus), "--output", str(output)),
            *("--block-size", "32", "--width", "16", "--layers", "1"),
            *("--heads", "2", "--vocab", "64", "--domain-windows", "64"),
            *("--batch-size", "4", "--max-iters", "4", "--eval-interval", "2"),
            *("--eval-iters", "1", "--warmup", "1", "--score-windows", "2"),
            *("--horizon", "8", "--num-samples", "4", "--device", "cpu"),
            *precision,
        ],
    )
    config = json.loads((output / "config.json").read_text())
    with (output / "history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = [0, 2, 4]
    assert [int(row["step"]) for row in rows] == steps
    flops = config["flops_per_token"] * config["tokens_per_update"]
    assert [int(row["train_flops"]) for row in rows] == [flops * s for s in steps]
    assert all(row["crps"] for row in rows)
    assert len((output / "scores.jsonl").read_text().splitlines()) == len(rows)
    assert config["sources"]["ou_process"]["weight"] == TEMPOPFN_WEIGHTS["ou_process"]
    model = DiffGPT(
        model=GPT(vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=1),
        order_of_derivative=1,
        domain_of_definition=np.array(config["domain_of_definition"]),
        tokenizer=config["tokenizer"],
    )
    model.load(str(output / "best.bin"))
    checkpoint = torch.load(output / "best.bin", weights_only=True)
    assert {v.dtype for v in checkpoint["model"].values()} == {torch.float32}
    with pytest.raises(FileExistsError):
        main(["--corpus", str(corpus), "--output", str(output)])
