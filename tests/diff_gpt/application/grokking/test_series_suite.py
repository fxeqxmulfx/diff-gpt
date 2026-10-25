"""Verify fast-first selection and forecasting without true future inputs."""

import csv
import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.application.grokking import series_suite
from diff_gpt.application.grokking.series import forecast_one_row
from diff_gpt.application.grokking.series_data import DataConfig, dataset
from diff_gpt.application.grokking.series_suite import quality, selection_key

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from diff_gpt.application.grokking.series import Config


def _curve(directory: Path, *, known: bool, anchor: bool = False) -> None:
    directory.mkdir()
    reference = {"x": {"floor_loss": 0.0 if known else None, "baseline_loss": 2.0}}
    metadata = {
        "config": {"data": {"dataset": "div97" if anchor else "weather_formula"}},
        "references": {"test": reference, "validation": reference},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    rows = [
        {
            "epoch": 0,
            "seconds": 0,
            "training_flops": 0,
            "test_loss": 2.5,
            "validation_loss": 2.5,
            "test_skill": -0.25,
            "test_accuracy": 0.01,
        },
        {
            "epoch": 10,
            "seconds": 2,
            "training_flops": 100,
            "test_loss": 0.2,
            "validation_loss": 1.7,
            "test_skill": 0.9,
            "test_accuracy": 0.5,
        },
        {
            "epoch": 20,
            "seconds": 4,
            "training_flops": 200,
            "test_loss": 0.01,
            "validation_loss": 1.9,
            "test_skill": 0.995,
            "test_accuracy": 0.995,
        },
    ]
    with (directory / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_fast_crossing_beats_better_but_slower_final_loss(tmp_path: Path) -> None:
    """Ranking follows compute to the target, with no reward for a plateau."""
    directory = tmp_path / "run"
    _curve(directory, known=True)
    fast = quality(directory)
    slower = {
        **fast,
        "crossings": {
            key: {**value, "training_flops": value["training_flops"] * 100}
            for key, value in fast["crossings"].items()
        },
        "best_selection_loss": 0.0001,
    }
    assert selection_key(fast) < selection_key(slower)
    assert fast["crossings"]["0.9"]["epoch"] == pytest.approx(10)
    assert fast["crossings"]["0.99"]["training_flops"] == pytest.approx(200)


def test_screen_cannot_use_crossings_after_its_budget(tmp_path: Path) -> None:
    """An extended trajectory does not retrospectively pass its fast screen."""
    directory = tmp_path / "run"
    _curve(directory, known=True)
    limited = quality(directory, budget=10)
    assert limited["crossings"]["0.99"] is None
    assert limited["crossings"]["0.9"] is not None


def test_anchor_selection_uses_accuracy(tmp_path: Path) -> None:
    """Noisy-anchor comparisons keep the original heldout accuracy definition."""
    directory = tmp_path / "run"
    _curve(directory, known=True, anchor=True)
    measured = quality(directory)
    assert measured["target_metric"] == "test_accuracy"
    assert measured["crossings"]["0.9"]["epoch"] == pytest.approx(20)


def test_unknown_floor_selection_uses_validation(tmp_path: Path) -> None:
    """The test loss does not choose a real-series configuration or epoch."""
    directory = tmp_path / "run"
    _curve(directory, known=False)
    measured = quality(directory)
    assert measured["selection_split"] == "validation"
    assert measured["best_epoch"] == pytest.approx(10)
    assert measured["best_selection_loss"] == pytest.approx(1.7)
    assert measured["target_direction"] == "lower"


def test_failed_replication_uses_next_fast_screened_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-seed success must survive confirmation before being retained."""
    confirmations = []
    original_sample_size, robust_sample_size = 512, 2048

    def fake_screen(
        configs: Sequence[Config], output: Path, stage: str, device: str
    ) -> Config:
        assert device == "cpu"
        selected, alternative = configs[0], configs[-1]
        measured = []
        for config in (selected, alternative):
            for seed in series_suite.SEEDS:
                seeded = replace(config, data=replace(config.data, seed=seed))
                directory = output / stage / seeded.name()
                directory.parent.mkdir(parents=True, exist_ok=True)
                _curve(directory, known=False)
                if (
                    config == selected
                    and config.data.order == 1
                    and seed == series_suite.SEEDS[1]
                ):
                    path = directory / "history.csv"
                    path.write_text(path.read_text().replace("1.7", "1.9"))
            measured.append(
                {
                    "config": asdict(config),
                    **quality(output / stage / config.name()),
                }
            )
        (output / f"screen-{stage}.json").write_text(json.dumps(measured))
        (output / f"selection-{stage}.json").write_text(
            json.dumps({"selected": asdict(selected)})
        )
        return selected

    monkeypatch.setattr(series_suite, "screen", fake_screen)
    monkeypatch.setattr(
        series_suite,
        "confirm",
        lambda config, *_: confirmations.append(config),
    )
    series_suite.temporal(tmp_path, "cpu")
    assert not (tmp_path / "selection-weather-k0-robust.json").exists()
    robust = json.loads((tmp_path / "selection-weather-k1-robust.json").read_text())
    assert robust["confirmed_fast_on_three_seeds"]
    assert robust["selected"]["data"]["max_train"] == robust_sample_size
    original = json.loads((tmp_path / "selection-weather-k1-fast.json").read_text())
    assert original["selected"]["data"]["max_train"] == original_sample_size
    assert len(confirmations) == len(series_suite.SEEDS)


class NextToken:
    """Return the token one above the last supplied value and retain inputs."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        """Start with an empty observation log."""
        self.contexts = []

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Use the final observed token, without consulting any target."""
        self.contexts.append(x.tolist())
        logits = torch.full((*x.shape, 64), -100.0)
        logits.scatter_(-1, (x + 1)[..., None], 100.0)
        return logits, None


def test_forecast_generates_future_columns_instead_of_reading_them() -> None:
    """The second future column receives a generated first column as context."""
    model = NextToken()
    tensors = {
        "x": torch.tensor([[2, 5, 9]]),
        "clean_y": torch.tensor([[-100, 9, 10]]),
        "truth": torch.tensor([[9.0, 10.0]]),
        "base": torch.tensor([[32.0, 32.0]]),
        "step": torch.ones(1, 2),
    }
    result = forecast_one_row(model, tensors, ["a", "b"], 64)
    assert model.contexts == [[[2, 5]], [[2, 5, 6]]]
    assert not result["teacher_forced_future_columns"]
    assert result["mse"] == pytest.approx(9)
    assert result["mae"] == pytest.approx(3)


def test_training_sample_cap_does_not_change_formula_calibration(
    tmp_path: Path,
) -> None:
    """A data-amount screen keeps its calibration, grid and raw pair assignment."""
    rng = np.random.default_rng(41)
    rh = rng.integers(20, 100, 2000)
    vapor = rng.integers(5, 40, 2000)
    path = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "date": np.arange(len(rh)),
            "rh (%)": rh,
            "VPmax (mbar)": vapor,
            "VPact (mbar)": rh * vapor / 100,
        }
    ).to_csv(path, index=False)
    base = DataConfig(source=str(path), max_train=128, max_eval=64, calibration_rows=32)
    small = dataset(base)
    large = dataset(replace(base, max_train=512))
    assert small.metadata["calibration_ids"] == large.metadata["calibration_ids"]
    for key in ("mean", "std", "domain", "scale", "full_table_oracle"):
        assert small.metadata[key] == large.metadata[key]
    np.testing.assert_array_equal(small.splits["test"].ids, large.splits["test"].ids)
