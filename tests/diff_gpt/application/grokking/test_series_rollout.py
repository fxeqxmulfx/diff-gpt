"""Prove that a multi-row forecast never receives the observed future."""

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from benchmarks.grokking import series_rollout
from benchmarks.grokking.series import Config
from benchmarks.grokking.series_data import DataConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_rollout_feeds_generated_rows_and_baselines_use_only_the_past(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wildly different actual future cannot appear in a model input."""
    values = np.arange(200, dtype=np.float64)[:, None]
    values[50:] = 1000
    config = Config(data=DataConfig(dataset="m4_hourly", context=2))
    metadata = {
        "config": asdict(config),
        "source_sha256": {},
        "data": {
            "source_sha256": "fixture",
            "fit": [{"mean": [0.0], "std": [1.0], "domain": [1000.0]}],
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    for name in ("checkpoint.pt", "best-validation.pt"):
        (tmp_path / name).write_bytes(b"fixture checkpoint")
    np.savez(
        tmp_path / "frozen_splits.npz",
        test__ids=np.array([[0, 48, 50]]),
        train__x=np.tile(np.arange(64)[:, None], (1, 2)),
        train__y=np.arange(64)[:, None],
        train__column=np.zeros((64, 1), dtype=int),
    )
    monkeypatch.setattr(
        series_rollout,
        "_raw_series",
        lambda _: ([values], ["x"], {"source_sha256": "fixture"}, 24),
    )
    monkeypatch.setattr(
        series_rollout, "load_checkpoint", lambda *_: {"model": {}, "epoch": 1}
    )

    class Model:
        def load_state_dict(self, _: dict) -> None:
            pass

        def eval(self) -> None:
            pass

    monkeypatch.setattr(series_rollout, "build", lambda *_: Model())
    contexts = []

    class Predictor:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def predict_many(
            self, frames: list[pd.DataFrame], **_: object
        ) -> list[pd.DataFrame]:
            contexts.extend(frame.to_numpy().copy() for frame in frames)
            return [
                pd.concat(
                    [frame, pd.DataFrame([frame.iloc[-1] + 1])], ignore_index=True
                )
                for frame in frames
            ]

    monkeypatch.setattr(series_rollout, "DiffGPT", Predictor)
    result = series_rollout.rollout(tmp_path, "cpu", maximum=1)
    np.testing.assert_array_equal(contexts[0][:, 0], [48, 49])
    np.testing.assert_array_equal(contexts[1][:, 0], [49, 50])
    assert max(float(context.max()) for context in contexts) < values[50, 0]
    scores = result["scores"]
    assert scores["final"]["raw"]["mse"] == pytest.approx(
        np.mean((np.arange(50, 98) - 1000) ** 2)
    )
    assert scores["seasonal_naive"]["raw"]["mse"] == pytest.approx(
        np.mean((np.tile(np.arange(26, 50), 2) - 1000) ** 2)
    )
    assert not result["teacher_forced_future"]
