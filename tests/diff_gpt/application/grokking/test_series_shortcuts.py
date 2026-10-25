"""Detect input copying without fitting any baseline to heldout answers."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks.grokking.series_shortcuts import formula_shortcuts

if TYPE_CHECKING:
    from pathlib import Path


def test_single_operand_controls_fit_only_training_and_detect_copying(
    tmp_path: Path,
) -> None:
    """A heldout pair can be solved by copying, while changed tests cannot alter fit."""
    arrays = {}
    for name, inputs, targets in (
        ("train", [[1, 2], [1, 2], [3, 4], [3, 4]], [2, 2, 4, 4]),
        ("test", [[3, 2], [1, 4]], [2, 4]),
    ):
        size = len(inputs)
        arrays.update(
            {
                f"{name}__x": np.array(inputs),
                f"{name}__y": np.column_stack((np.full(size, -100), targets)),
                f"{name}__truth": np.array(targets, dtype=float)[:, None],
                f"{name}__base": np.full((size, 1), 32.0),
                f"{name}__step": np.ones((size, 1)),
            }
        )
    path = tmp_path / "frozen_splits.npz"
    np.savez(path, **arrays)
    metadata = {
        "config": {"data": {"dataset": "weather_formula", "vocab_size": 64}},
        "data": {"input_columns": ["a", "b"]},
    }
    before = formula_shortcuts(tmp_path, metadata)
    copy = before["operands"]["b"]["native_token_copy"]["test"]
    assert copy["accuracy"] == pytest.approx(1)
    assert copy["mse"] == pytest.approx(0)
    assert (
        before["operands"]["b"]["training_fitted_conditional"]["test"]["loss"]
        < before["operands"]["a"]["training_fitted_conditional"]["test"]["loss"]
    )
    arrays["test__y"][:, -1] = 50
    np.savez(path, **arrays)
    after = formula_shortcuts(tmp_path, metadata)
    for operand in ("a", "b"):
        assert (
            before["operands"][operand]["training_probability_sha256"]
            == after["operands"][operand]["training_probability_sha256"]
        )
    assert after["operands"]["b"]["native_token_copy"]["test"][
        "accuracy"
    ] == pytest.approx(0)
