"""Verify stress-test controls, analytical witnesses and the frozen model link."""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch.testing import assert_close

from research.early99.experiments.factories import (
    peri_rms_scaled_ssmax_lerope_log_angle_model,
)
from research.transformer_stress.common import full_stack
from research.transformer_stress.components import normalization, stream_mixing
from research.transformer_stress.residual import (
    ResidualNetwork,
    checks,
    datasets,
    evaluate,
    train_cell,
)
from research.transformer_stress.run import PLAN

if TYPE_CHECKING:
    from pathlib import Path

ROUNDING_TOLERANCE = 1e-6
LARGE_RELATIVE_ERROR = 0.9
COLUMN_ERROR_WITNESS = 0.01


def test_exact_stack_matches_frozen_benchmark_factory() -> None:
    """Prevent a test surrogate from silently replacing the requested GPT."""
    settings = json.loads(PLAN.read_text())["components"]
    task = SimpleNamespace(vocab=settings["full_stack_vocab"], context=8)
    case = SimpleNamespace(
        width=settings["full_stack_width"], layers=2, heads=settings["full_stack_heads"]
    )
    torch.manual_seed(9)
    tested = full_stack(settings, case.layers, task.context)
    torch.manual_seed(9)
    frozen = peri_rms_scaled_ssmax_lerope_log_angle_model(task, case, "cpu").model
    assert_close(tested.state_dict(), frozen.state_dict(), atol=0, rtol=0)
    assert tested.token_embedding_table.weight is tested.lm_head.weight
    tokens = torch.tensor([[2, 2, 4, 6, 1, 5, 7, 8]])
    assert_close(tested(tokens)[0], frozen(tokens)[0], atol=0, rtol=0)


def test_gate_and_teacher_controls() -> None:
    """Check finite-epsilon gate suppression and the identity error reference."""
    settings = json.loads(PLAN.read_text())["residual_learning"]
    splits = datasets(settings)
    assert checks(settings, splits)["common_initialization"]
    identity = ResidualNetwork(settings, settings["branches"][0], "peri_rezero")
    for strength in settings["teacher_strengths"]:
        result = evaluate(identity, splits["heldout"], strength)
        assert result["relative_mse"] == 1


def test_overflow_cannot_pass_as_finite_normalization() -> None:
    """A bounded finite output can still disagree completely with real RMS."""
    rows = normalization({"dimensions": [8], "rms_scales": [1, 1e30]})
    controls = [row for row in rows if row["scale"] == 1]
    stressed = [row for row in rows if row["scale"] != 1]
    assert all(row["forward_relative_error"] < ROUNDING_TOLERANCE for row in controls)
    assert all(
        not row["output_finite"] or row["forward_relative_error"] > LARGE_RELATIVE_ERROR
        for row in stressed
    )


def test_stochastic_mixing_does_not_imply_invertibility() -> None:
    """Retain a zero-singular-value witness and finite-Sinkhorn column error."""
    rows = stream_mixing()
    exact = rows[0]
    assert exact["disagreement_before"] > 0
    assert exact["disagreement_after"] == 0
    assert_close(torch.tensor(exact["spectral_norm"]), torch.tensor(1.0))
    first = next(
        row
        for row in rows[1:]
        if row["iterations"] == 1
        and row["logit_scale"] == 1
        and row["implementation"] == "literal_exp"
    )
    assert first["row_sum_max_error"] < ROUNDING_TOLERANCE
    assert first["column_sum_max_error"] > COLUMN_ERROR_WITNESS


def test_training_records_issued_updates_and_all_observations(tmp_path: Path) -> None:
    """Count the zero-rate first update and keep full-budget checkpoint/history."""
    settings = json.loads(PLAN.read_text())["residual_learning"]
    settings.update(
        {
            "updates": 2,
            "observe_updates": 1,
            "train_examples": 64,
            "heldout_examples": 64,
        }
    )
    splits = datasets(settings)
    result = train_cell(settings, splits, "peri_sqrt", 4, 0.25, 0, tmp_path / "cell")
    assert result["completed_updates"] == settings["updates"]
    history = json.loads((tmp_path / "cell/history.json").read_text())
    assert [row["updates"] for row in history] == [0, 1, 2]
    assert history[0]["heldout"] == history[1]["heldout"]
    checkpoint = torch.load(tmp_path / "cell/checkpoint.pt", weights_only=False)
    assert all(
        state["step"].item() == settings["updates"]
        for state in checkpoint["optimizer"]["state"].values()
    )
