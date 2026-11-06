"""Apply the unchanged native validation harness to the isolated usual-gain arm."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from research.cubic_usual_gain import readiness
from research.cubic_usual_gain import run as runner
from research.cubic_usual_gain.factories import (
    case_for,
    configuration,
    factories,
    plan,
    train,
)
from research.cubic_usual_gain.inventory import inventory
from tests.diff_gpt.application import test_cubic_capacity as native_checks

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_restore_only_gain_preserves_initialization_and_every_gradient(
    seed: int,
) -> None:
    """Restore the prior inverse-width model and check exact parent equivalence."""
    record = inventory(seed)
    assert record["matched_weights_rng_buffers_and_tying"]
    assert record["restored_inverse_forward_all_parameter_gradients_exact"]
    assert record["optimizer_updates"] == 0


def bind_native_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep original tolerances and checks, replacing only model/case construction."""
    fixture = replace(
        case_for(),
        name="usual-gain-validation",
        data={"operation": "div", "prime": 7},
        batch=8,
        max_updates=12,
        observe=2,
        target=1.1,
        warmup=2,
        learning_rate=0.003,
    )
    monkeypatch.setattr(
        native_checks, "specification", lambda _: plan()["configuration"]
    )
    monkeypatch.setattr(native_checks, "fixture_case", lambda _: fixture)
    monkeypatch.setattr(native_checks, "configuration", lambda _: configuration())
    monkeypatch.setattr(native_checks, "factories", lambda _: factories())

    def train_declared(*args: object, specification: dict, **kwargs: object) -> dict:
        assert specification == plan()["configuration"]
        return train(*args, **kwargs)

    monkeypatch.setattr(native_checks, "trainer", SimpleNamespace(run=train_declared))


def test_native_checkpoint_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require exact parameters, moments, sampler and RNG after a real interruption."""
    bind_native_checks(monkeypatch)
    native_checks.test_actual_native_checkpoint_continuation(
        tmp_path, monkeypatch, "usual"
    )


def test_capacity_completion_and_source_archive_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject premature training and drift in a previously archived experiment."""
    with monkeypatch.context() as local:
        local.setattr(readiness, "ROOT", tmp_path)
        with pytest.raises(ValueError, match="Finish and audit all capacity"):
            readiness.capacity_completed()
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    output = runner.prepare_output()
    runner.prepare_output()
    source = output / "experiment_source/research/cubic_usual_gain/factories.py"
    source.write_text(source.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="Changed:"):
        runner.prepare_output()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_capture_updates_and_nonempty_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse the capture/compiler, tail batch and restored-state checks unchanged."""
    bind_native_checks(monkeypatch)
    native_checks.test_cuda_native_capture_updates_and_nonempty_restore(
        tmp_path, "usual"
    )
