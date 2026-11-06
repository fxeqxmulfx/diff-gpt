"""Validate capacity-only task changes and unchanged native update/resume behavior."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
from torch.testing import assert_close

# isort: off
from research.current_grokking import run as _bootstrap  # noqa: F401
from lab.infrastructure.benchmarks.samplers import EpochSampler, gather
# isort: on

from research.cubic_capacity import run as runner
from research.cubic_capacity import trainer
from research.cubic_capacity.factories import (
    case_for,
    configuration,
    factories,
    plan,
    resolve,
)
from research.cubic_capacity.inventory import inventory
from research.current_grokking.data import arithmetic
from research.current_grokking.train import Stepper
from research.early99_cosine import trainer as native_trainer
from research.early99_cosine.factories import FactoryStepper

if TYPE_CHECKING:
    from pathlib import Path

    from research.current_grokking.protocol import Case

ARCHITECTURES = ["original-lr2", "normalized-lr2"]
INTERRUPT_AT = 4
COMPILER_LIMIT = 1e-3
EXPECTED_CELLS = 27


def specification(architecture: str) -> dict:
    """Select the declared largest shape for the real native-update checks."""
    return resolve(f"cubic-depth-width-{architecture}")


def fixture_case(architecture: str) -> Case:
    """Keep the largest model but use a small validation-only arithmetic dataset."""
    return replace(
        case_for(specification(architecture)),
        name="capacity-validation",
        data={"operation": "div", "prime": 7},
        batch=8,
        max_updates=12,
        observe=2,
        target=1.1,
        warmup=2,
        learning_rate=0.003,
    )


def test_all_declared_initializations_data_order_and_parameters() -> None:
    """Check all 27 architecture/rate/size/seed combinations without training."""
    rows = inventory()
    assert len(rows) == EXPECTED_CELLS
    assert all(row["initial_weights_buffers_rng_and_tying_verified"] for row in rows)


def test_declared_configuration_rejects_rate_or_architecture_changes() -> None:
    """Prevent an undeclared capacity or optimizer change from entering the runner."""
    original = plan()["configurations"][0]
    for field, value in [
        ("learning_rate_multiplier", 7.0),
        ("architecture", "unknown"),
        ("size", "current"),
    ]:
        changed = {**original, field: value}
        with pytest.raises(ValueError, match="Undeclared"):
            configuration(changed)
        with pytest.raises(ValueError, match="Undeclared"):
            case_for(changed)


def test_archive_rejects_changed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a started experiment bound to its original source bytes."""
    monkeypatch.setattr(runner, "RUNS", tmp_path)
    spec = plan()["configurations"][0]
    output = runner.prepare_output(spec)
    runner.prepare_output(spec)
    source = output / "experiment_source/research/cubic_capacity/factories.py"
    source.write_text(source.read_text() + "\n# altered\n")
    with pytest.raises(ValueError, match="Changed:"):
        runner.prepare_output(spec)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_actual_native_checkpoint_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    """Restore the real largest-size trainer and require exact trajectory equality."""
    spec, case = specification(architecture), fixture_case(architecture)
    monkeypatch.setattr(native_trainer, "CHECKPOINT_INTERVAL", 2)
    full = trainer.run(
        case, 4, tmp_path / "full", "cpu", specification=spec, compiled=False
    )
    save = native_trainer.save_checkpoint

    def interrupt(
        path: Path,
        model: object,
        stepper: object,
        sampler: object,
        completed: int,
        history: list,
        metadata: dict,
    ) -> None:
        save(path, model, stepper, sampler, completed, history, metadata)
        if completed == INTERRUPT_AT:
            message = "Validation interruption after checkpoint"
            raise RuntimeError(message)

    monkeypatch.setattr(native_trainer, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="Validation interruption"):
        trainer.run(
            case, 4, tmp_path / "resumed", "cpu", specification=spec, compiled=False
        )
    monkeypatch.setattr(native_trainer, "save_checkpoint", save)
    resumed = trainer.run(
        case,
        4,
        tmp_path / "resumed",
        "cpu",
        specification=spec,
        compiled=False,
        resume=True,
    )
    first = torch.load(
        tmp_path / "full" / case.name / "seed4/checkpoint.pt", weights_only=True
    )
    second = torch.load(
        tmp_path / "resumed" / case.name / "seed4/checkpoint.pt", weights_only=True
    )
    for field in ["model", "optimizer", "sampler", "cpu_rng"]:
        assert_close(first[field], second[field], rtol=0, atol=0)
    assert full["endpoint"]["validation"] == resumed["endpoint"]["validation"]
    for field in ["actual_examples_lower", "actual_examples_upper"]:
        assert resumed["execution_journal"][field] == full["execution_journal"][field]
    assert resumed["execution_journal"]["resumed_segments"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_cuda_native_capture_updates_and_nonempty_restore(
    tmp_path: Path, architecture: str
) -> None:
    """Separate capture from compiler rounding at the largest declared model size."""
    torch.compiler.reset()
    torch.set_float32_matmul_precision("highest")
    spec, case = specification(architecture), fixture_case(architecture)
    cfg, maker = configuration(spec), factories(spec)
    task = arithmetic(case, "cuda")
    torch.manual_seed(11)
    model = maker.model(task, case, "cuda")
    reference, eager = copy.deepcopy(model), copy.deepcopy(model)
    actual = FactoryStepper(
        model,
        task,
        case,
        "cuda",
        configuration=cfg,
        factories=maker,
        journal_path=tmp_path / "journal.json",
        compiled=True,
    )
    compiled = Stepper(reference, task, cfg.training_case(case), "cuda", compiled=True)
    uncompiled = Stepper(eager, task, cfg.training_case(case), "cuda", compiled=False)
    sampler = EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=10011
    )
    initial = copy.deepcopy(model.state_dict())
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    actual.prepare(sampler.sizes())
    assert_close(model.state_dict(), initial, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert not actual.execution_journal.path.exists()
    assert all(float(s["step"]) == 0 for s in actual.optimizer.state.values())

    def compare_updates(
        indices: torch.Tensor, completed: int, tested: FactoryStepper
    ) -> None:
        for stepper in [compiled, uncompiled]:
            stepper.rate.fill_(
                cfg.learning_rate_at(case, completed, len(task.splits["train"]))
            )
            stepper.update(indices.to("cuda"))
        assert_close(
            tested.model.state_dict(), reference.state_dict(), rtol=5e-4, atol=4e-6
        )
        assert_close(
            tested.optimizer.state_dict(),
            compiled.optimizer.state_dict(),
            rtol=5e-4,
            atol=4e-6,
        )
        tensors = list(zip(tested.model.parameters(), eager.parameters(), strict=True))
        for first, second in zip(
            tested.optimizer.state.values(),
            uncompiled.optimizer.state.values(),
            strict=True,
        ):
            assert first["step"].item() == second["step"].item()
            tensors.extend(
                (first[field], second[field]) for field in ["exp_avg", "exp_avg_sq"]
            )
        for first, second in tensors:
            assert torch.isfinite(first).all()
            assert torch.isfinite(second).all()
            relative = (
                first.double() - second.double()
            ).norm() / second.double().norm().clamp_min(1e-30)
            assert relative.item() <= COMPILER_LIMIT

    sizes = []
    for completed in range(case.max_updates):
        indices = gather(sampler.next()[0])
        sizes.append(len(indices))
        actual.step(indices, completed)
        if completed == 0:
            assert_close(model.state_dict(), initial, rtol=0, atol=0)
        compare_updates(indices, completed, actual)
        journal = actual.execution_journal.summary(completed + 1)
        assert (
            journal["actual_examples_lower"]
            == journal["actual_examples_upper"]
            == sum(sizes)
        )
    assert min(sizes) < case.batch
    continued_model = copy.deepcopy(model)
    continued = FactoryStepper(
        continued_model,
        task,
        case,
        "cuda",
        configuration=cfg,
        factories=maker,
        journal_path=actual.execution_journal.path,
        compiled=True,
    )
    continued.optimizer.load_state_dict(actual.optimizer.state_dict())
    for group in continued.optimizer.param_groups:
        group["lr"] = continued.rate
    state, moments = (
        copy.deepcopy(continued_model.state_dict()),
        copy.deepcopy(continued.optimizer.state_dict()),
    )
    journal_bytes = continued.execution_journal.path.read_bytes()
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    continued.prepare(sampler.sizes())
    assert_close(continued_model.state_dict(), state, rtol=0, atol=0)
    assert_close(continued.optimizer.state_dict(), moments, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert continued.execution_journal.path.read_bytes() == journal_bytes
    indices = gather(sampler.next()[0])
    sizes.append(len(indices))
    continued.step(indices, case.max_updates)
    compare_updates(indices, case.max_updates, continued)
    journal = continued.execution_journal.summary(case.max_updates + 1)
    assert (
        journal["actual_examples_lower"]
        == journal["actual_examples_upper"]
        == sum(sizes)
    )
