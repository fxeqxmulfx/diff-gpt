"""Validate the declared schedule and journal without changing native update math."""

from __future__ import annotations

import ast
import copy
import inspect
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
from lab.infrastructure.benchmarks.samplers import EpochSampler, gather
from torch.testing import assert_close

from research.current_grokking.data import arithmetic
from research.current_grokking.protocol import Case
from research.early99.experiments.factories import FactoryStepper as ParentStepper
from research.early99.experiments.trainer import run as parent_run
from research.early99_cosine import trainer
from research.early99_cosine.factories import (
    Configuration,
    FactoryStepper,
    configured_factories,
)
from research.early99_dimension_gain.factories import dimension_model
from research.early99_execution.journal import verify_saved_journal

if TYPE_CHECKING:
    from pathlib import Path

INTERRUPT_AT = 4


def small_case() -> Case:
    """Use the real arithmetic evaluator and a nonwrapping short final batch."""
    return Case(
        "cosine-validation",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=16,
        heads=2,
        layers=1,
        batch=8,
        learning_rate=0.003,
        weight_decay=0.1,
        warmup=2,
        max_updates=12,
        observe=2,
        after_target=0,
        target=1.0,
    )


def test_declared_cosine_endpoints_use_real_epochs_with_tail_batches() -> None:
    """Fifty warmup steps followed by 300/600 steps give 100/200 exact epochs."""
    case = replace(small_case(), learning_rate=0.01, warmup=50)
    config = Configuration()
    for updates, rate in [
        (0, 0),
        (25, 0.01),
        (50, 0.02),
        (350, 0.011),
        (650, 0.002),
        (950, 0.002),
    ]:
        assert config.learning_rate_at(case, updates, 21) == pytest.approx(rate)


def test_cosine_stays_bounded_and_decreases_after_warmup() -> None:
    """Check all step positions across warmup, decay and a held floor."""
    case = small_case()
    config = Configuration(decay_epochs=2)
    peak = case.learning_rate * config.learning_rate_multiplier
    rates = [config.learning_rate_at(case, completed, 21) for completed in range(100)]
    assert rates[0] == 0
    assert all(0 <= rate <= peak for rate in rates)
    assert all(peak / 10 <= rate for rate in rates[case.warmup :])
    assert all(
        a >= b
        for a, b in zip(rates[case.warmup :], rates[case.warmup + 1 :], strict=False)
    )
    assert rates[-1] == pytest.approx(peak / 10)


def test_journal_constructor_and_preparation_preserve_initial_model_rng(
    tmp_path: Path,
) -> None:
    """Creating execution accounting consumes no optimizer step, model value or RNG."""
    case, config = small_case(), Configuration()
    task = arithmetic(case, "cpu")
    torch.manual_seed(7)
    model = dimension_model(task, case, "cpu")
    values, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    path = tmp_path / "journal.json"
    stepper = FactoryStepper(
        model,
        task,
        case,
        "cpu",
        configuration=config,
        factories=configured_factories(config),
        journal_path=path,
        compiled=False,
    )
    stepper.prepare([case.batch])
    assert_close(model.state_dict(), values, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert stepper.optimizer.state_dict()["state"] == {}
    assert not path.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_width", [8, 32])
def test_cuda_journal_preserves_native_graph_parameters_moments_and_tail_batches(
    tmp_path: Path,
    head_width: int,
) -> None:
    """Compare actual original and journaled graphs through warmup, decay and floor."""
    torch.compiler.reset()
    torch.set_float32_matmul_precision("highest")
    case = replace(small_case(), width=2 * head_width, layers=2)
    config = Configuration(decay_epochs=2)
    task = arithmetic(case, "cuda")
    torch.manual_seed(11)
    model = dimension_model(task, case, "cuda")
    reference = copy.deepcopy(model)
    actual = FactoryStepper(
        model,
        task,
        case,
        "cuda",
        configuration=config,
        factories=configured_factories(config),
        journal_path=tmp_path / "execution-journal.json",
        compiled=True,
    )
    parent = ParentStepper(
        reference,
        task,
        case,
        "cuda",
        configuration=config,
        factories=configured_factories(config),
        compiled=True,
    )
    sampler = EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=10011
    )
    initial, rng, cuda_rng = (
        copy.deepcopy(model.state_dict()),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
    )
    actual.prepare(sampler.sizes())
    assert_close(model.state_dict(), initial, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert not actual.execution_journal.path.exists()
    assert all(float(state["step"]) == 0 for state in actual.optimizer.state.values())
    parent.prepare(sampler.sizes())
    sizes = []
    for completed in range(case.max_updates):
        parts, _ = sampler.next()
        indices = gather(parts)
        sizes.append(len(indices))
        actual.step(indices, completed)
        parent.step(indices, completed)
        assert_close(model.state_dict(), reference.state_dict(), rtol=5e-4, atol=4e-6)
        assert_close(
            actual.optimizer.state_dict(),
            parent.optimizer.state_dict(),
            rtol=5e-4,
            atol=4e-6,
        )
        summary = actual.execution_journal.summary(completed + 1)
        assert summary["actual_examples_lower"] == sum(sizes)
        assert summary["actual_examples_upper"] == sum(sizes)
    assert min(sizes) < case.batch
    measured = actual.evaluate(task.selection)
    expected = parent.evaluate(task.selection)
    assert measured["sequence_accuracy"] == expected["sequence_accuracy"]
    assert measured["loss"] == pytest.approx(expected["loss"], rel=1e-4, abs=3e-6)

    # Exercise the actual nonempty-moment CUDA restore path, preserving the
    # original journal until the first genuine continued update.
    continued_model = copy.deepcopy(model)
    continued = FactoryStepper(
        continued_model,
        task,
        case,
        "cuda",
        configuration=config,
        factories=configured_factories(config),
        journal_path=actual.execution_journal.path,
        compiled=True,
    )
    continued.optimizer.load_state_dict(actual.optimizer.state_dict())
    for group in continued.optimizer.param_groups:
        group["lr"] = continued.rate
    values, moments = (
        copy.deepcopy(continued_model.state_dict()),
        copy.deepcopy(continued.optimizer.state_dict()),
    )
    raw_journal = continued.execution_journal.path.read_bytes()
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    continued.prepare(sampler.sizes())
    assert_close(continued_model.state_dict(), values, rtol=0, atol=0)
    assert_close(continued.optimizer.state_dict(), moments, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert continued.execution_journal.path.read_bytes() == raw_journal
    parts, _ = sampler.next()
    indices = gather(parts)
    continued.step(indices, case.max_updates)
    parent.step(indices, case.max_updates)
    assert_close(
        continued_model.state_dict(), reference.state_dict(), rtol=5e-4, atol=4e-6
    )
    assert_close(
        continued.optimizer.state_dict(),
        parent.optimizer.state_dict(),
        rtol=5e-4,
        atol=4e-6,
    )
    summary = continued.execution_journal.summary(case.max_updates + 1)
    assert (
        summary["actual_examples_lower"]
        == summary["actual_examples_upper"]
        == sum(sizes) + len(indices)
    )
    verify_saved_journal(
        tmp_path,
        case.max_updates + 1,
        len(task.splits["train"]),
        case.batch,
        [{"completed": case.max_updates}],
        summary,
    )


def test_trainer_resume_restores_native_state_and_journal_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare a real uninterrupted trajectory with checkpointed continuation."""
    case, config = small_case(), Configuration(decay_epochs=2)
    monkeypatch.setattr(trainer, "CHECKPOINT_INTERVAL", 2)
    full = trainer.run(
        case, 4, tmp_path / "full", "cpu", configuration=config, compiled=False
    )
    save = trainer.save_checkpoint

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
            message = "Interrupted after checkpoint"
            raise RuntimeError(message)

    monkeypatch.setattr(trainer, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="Interrupted after checkpoint"):
        trainer.run(
            case, 4, tmp_path / "resume", "cpu", configuration=config, compiled=False
        )
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    resumed = trainer.run(
        case,
        4,
        tmp_path / "resume",
        "cpu",
        configuration=config,
        compiled=False,
        resume=True,
    )
    first = torch.load(
        tmp_path / "full" / case.name / "seed4/checkpoint.pt", weights_only=True
    )
    second = torch.load(
        tmp_path / "resume" / case.name / "seed4/checkpoint.pt", weights_only=True
    )
    for key in ["model", "optimizer", "sampler", "cpu_rng"]:
        assert_close(first[key], second[key], rtol=0, atol=0)
    assert resumed["endpoint"]["validation"] == full["endpoint"]["validation"]
    assert (
        resumed["execution_journal"]["actual_examples_lower"]
        == full["execution_journal"]["actual_examples_lower"]
    )
    assert (
        resumed["execution_journal"]["actual_examples_upper"]
        == full["execution_journal"]["actual_examples_upper"]
    )
    assert resumed["execution_journal"]["resumed_segments"] == 1


def test_trainer_preserves_entire_original_protocol_outside_execution_accounting() -> (
    None
):
    """Restrict changes to construction, journal metadata and its final record."""
    source = inspect.getsource(parent_run)
    edits = [
        (
            '        "warmup_uses_completed_updates": True,',
            (
                '        "warmup_uses_completed_updates": True,\n'
                '        "execution_journal": "durable per genuine update; '
                'CUDA synchronized after step",'
            ),
        ),
        (
            "        compiled=compiled,\n    )",
            (
                "        compiled=compiled,\n"
                '        journal_path=directory / "execution-journal.json",\n    )'
            ),
        ),
        (
            '        "metadata": metadata,\n    }',
            (
                '        "metadata": metadata,\n'
                '        "execution_journal": '
                "stepper.execution_journal.summary(completed),\n    }"
            ),
        ),
    ]
    for before, after in edits:
        assert source.count(before) == 1
        source = source.replace(before, after)
    assert ast.dump(ast.parse(source)) == ast.dump(
        ast.parse(inspect.getsource(trainer.run))
    )
