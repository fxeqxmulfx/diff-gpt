"""Check the actual Schedule-Free benchmark adapter, scoring point and resume."""

from __future__ import annotations

import ast
import copy
import inspect
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
from torch.testing import assert_close

from research.current_grokking.data import arithmetic
from research.current_grokking.protocol import Case
from research.current_grokking.train import Stepper
from research.early99.experiments.trainer import run as original_run
from research.early99_dimension_gain.factories import dimension_model
from research.early99_schedule_free import trainer
from research.early99_schedule_free.factories import (
    Configuration,
    ScheduleFreeStepper,
    verify_optimizer_state,
)

if TYPE_CHECKING:
    from pathlib import Path

INTERRUPTED_UPDATE = 4


def small_case() -> Case:
    """Use the original arithmetic evaluator with a real short final batch."""
    return Case(
        "schedule-free-validation",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=16,
        heads=2,
        layers=1,
        batch=8,
        learning_rate=0.003,
        weight_decay=0.1,
        warmup=3,
        max_updates=6,
        observe=2,
        after_target=0,
        target=1.0,
    )


def test_configuration_preserves_model_and_rejects_double_warmup() -> None:
    """Keep original tensors/RNG and expose accurate optimizer metadata."""
    case = small_case()
    task = arithmetic(case, "cpu")
    config = Configuration()
    torch.manual_seed(5)
    model = dimension_model(task, case, "cpu")
    state, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    stepper = ScheduleFreeStepper(
        model, task, case, "cpu", configuration=config, compiled=False
    )
    assert_close(model.state_dict(), state, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.model.token_embedding_table.weight is model.model.lm_head.weight
    assert stepper.optimizer.param_groups[0]["warmup"] == 0
    assert config.metadata(case)["optimizer"].endswith("ScheduleFreeAdamW")
    assert config.metadata(case)["schedule_free"]["checkpoint_point"] == "x"
    assert config.metadata(case)["dimension_gain"]["fixed_gain"] == 1 / 8
    with pytest.raises(ValueError, match="external warmup"):
        Configuration(optimizer_internal_warmup=50)
    with pytest.raises(ValueError, match="positive beta1"):
        Configuration(beta1=0)


def test_no_hidden_updates_and_zero_rate_first_update() -> None:
    """Prepare must preserve all state; the first real batch updates only v/count."""
    case, config = small_case(), Configuration()
    task = arithmetic(case, "cpu")
    torch.manual_seed(11)
    model = dimension_model(task, case, "cpu")
    stepper = ScheduleFreeStepper(
        model, task, case, "cpu", configuration=config, compiled=False
    )
    state, moments = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(stepper.optimizer.state_dict()),
    )
    rng = torch.get_rng_state().clone()
    stepper.prepare([8, 5])
    assert_close(model.state_dict(), state, rtol=0, atol=0)
    assert_close(stepper.optimizer.state_dict(), moments, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    stepper.step(torch.arange(8), 0)
    group = stepper.optimizer.param_groups[0]
    assert group["step"] == 1
    assert group["lr"] == group["maximum"] == group["total"] == 0
    assert_close(model.state_dict(), state, rtol=0, atol=0)
    assert any(
        torch.count_nonzero(s["second"]) for s in stepper.optimizer.state.values()
    )
    for parameter, moment in stepper.optimizer.state.items():
        assert_close(moment["base"], parameter, rtol=0, atol=0)
        assert_close(
            moment["second"],
            (1 - config.beta2) * parameter.grad.square(),
            rtol=3e-7,
            atol=1e-12,
        )
    for completed, size in enumerate([5, 8, 5, 8], start=1):
        stepper.step(torch.arange(size), completed)
        assert group["lr"] == config.learning_rate_at(
            case, completed, len(task.splits["train"])
        )
    stepper.evaluate(task.selection)
    verify_optimizer_state(stepper, 5)
    with pytest.raises(ValueError, match="positive sizes"):
        stepper.prepare([0])


def test_full_split_evaluation_reads_x_and_repeated_switch_is_idempotent() -> None:
    """Compare full held-out logits/metrics to an explicit independently recovered x."""
    case = replace(small_case(), learning_rate=0.03)
    config, task = Configuration(), arithmetic(case, "cpu")
    torch.manual_seed(6)
    model = dimension_model(task, case, "cpu")
    stepper = ScheduleFreeStepper(
        model, task, case, "cpu", configuration=config, compiled=False
    )
    for completed in range(5):
        stepper.step(torch.arange(8), completed)
    y = [p.detach().clone() for p in model.parameters()]
    ref = copy.deepcopy(model)
    with torch.no_grad():
        for p, point, state in zip(
            ref.parameters(), y, stepper.optimizer.state.values(), strict=True
        ):
            p.copy_((point - (1 - config.beta1) * state["base"]) / config.beta1)
    evaluator = Stepper(ref, task, case, "cpu", compiled=False)
    expected = evaluator.evaluate(task.selection)
    actual = stepper.evaluate(task.selection)
    assert_close(actual["loss"], expected["loss"], rtol=2e-6, atol=1e-7)
    assert actual["sequence_accuracy"] == expected["sequence_accuracy"]
    assert any(
        not torch.equal(p, point)
        for p, point in zip(model.parameters(), y, strict=True)
    )
    x = copy.deepcopy(model.state_dict())
    state = copy.deepcopy(stepper.optimizer.state_dict())
    assert stepper.evaluate(task.selection) == actual
    assert_close(model.state_dict(), x, rtol=0, atol=0)
    assert_close(stepper.optimizer.state_dict(), state, rtol=0, atol=0)
    verify_optimizer_state(stepper, 5)


def test_complete_trainer_checkpoint_resume_preserves_x_state_and_batch_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupt after a real checkpoint; compare continuation to uninterrupted work."""
    case, config = small_case(), Configuration()
    monkeypatch.setattr(trainer, "CHECKPOINT_INTERVAL", 2)
    full = trainer.run(
        case, 4, tmp_path / "full", "cpu", configuration=config, compiled=False
    )
    assert full["completed"] == case.max_updates
    save = trainer.save_checkpoint

    class Interrupted(RuntimeError):
        pass

    def stop_after_checkpoint(
        path: Path,
        model: object,
        stepper: object,
        sampler: object,
        completed: int,
        history: list,
        metadata: dict,
    ) -> None:
        save(path, model, stepper, sampler, completed, history, metadata)
        if completed == INTERRUPTED_UPDATE:
            raise Interrupted

    monkeypatch.setattr(trainer, "save_checkpoint", stop_after_checkpoint)
    with pytest.raises(Interrupted):
        trainer.run(
            case, 4, tmp_path / "resume", "cpu", configuration=config, compiled=False
        )
    checkpoint = tmp_path / "resume" / case.name / "seed4/checkpoint.pt"
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["completed"] == INTERRUPTED_UPDATE
    assert saved["optimizer"]["param_groups"][0]["training"] is False
    assert saved["optimizer"]["param_groups"][0]["step"] == INTERRUPTED_UPDATE
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
    second = torch.load(checkpoint, weights_only=True)
    for name in ("model", "optimizer", "sampler", "cpu_rng"):
        assert_close(first[name], second[name], rtol=0, atol=0)
    assert resumed["completed"] == full["completed"]
    assert resumed["endpoint"]["validation"] == full["endpoint"]["validation"]
    assert resumed["metadata"]["optimizer_checkpoint_point"] == "x"
    assert resumed["metadata"]["resumptions"][0]["completed"] == INTERRUPTED_UPDATE


def test_trainer_preserves_original_protocol_outside_declared_optimizer_boundary() -> (
    None
):
    """Compare the whole loop after enumerating only the required optimizer edits."""
    source = inspect.getsource(original_run)
    edits = [
        ("    factories: Factories | None = None,\n", ""),
        (
            (
                "    if factories is None:\n"
                "        factories = configured_factories(configuration)\n"
            ),
            "",
        ),
        (
            "model = factories.model(task, case, device)",
            "model = dimension_model(task, case, device)",
        ),
        (
            '"optimizer": "torch.optim.AdamW",',
            '"optimizer": "diff_gpt.domain.optimizer.schedule_free.ScheduleFreeAdamW",',
        ),
        (
            '"optimizer_fused": device == "cuda",',
            (
                '"optimizer_fused": False,\n'
                '        "optimizer_cuda_graphs": False,\n'
                '        "optimizer_training_point": "y",\n'
                '        "optimizer_evaluation_point": "x",\n'
                '        "optimizer_checkpoint_point": "x",'
            ),
        ),
        ("stepper = FactoryStepper(", "stepper = ScheduleFreeStepper("),
        ("        factories=factories,\n", ""),
        (
            (
                "        for group in stepper.optimizer.param_groups:\n"
                '            group["lr"] = stepper.rate\n'
            ),
            "",
        ),
        (
            (
                '    steps = [int(state["step"]) for state in '
                "stepper.optimizer.state.values()]\n"
                "    if not steps or any(step != completed for step in steps):\n"
                '        msg = f"Optimizer steps disagree with completed updates: '
                '{steps}, {completed}"\n'
                "        raise AssertionError(msg)\n"
            ),
            "    verify_optimizer_state(stepper, completed)\n",
        ),
    ]
    for before, after in edits:
        assert source.count(before) == 1
        source = source.replace(before, after)
    assert ast.dump(ast.parse(source)) == ast.dump(
        ast.parse(inspect.getsource(trainer.run))
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_width", [8, 32])
def test_cuda_compiled_and_eager_schedule_free_state_and_scoring(
    head_width: int,
) -> None:
    """Check actual GPU updates, both weight points, every moment and short batches."""
    torch.compiler.reset()
    case = replace(small_case(), width=2 * head_width, layers=2)
    config, task = Configuration(), arithmetic(case, "cuda")
    torch.manual_seed(7)
    model = dimension_model(task, case, "cuda")
    reference = copy.deepcopy(model)
    actual = ScheduleFreeStepper(
        model, task, case, "cuda", configuration=config, compiled=True
    )
    eager = ScheduleFreeStepper(
        reference, task, case, "cuda", configuration=config, compiled=False
    )
    state, moments = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(actual.optimizer.state_dict()),
    )
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    actual.prepare([8, 5])
    assert_close(model.state_dict(), state, rtol=0, atol=0)
    assert_close(actual.optimizer.state_dict(), moments, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    for completed, size in enumerate([8, 5, 8, 5, 8]):
        for stepper in (actual, eager):
            stepper.step(torch.arange(size), completed)
        assert_close(model.state_dict(), reference.state_dict(), rtol=5e-4, atol=4e-6)
        measured = actual.evaluate(task.selection)
        expected = eager.evaluate(task.selection)
        assert_close(measured["loss"], expected["loss"], rtol=1e-4, atol=3e-6)
        assert measured["sequence_accuracy"] == expected["sequence_accuracy"]
        assert_close(model.state_dict(), reference.state_dict(), rtol=5e-4, atol=4e-6)
        for a, b in zip(
            actual.optimizer.state.values(), eager.optimizer.state.values(), strict=True
        ):
            assert_close(a["base"], b["base"], rtol=5e-4, atol=4e-6)
            assert_close(a["second"], b["second"], rtol=5e-4, atol=3e-7)
        for stepper in (actual, eager):
            verify_optimizer_state(stepper, completed + 1)
