"""Independent matrix references, partitions, actual CUDA updates and restoration."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from research.current_grokking.data import arithmetic, build
from research.current_grokking.protocol import Case
from research.current_grokking.run import EpochSampler, gather
from research.early99.protocol import selected, specification
from research.early99_cosine.trainer import run as journaled_run
from research.early99_dimension_gain.factories import dimension_model
from research.early99_execution.journal import verify_saved_journal
from research.early99_muon import trainer
from research.early99_muon.factories import (
    Configuration,
    MuonStepper,
    verify_optimizer_state,
)
from research.early99_muon.optimizer import MatrixMuon, MuonBundle, matrix_direction

INTERRUPT_AT = 4
RELATIVE_DIRECTION_TOLERANCE = 5e-5
NUMERIC_SPECTRAL_LIMIT = 1.25001
EXTREME_GRADIENT_STEP = 3


def assert_nested_close(
    actual: object, expected: object, *, rtol: float, atol: float
) -> None:
    """Compare all tensor/scalar state and require exact nonnumeric metadata."""
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_close(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_nested_close(a, b, rtol=rtol, atol=atol)
    elif isinstance(actual, str) or actual is None:
        assert actual == expected
    else:
        assert_close(actual, expected, rtol=rtol, atol=atol)


def small_case() -> Case:
    """Use the actual arithmetic scorer, tail batches and completed-step warmup."""
    return Case(
        "muon-validation",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=16,
        heads=2,
        layers=1,
        batch=8,
        learning_rate=0.003,
        weight_decay=0.1,
        warmup=3,
        max_updates=8,
        observe=2,
        after_target=0,
        target=1.0,
    )


def svd_reference(matrix: torch.Tensor) -> torch.Tensor:
    """Evaluate singular-value polynomials, independently of matrix NS products."""
    wide = matrix.double().cpu()
    u, singular, vh = torch.linalg.svd(wide, full_matrices=False)
    singular = singular / torch.linalg.vector_norm(wide).clamp_min(1e-7)
    for _ in range(5):
        singular = 3.4445 * singular - 4.775 * singular**3 + 2.0315 * singular**5
    return (u * singular.unsqueeze(0)) @ vh


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (4, 4), (1, 1)])
@pytest.mark.parametrize("scale", [0.0, 1e-30, 1e-8, 1.0, 1e20, 1e30])
def test_matrix_direction_matches_independent_svd_across_magnitudes(
    shape: tuple[int, int], scale: float
) -> None:
    """Check square, tall, wide and scalar matrices without overflow/underflow loss."""
    torch.manual_seed(21)
    gradient = torch.randn(shape) * scale
    actual = matrix_direction(gradient).double()
    expected = svd_reference(gradient)
    assert torch.isfinite(actual).all()
    if scale == 0:
        assert torch.count_nonzero(actual) == 0
    else:
        relative = torch.linalg.vector_norm(
            actual - expected
        ) / torch.linalg.vector_norm(expected)
        assert relative < RELATIVE_DIRECTION_TOLERANCE
        assert torch.count_nonzero(actual) > 0
    assert torch.linalg.svdvals(actual).max() <= NUMERIC_SPECTRAL_LIMIT


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (4, 4)])
def test_rank_deficient_direction_and_nonpolar_identity(shape: tuple[int, int]) -> None:
    """Zero singular subspaces stay zero within tolerance; finite NS is not polar."""
    torch.manual_seed(9)
    gradient = torch.randn(shape[0], 1) @ torch.randn(1, shape[1])
    actual = matrix_direction(gradient).double()
    assert_nested_close(actual, svd_reference(gradient), rtol=3e-5, atol=3e-6)
    scalar = matrix_direction(torch.ones(1, 1))
    assert not torch.allclose(scalar, torch.ones_like(scalar), rtol=1e-3, atol=1e-3)


def test_rejects_unsupported_parameters_and_nonfinite_gradients() -> None:
    """Never silently zero an invalid direction or duplicate a tied update."""
    parameter = torch.nn.Parameter(torch.ones(3, 4))
    with pytest.raises(ValueError, match="unique"):
        MatrixMuon([("one", parameter), ("two", parameter)], lr=0.1, weight_decay=0)
    with pytest.raises(ValueError, match="nonmatrix"):
        MatrixMuon(
            [("vector", torch.nn.Parameter(torch.ones(3)))], lr=0.1, weight_decay=0
        )
    for bad in [float("inf"), float("nan")]:
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            matrix_direction(torch.full((3, 4), bad))
    optimizer = MatrixMuon([("matrix", parameter)], lr=0.1, weight_decay=0)
    parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError, match="finite dense"):
        optimizer.step()
    assert optimizer.state_dict()["state"] == {}


class TinyTiedModel(torch.nn.Module):
    """A small rectangular hidden matrix and one aliased embedding/readout."""

    def __init__(self) -> None:
        """Create the two unique matrices used by independent update checks."""
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(5, 3)
        self.lm_head = torch.nn.Linear(3, 5, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.hidden = torch.nn.Linear(3, 4, bias=False)


@pytest.mark.parametrize("kind", ["muon_adamw", "muon"])
def test_complete_matrix_momentum_decay_and_auxiliary_trajectory(kind: str) -> None:
    """Compare every issued update to FP64 SVD math and separate native AdamW."""
    torch.manual_seed(10)
    model = TinyTiedModel()
    actual = MuonBundle(
        model, kind, matrix_lr=0, auxiliary_lr=0, weight_decay=0.2, device="cpu"
    )
    references = {n: p.detach().double().clone() for n, p in model.named_parameters()}
    buffers = {n: torch.zeros_like(p) for n, p in references.items()}
    auxiliary = torch.nn.Parameter(model.token_embedding_table.weight.detach().clone())
    adamw = torch.optim.AdamW(
        [auxiliary], lr=0, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.2, foreach=False
    )
    for step, rate in enumerate([0, 0.002, 0.004, 0.006, 0.006, 0.006]):
        actual.muon.param_groups[0]["lr"] = rate
        if actual.adamw is not None:
            actual.adamw.param_groups[0]["lr"] = rate / 2
        adamw.param_groups[0]["lr"] = rate / 2
        for name, parameter in model.named_parameters():
            scale = 1e20 if step == EXTREME_GRADIENT_STEP else 1
            parameter.grad = torch.randn_like(parameter) * scale
            if kind == "muon_adamw" and name == "token_embedding_table.weight":
                # Keep auxiliary AdamW inside its ordinary finite-moment range.
                parameter.grad = parameter.grad.clamp(-3, 3)
                auxiliary.grad = parameter.grad.clone()
            else:
                gradient = parameter.grad.double()
                buffers[name] = 0.95 * buffers[name] + 0.05 * gradient
                h = 0.05 * gradient + 0.95 * buffers[name]
                direction = svd_reference(h)
                references[name] = (1 - rate * 0.2) * references[name] - (
                    rate * 0.2 * math.sqrt(max(parameter.shape)) * direction
                )
        actual.step()
        if kind == "muon_adamw":
            adamw.step()
        for name, parameter in model.named_parameters():
            if kind == "muon_adamw" and name == "token_embedding_table.weight":
                assert_nested_close(parameter, auxiliary, rtol=0, atol=0)
            else:
                assert_nested_close(
                    parameter.double(), references[name], rtol=2e-5, atol=2e-6
                )
                state = actual.muon.state[parameter]
                assert int(state["step"]) == step + 1
                assert_nested_close(
                    state["momentum_buffer"].double(),
                    buffers[name],
                    rtol=2e-6,
                    atol=2e-7,
                )
    assert (actual.adamw is None) == (kind == "muon")
    changed = copy.deepcopy(actual.state_dict())
    changed["partition"]["kind"] = "wrong"
    with pytest.raises(ValueError, match="different optimizer partition"):
        actual.load_state_dict(changed)


def test_all_35_parameter_partitions_match_frozen_inventory() -> None:
    """Check model identity, tied deduplication and every matrix in both variants."""
    inventory = json.loads(
        Path(
            "research/early99_optimizer_followups/parameter-inventory.json"
        ).read_text()
    )
    by_name = {r["case"]: r for r in inventory["rows"]}
    for row in selected("all"):
        case = specification(row)
        task = build(case, "cpu")
        torch.manual_seed(7)
        model = dimension_model(task, case, "cpu")
        original, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
        expected = by_name[case.name]
        for kind in ["muon_adamw", "muon"]:
            bundle = MuonBundle(
                model.model,
                kind,
                matrix_lr=0.01,
                auxiliary_lr=0.02,
                weight_decay=case.weight_decay,
                device="cpu",
            )
            matrix_names = [p["name"] for p in bundle.partition["muon"]]
            auxiliary_names = [p["name"] for p in bundle.partition["adamw"]]
            assert matrix_names == (
                [p["name"] for p in expected["hybrid_muon_matrices"]]
                if kind == "muon_adamw"
                else expected["pure_muon_matrices"]
            )
            assert auxiliary_names == (
                ["token_embedding_table.weight"] if kind == "muon_adamw" else []
            )
            assert (
                sum(p.numel() for g in bundle.param_groups for p in g["params"])
                == (expected["parameters"])
            )
            assert_nested_close(model.state_dict(), original, rtol=0, atol=0)
            assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kind", ["muon_adamw", "muon"])
def test_preparation_and_zero_rate_step_preserve_weights_and_rng(
    tmp_path: Path, kind: str
) -> None:
    """Preparation issues no update; zero-rate first batch advances moments/counts."""
    case, config = small_case(), Configuration(optimizer_kind=kind)
    task = arithmetic(case, "cpu")
    model = dimension_model(task, case, "cpu")
    state, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    actual = MuonStepper(
        model,
        task,
        case,
        "cpu",
        configuration=config,
        compiled=False,
        journal_path=tmp_path / "execution-journal.json",
    )
    moments = copy.deepcopy(actual.optimizer.state_dict())
    actual.prepare([8, 5])
    assert_nested_close(actual.optimizer.state_dict(), moments, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not actual.execution_journal.path.exists()
    actual.step(torch.arange(8), 0)
    assert_nested_close(model.state_dict(), state, rtol=0, atol=0)
    verify_optimizer_state(actual, 1)
    assert any(
        torch.count_nonzero(s["momentum_buffer"])
        for s in actual.optimizer.muon.state.values()
    )
    with pytest.raises(ValueError, match="positive sizes"):
        actual.prepare([0])


@pytest.mark.parametrize("kind", ["muon_adamw", "muon"])
def test_complete_trainer_resume_preserves_moments_batches_and_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Compare a real checkpoint continuation with uninterrupted original batches."""
    case, config = small_case(), Configuration(optimizer_kind=kind)
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
        assert_nested_close(first[key], second[key], rtol=0, atol=0)
    assert resumed["endpoint"]["validation"] == full["endpoint"]["validation"]
    for key in ["actual_examples_lower", "actual_examples_upper"]:
        assert resumed["execution_journal"][key] == full["execution_journal"][key]
    assert resumed["execution_journal"]["resumed_segments"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("head_width", "kind"), [(8, "muon_adamw"), (32, "muon")])
def test_cuda_compiled_updates_svd_reference_and_nonempty_restore(
    tmp_path: Path, head_width: int, kind: str
) -> None:
    """Exercise actual model gradients, tail batches, native auxiliary and resume."""
    torch.compiler.reset()
    torch.set_float32_matmul_precision("highest")
    case = replace(small_case(), width=2 * head_width, layers=2)
    config, task = Configuration(optimizer_kind=kind), arithmetic(case, "cuda")
    torch.manual_seed(15)
    model = dimension_model(task, case, "cuda")
    reference = copy.deepcopy(model)
    actual = MuonStepper(
        model,
        task,
        case,
        "cuda",
        configuration=config,
        compiled=True,
        journal_path=tmp_path / "execution-journal.json",
    )
    eager = MuonStepper(
        reference,
        task,
        case,
        "cuda",
        configuration=config,
        compiled=False,
        journal_path=tmp_path / "reference/execution-journal.json",
    )
    sampler = EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=10015
    )
    initial, moments = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(actual.optimizer.state_dict()),
    )
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    actual.prepare(sampler.sizes())
    assert_nested_close(model.state_dict(), initial, rtol=0, atol=0)
    assert_nested_close(actual.optimizer.state_dict(), moments, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    sizes = []
    for completed in range(6):
        parts, _ = sampler.next()
        indices = gather(parts)
        sizes.append(len(indices))
        actual.step(indices, completed)
        eager.step(indices, completed)
        assert_nested_close(
            model.state_dict(), reference.state_dict(), rtol=8e-4, atol=5e-6
        )
        assert_nested_close(
            actual.optimizer.state_dict(),
            eager.optimizer.state_dict(),
            rtol=8e-4,
            atol=5e-6,
        )
        verify_optimizer_state(actual, completed + 1)
    assert min(sizes) < case.batch
    assert (
        actual.evaluate(task.selection)["sequence_accuracy"]
        == eager.evaluate(task.selection)["sequence_accuracy"]
    )
    # The independent reference operates on real CUDA model momentum, not just toy data.
    for state in actual.optimizer.muon.state.values():
        direction = matrix_direction(state["momentum_buffer"])
        assert_nested_close(
            direction.cpu().double(),
            svd_reference(state["momentum_buffer"]),
            rtol=3e-4,
            atol=5e-6,
        )

    restored_model = copy.deepcopy(model)
    restored = MuonStepper(
        restored_model,
        task,
        case,
        "cuda",
        configuration=config,
        compiled=True,
        journal_path=actual.execution_journal.path,
    )
    restored.optimizer.load_state_dict(copy.deepcopy(actual.optimizer.state_dict()))
    verify_optimizer_state(restored, 6)
    raw = restored.execution_journal.path.read_bytes()
    restored.prepare(sampler.sizes())
    assert restored.execution_journal.path.read_bytes() == raw
    parts, _ = sampler.next()
    indices = gather(parts)
    restored.step(indices, 6)
    eager.step(indices, 6)
    assert_nested_close(
        restored_model.state_dict(), reference.state_dict(), rtol=8e-4, atol=5e-6
    )
    assert_nested_close(
        restored.optimizer.state_dict(),
        eager.optimizer.state_dict(),
        rtol=8e-4,
        atol=5e-6,
    )
    summary = restored.execution_journal.summary(7)
    assert (
        summary["actual_examples_lower"]
        == summary["actual_examples_upper"]
        == sum(sizes) + len(indices)
    )
    verify_saved_journal(
        tmp_path, 7, len(task.splits["train"]), case.batch, [{"completed": 6}], summary
    )


def test_entire_trainer_protocol_changes_only_at_declared_optimizer_boundary() -> None:
    """Compare full AST, including observations, sampler, stopping and checkpoints."""
    source = inspect.getsource(journaled_run)
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
            (
                '"optimizer": "torch.optim.AdamW",\n'
                '        "optimizer_betas": list(configuration.betas),\n'
                '        "optimizer_eps": 1e-8,\n'
                '        "optimizer_fused": device == "cuda",'
            ),
            (
                '"optimizer": "research.early99_muon.optimizer.MuonBundle",\n'
                '        "optimizer_cuda_graphs": False,\n'
                '        "matrix_dtype": "float32",\n'
                '        "matrix_normalization_dtype": "float64",\n'
                '        "auxiliary_native_fused": '
                'configuration.optimizer_kind == "muon_adamw"\n'
                '        and device == "cuda",'
            ),
        ),
        ("stepper = FactoryStepper(", "stepper = MuonStepper("),
        ("        factories=factories,\n", ""),
        (
            "    sampler = EpochSampler(",
            (
                '    metadata["optimizer_partition"] = stepper.optimizer.partition\n'
                "    sampler = EpochSampler("
            ),
        ),
        (
            (
                "        for group in stepper.optimizer.param_groups:\n"
                '            group["lr"] = stepper.rate\n'
            ),
            "",
        ),
        (
            '        completed, history = saved["completed"], saved["history"]',
            (
                '        completed, history = saved["completed"], saved["history"]\n'
                "        verify_optimizer_state(stepper, completed)"
            ),
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


def test_declared_rates_and_rejection_of_other_algorithms() -> None:
    """LR4 changes matrix rate only; metadata matches the archived declarations."""
    case = replace(small_case(), warmup=50)
    plan = json.loads(Path("research/early99_muon/plan.json").read_text())
    for declared in plan["configurations"]:
        config = Configuration(**declared)
        assert config.metadata(case)["configuration"] == declared
        for completed in [0, 25, 50, 1000]:
            expected = min(1.0, completed / 50) * case.learning_rate
            assert config.learning_rate_at(case, completed, 21) == (
                config.learning_rate_multiplier * expected
            )
            assert config.auxiliary_rate_at(case, completed) == 2 * expected
    for changed in [
        {"muon_ns_steps": 6},
        {"muon_compute_dtype": "bfloat16"},
        {"optimizer_kind": "unknown"},
        {"beta2": 0.999},
        {"schedule": "cosine"},
    ]:
        with pytest.raises(ValueError, match="declared Muon rule"):
            Configuration(**changed)
