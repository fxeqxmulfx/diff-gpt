"""Candidate isolation, baseline equivalence and issued-update accounting."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
from torch.testing import assert_close

from research.current_grokking.data import arithmetic
from research.current_grokking.protocol import Case
from research.current_grokking.run import run as baseline_run
from research.current_grokking.train import Stepper
from research.early99.experiments import compare
from research.early99.experiments.factories import (
    Configuration,
    FactoryStepper,
    configured_factories,
)
from research.early99.experiments.run import prepare_output
from research.early99.experiments.trainer import run

if TYPE_CHECKING:
    from pathlib import Path

UPDATES = 8
SEEDS = 3
GRAPH_UPDATES = 6


def test_factory_loop_matches_original_weights_moments_and_scores(
    tmp_path: Path,
) -> None:
    """The extracted baseline factory must preserve an actual short-tail run."""
    case = Case(
        "factory-equivalence",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=16,
        heads=2,
        batch=8,
        observe=1,
        max_updates=UPDATES,
        after_target=0,
    )
    original = baseline_run(case, 1, tmp_path / "original", "cpu", compiled=False)
    candidate = run(case, 1, tmp_path / "candidate", "cpu", compiled=False)
    for key in ("completed", "first_target", "final_test", "optimizer_steps_verified"):
        assert candidate[key] == original[key]
    for key in ("train", "validation"):
        assert candidate["endpoint"][key] == original["endpoint"][key]
    checkpoints = [
        torch.load(
            tmp_path / directory / case.name / "seed1/checkpoint.pt", weights_only=True
        )
        for directory in ("original", "candidate")
    ]
    for key in ("model", "optimizer", "sampler", "cpu_rng"):
        assert_close(checkpoints[0][key], checkpoints[1][key], rtol=0, atol=0)
    replay = run(case, 1, tmp_path / "candidate", "cpu", compiled=False, resume=True)
    assert replay["endpoint"] == candidate["endpoint"]
    with pytest.raises(ValueError, match="Resume changes experiment"):
        run(
            case,
            1,
            tmp_path / "candidate",
            "cpu",
            compiled=False,
            resume=True,
            configuration=Configuration("adamw-lr2", 2),
        )


def test_candidate_plan_refuses_configuration_execution_and_source_changes(
    tmp_path: Path,
) -> None:
    """A run cannot silently replace its hyperparameters or code archive."""
    configuration = Configuration("adamw-lr2", 2)
    prepare_output(tmp_path, configuration, "cpu", compiled=False)
    prepare_output(tmp_path, configuration, "cpu", compiled=False)
    with pytest.raises(ValueError, match="settings/sources changed"):
        prepare_output(tmp_path, Configuration(), "cpu", compiled=False)
    with pytest.raises(ValueError, match="settings/sources changed"):
        prepare_output(tmp_path, configuration, "cuda", compiled=False)
    archive = tmp_path / "experiment_source/research/early99/experiments/factories.py"
    archive.write_text("changed")
    with pytest.raises(ValueError, match="Archived experiment source changed"):
        prepare_output(tmp_path, configuration, "cpu", compiled=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("multiplier", "beta2", "model_variant", "schedule"),
    [
        (1.0, 0.98, "baseline", "constant"),
        (2.0, 0.98, "baseline", "constant"),
        (2.5, 0.98, "baseline", "constant"),
        (3.0, 0.98, "baseline", "constant"),
        (2.0, 0.95, "baseline", "constant"),
        (3.0, 0.98, "baseline", "cosine"),
        (2.0, 0.98, "peri-rms", "constant"),
        (2.0, 0.98, "peri-rms-embedding", "constant"),
        (2.0, 0.98, "peri-rms-embedding-scaled", "constant"),
        (1.0, 0.98, "peri-rms-embedding-scaled-csa", "constant"),
        (2.0, 0.98, "peri-rms-embedding-scaled-csa", "constant"),
    ],
)
def test_candidate_capture_preserves_rng_and_matches_native_updates(
    multiplier: float,
    beta2: float,
    model_variant: str,
    schedule: str,
) -> None:
    """Both batch shapes and evaluation preserve state; replay counts real steps."""
    # Scientific cells use fresh processes. Do not share Dynamo's finite
    # recompilation budget between unrelated parameterized test variants.
    torch.compiler.reset()
    case = Case(
        "candidate-graph-check",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=32,
        batch=8,
        warmup=3,
    )
    task = arithmetic(case, "cuda")
    configuration = Configuration(
        "graph-check",
        multiplier,
        beta2=beta2,
        model_variant=model_variant,
        schedule=schedule,
        decay_epochs=0.5,
    )
    factories = configured_factories(configuration)
    torch.manual_seed(1)
    model = factories.model(task, case, "cuda")
    initial = copy.deepcopy(model.state_dict())
    stepper = FactoryStepper(
        model,
        task,
        case,
        "cuda",
        configuration=configuration,
        factories=factories,
        compiled=True,
    )
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    stepper.prepare([8, 5])
    assert_close(model.state_dict(), initial, rtol=0, atol=0)
    assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    assert all(int(state["step"]) == 0 for state in stepper.optimizer.state.values())
    torch.manual_seed(1)
    reference = factories.model(task, case, "cuda")
    eager = Stepper(
        reference,
        task,
        replace(case, learning_rate=case.learning_rate * multiplier),
        "cuda",
        compiled=False,
    )
    eager.optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=eager.rate,
        betas=(0.9, beta2),
        eps=1e-8,
        weight_decay=case.weight_decay,
        capturable=True,
        fused=True,
    )
    issued_sizes = [8, 8, 5, 8, 8, 5]
    for completed, size in enumerate(issued_sizes):
        before_eval = copy.deepcopy(stepper.optimizer.state_dict())
        stepper.evaluate("validation")
        assert_close(stepper.optimizer.state_dict(), before_eval, rtol=0, atol=0)
        stepper.step(torch.arange(size), completed)
        expected_rate = (
            case.learning_rate * multiplier * min(1.0, completed / case.warmup)
        )
        if schedule == "cosine" and completed > case.warmup:
            # Independent sample-count reference, including the short tail.
            after_warmup = sum(issued_sizes[case.warmup : completed])
            phase = min(after_warmup / len(task.splits["train"]) / 0.5, 1.0)
            expected_rate *= 0.5 + 0.25 * (1 + math.cos(math.pi * phase))
        eager.rate.fill_(expected_rate)
        reference.train()
        eager.update(torch.arange(size, device="cuda"))
    torch.cuda.synchronize()
    assert all(
        int(state["step"]) == GRAPH_UPDATES
        for state in stepper.optimizer.state.values()
    )
    assert_close(model.state_dict(), reference.state_dict(), rtol=2e-5, atol=2e-6)
    assert_close(
        stepper.optimizer.state_dict(),
        eager.optimizer.state_dict(),
        rtol=2e-5,
        atol=2e-6,
    )
    # Preparing a resumed optimizer must preserve already nonzero moments too.
    saved_model, saved_optimizer = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(stepper.optimizer.state_dict()),
    )
    saved_rng = torch.cuda.get_rng_state()
    stepper.prepare([8, 5])
    assert_close(model.state_dict(), saved_model, rtol=0, atol=0)
    assert_close(stepper.optimizer.state_dict(), saved_optimizer, rtol=0, atol=0)
    assert_close(torch.cuda.get_rng_state(), saved_rng, rtol=0, atol=0)


def test_comparison_keeps_censoring_and_suppresses_filtered_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two fast seeds cannot hide the third seed's budget miss."""
    monkeypatch.setattr(compare, "selected", lambda _: [{"name": "fixture"}])

    def cell(row: dict, seed: int, baseline: Path, candidate: Path) -> dict:
        del baseline, candidate
        return {
            "case": row["name"],
            "seed": seed,
            "baseline_status": "passed",
            "candidate_status": "budget_exhausted" if seed == SEEDS - 1 else "passed",
            "baseline_first99_epochs": 4,
            "candidate_first99_epochs": None if seed == SEEDS - 1 else 2,
            "epoch_ratio": None if seed == SEEDS - 1 else 0.5,
        }

    monkeypatch.setattr(compare, "paired_cell", cell)
    destination = tmp_path / "comparisons/incumbent"
    result = compare.comparison(tmp_path, tmp_path, "short", output=destination)
    assert result["expected_cells"] == SEEDS
    assert result["candidate_passed"] == SEEDS - 1
    assert result["geometric_mean_epoch_ratio"] is None
    assert result["tasks"][0]["candidate_median_first99_epochs"] is None
    assert len(result["failures_and_pending"]) == 1
    assert (destination / "comparison-short.json").is_file()
    assert not (tmp_path / "comparison-short.json").exists()
