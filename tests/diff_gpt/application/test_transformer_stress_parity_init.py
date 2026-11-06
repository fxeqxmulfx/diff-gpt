"""Verify that only scalar initialization changes in the paired follow-up."""

import ast
import inspect
import json

import torch
from torch.testing import assert_close

from research.transformer_stress.common import digest
from research.transformer_stress.residual import train_cell
from research.transformer_stress.run import PLAN as ORIGINAL_PLAN
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork
from research.transformer_stress_parity_init.learning import train_initialized_cell
from research.transformer_stress_parity_init.models import InitializedNetwork
from research.transformer_stress_parity_init.run import PLAN, summarize


def test_identical_matrices_rng_and_capacity_across_initial_values() -> None:
    """Changing the even coefficient must not redraw or resize any matrix."""
    settings = json.loads(ORIGINAL_PLAN.read_text())["residual_learning"]
    settings["activation_variant"] = "learned_even"
    fingerprints, capacities, rngs = [], [], []
    for initial in (0.0, 0.5, 1.0):
        torch.manual_seed(2)
        model = InitializedNetwork(
            dict(settings, even_gain_initial=initial), 4, "peri_rezero"
        )
        fingerprints.append(
            digest(
                {
                    name: value
                    for name, value in model.named_parameters()
                    if not name.endswith("even_gain")
                }
            )
        )
        capacities.append(sum(p.numel() for p in model.parameters()))
        rngs.append(torch.get_rng_state())
        assert all(branch.even_gain.item() == initial for branch in model.branches)
    assert len(set(fingerprints)) == len(set(capacities)) == 1
    assert all(torch.equal(rng, rngs[0]) for rng in rngs)


def test_initial_one_preserves_parent_function_with_active_residuals() -> None:
    """Test the advertised starting function beyond the trivial identity skip."""
    settings = json.loads(ORIGINAL_PLAN.read_text())["residual_learning"]
    settings.update(activation_variant="learned_even", even_gain_initial=1.0)
    torch.manual_seed(2)
    parent = ShiftedResidualNetwork(settings, 4, "peri_rezero").double()
    torch.manual_seed(2)
    candidate = InitializedNetwork(settings, 4, "peri_rezero").double()
    with torch.no_grad():
        parent.gates.copy_(torch.linspace(0.1, 0.4, 4))
        candidate.gates.copy_(parent.gates)
    x = torch.randn(8, settings["dimension"], dtype=torch.float64)
    before, after = parent(x), candidate(x)
    assert_close(before, after, rtol=1e-11, atol=1e-12)
    before.square().sum().backward()
    after.square().sum().backward()
    changed = dict(candidate.named_parameters())
    for name, parameter in parent.named_parameters():
        assert_close(parameter.grad, changed[name].grad, rtol=1e-10, atol=1e-11)


def test_unchanged_training_protocol() -> None:
    """Retain the original sampler, optimizer, budget and complete scoring code."""
    original = inspect.getsource(train_cell)
    changed = (
        inspect.getsource(train_initialized_cell)
        .replace("train_initialized_cell", "train_cell", 1)
        .replace("InitializedNetwork", "ResidualNetwork", 1)
    )
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(changed))


def test_no_partial_selection() -> None:
    """Both tasks and both new initial values must finish before ranking."""
    result = summarize(
        [],
        json.loads(PLAN.read_text()),
        json.loads(ORIGINAL_PLAN.read_text())["residual_learning"],
    )
    assert not result["complete"]
    assert result["best_initialization_arm"] is None
