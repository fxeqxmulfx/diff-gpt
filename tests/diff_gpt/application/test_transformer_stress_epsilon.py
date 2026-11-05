"""Check the single-change epsilon ablation and its actual radial derivative."""

import ast
import inspect
import json

import torch
from torch.testing import assert_close

from diff_gpt.domain.model.rms_norm import rms_norm
from research.transformer_stress.common import EPS, digest
from research.transformer_stress.residual import train_cell
from research.transformer_stress.run import PLAN
from research.transformer_stress_epsilon.learning import train_epsilon_cell
from research.transformer_stress_epsilon.models import SoftPeriNetwork
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork


def test_parent_equivalence_uses_active_nonzero_gates() -> None:
    """Compare outputs and gradients away from the trivial all-zero-branch start."""
    settings = json.loads(PLAN.read_text())["residual_learning"]
    settings["branch_epsilon"] = EPS
    torch.manual_seed(2)
    original = ShiftedResidualNetwork(settings, 4, "peri_rezero")
    torch.manual_seed(2)
    candidate = SoftPeriNetwork(settings, 4, "peri_rezero")
    assert digest(dict(original.named_parameters())) == digest(
        dict(candidate.named_parameters())
    )
    with torch.no_grad():
        original.gates.copy_(torch.linspace(0.1, 0.4, 4))
        candidate.gates.copy_(original.gates)
    x = torch.randn(8, settings["dimension"])
    before, after = original(x), candidate(x)
    assert_close(before, after, rtol=0, atol=0)
    before.square().sum().backward()
    after.square().sum().backward()
    for parent, changed in zip(
        original.parameters(), candidate.parameters(), strict=True
    ):
        assert_close(parent.grad, changed.grad, rtol=0, atol=0)
    candidate.branch_epsilon = 1
    assert not torch.equal(candidate(x), before)


def test_finite_epsilon_preserves_more_radial_signal() -> None:
    """Use the closed-form derivative of a scaled unit-RMS vector."""
    unit = torch.tensor([1.0, -1.0], dtype=torch.float64)
    derivatives = []
    for epsilon in (EPS, 0.001, 0.1, 1.0):
        scale = torch.ones((), dtype=torch.float64, requires_grad=True)
        output = (rms_norm(unit * scale, epsilon) * unit).mean()
        derivative = torch.autograd.grad(output, scale)[0]
        expected = torch.tensor(epsilon / (1 + epsilon) ** 1.5, dtype=torch.float64)
        assert_close(derivative, expected, rtol=1e-8, atol=0)
        derivatives.append(derivative.item())
    assert derivatives == sorted(derivatives)


def test_epsilon_ablation_preserves_training_protocol() -> None:
    """Keep the exact original optimizer, sampler, budget and scoring implementation."""
    original = inspect.getsource(train_cell)
    changed = (
        inspect.getsource(train_epsilon_cell)
        .replace("train_epsilon_cell", "train_cell", 1)
        .replace("SoftPeriNetwork", "ResidualNetwork", 1)
    )
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(changed))
