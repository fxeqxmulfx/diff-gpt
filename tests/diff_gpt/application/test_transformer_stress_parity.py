"""Check representation, paired initialization and unchanged learning protocol."""

import ast
import inspect
import json

import torch
from torch.testing import assert_close

from research.transformer_stress.common import digest
from research.transformer_stress.residual import datasets, train_cell
from research.transformer_stress.run import PLAN as ORIGINAL_PLAN
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork
from research.transformer_stress_parity.learning import train_parity_cell
from research.transformer_stress_parity.models import ConfiguredNetwork
from research.transformer_stress_parity.run import PLAN, mixed_splits, summarize


def test_same_matrices_and_parent_representation_with_active_gates() -> None:
    """Gain one recovers parent outputs and common gradients beyond identity init."""
    settings = json.loads(ORIGINAL_PLAN.read_text())["residual_learning"]
    settings["activation_variant"] = "learned_even"
    depth = 4
    torch.manual_seed(2)
    parent = ShiftedResidualNetwork(settings, depth, "peri_rezero").double()
    torch.manual_seed(2)
    candidate = ConfiguredNetwork(settings, depth, "peri_rezero").double()
    common = {
        name: value
        for name, value in candidate.named_parameters()
        if not name.endswith("even_gain")
    }
    assert digest(dict(parent.named_parameters())) == digest(common)
    assert sum(p.numel() for p in candidate.parameters()) == (
        sum(p.numel() for p in parent.parameters()) + depth
    )
    with torch.no_grad():
        parent.gates.copy_(torch.linspace(0.1, 0.4, depth))
        candidate.gates.copy_(parent.gates)
        for branch in candidate.branches:
            branch.even_gain.fill_(1)
    x = torch.randn(8, settings["dimension"], dtype=torch.float64)
    before, after = parent(x), candidate(x)
    assert_close(before, after, rtol=1e-11, atol=1e-12)
    before.square().sum().backward()
    after.square().sum().backward()
    for name, value in parent.named_parameters():
        assert_close(value.grad, common[name].grad, rtol=1e-10, atol=1e-11)


def test_odd_start_with_active_residual_gates() -> None:
    """Initial branch symmetry holds without relying on all-zero residual gates."""
    settings = json.loads(ORIGINAL_PLAN.read_text())["residual_learning"]
    settings["activation_variant"] = "learned_even"
    model = ConfiguredNetwork(settings, 4, "peri_rezero")
    with torch.no_grad():
        model.gates.fill_(0.2)
    x = torch.randn(8, settings["dimension"])
    assert_close(model(-x), -model(x), rtol=0, atol=0)


def test_even_path_has_a_learning_gradient() -> None:
    """The odd initialization does not lock out a needed nonzero even function."""
    settings = json.loads(ORIGINAL_PLAN.read_text())["residual_learning"]
    settings["activation_variant"] = "learned_even"
    branch = ConfiguredNetwork(settings, 4, "peri_rezero").branches[0].double()
    x = torch.randn(8, settings["dimension"], dtype=torch.float64)
    with torch.no_grad():
        branch.even_gain.fill_(1)
        target_even = (branch(x) + branch(-x)) / 2
        branch.even_gain.zero_()
    even = (branch(x) + branch(-x)) / 2
    gradient = torch.autograd.grad((even * target_even).sum(), branch.even_gain)[0]
    assert_close(gradient, target_even.square().sum(), rtol=1e-12, atol=1e-14)
    assert gradient > 0


def test_mixed_teacher_keeps_inputs_and_original_labels() -> None:
    """Change the new teacher only, retaining a measurable even component."""
    settings = json.loads(ORIGINAL_PLAN.read_text())["residual_learning"]
    original = datasets(settings)
    before = {name: original[name]["residual"].clone() for name in ("train", "heldout")}
    mixed = mixed_splits(original, 0.5)
    for split in ("train", "heldout"):
        assert torch.equal(original[split]["residual"], before[split])
        assert mixed[split]["input"] is original[split]["input"]
        z = original[split]["input"] @ original["rotation"]
        assert_close(mixed[split]["residual"], z.tanh() + 0.5 * (z.square() - 1))
        assert (mixed[split]["residual"] - original[split]["residual"]).norm() > 0


def test_training_loop_changes_only_constructor() -> None:
    """Keep the original optimizer, samples, budget, stopping and scoring code."""
    original = inspect.getsource(train_cell)
    changed = (
        inspect.getsource(train_parity_cell)
        .replace("train_parity_cell", "train_cell", 1)
        .replace("ConfiguredNetwork", "ResidualNetwork", 1)
    )
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(changed))


def test_no_selection_before_both_teachers_finish() -> None:
    """Do not rank a symmetry-specific winner from a partially completed study."""
    result = summarize(
        [],
        json.loads(PLAN.read_text()),
        json.loads(ORIGINAL_PLAN.read_text())["residual_learning"],
    )
    assert not result["complete"]
    assert result["best_activation_variant"] is None
