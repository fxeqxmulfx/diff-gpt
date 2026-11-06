"""Independent projection equations, unchanged protocol and complete evidence."""

import ast
import inspect
import itertools

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from research.transformer_stress.common import EPS, digest
from research.transformer_stress_bias.learning import train_bias_cell
from research.transformer_stress_bias.models import BiasedNetwork
from research.transformer_stress_projection import run
from research.transformer_stress_projection.learning import train_projection_cell
from research.transformer_stress_projection.models import (
    ProjectedNetwork,
    project_branch,
)

CONTROL_CELLS = 24
CONTROL_PASSES = 17


def specification() -> tuple[dict, dict]:
    """Read the fixed declaration and complete parent's original protocol."""
    return run.read(run.PLAN), run.read(run.PARENT / "manifest.json")["parent_settings"]


def test_initial_parameters_rng_capacity_and_identity() -> None:
    """Every parameter/order/draw is preserved, including biases and residual gates."""
    plan, original = specification()
    for depth in original["branches"]:
        settings = run.settings_for(plan, original, "ball")
        torch.manual_seed(2)
        parent = BiasedNetwork(settings, depth, plan["variant"])
        state = torch.get_rng_state().clone()
        torch.manual_seed(2)
        model = ProjectedNetwork(settings, depth, plan["variant"])
        assert torch.equal(state, torch.get_rng_state())
        assert digest(dict(parent.named_parameters())) == digest(
            dict(model.named_parameters())
        )
        assert list(parent.state_dict()) == list(model.state_dict())
        assert parent.added_bias_parameters() == model.added_bias_parameters()
        assert sum(p.numel() for p in parent.parameters()) == sum(
            p.numel() for p in model.parameters()
        )
        x = torch.randn(
            5, original["dimension"], generator=torch.Generator().manual_seed(7)
        )
        assert torch.equal(parent(x), x)
        assert torch.equal(model(x), x)
        assert {k: v for k, v in settings.items() if k != "projection_mode"} == dict(
            original,
            activation_variant="learned_even",
            bias_mode="all",
            learning_rate=0.01,
            weight_decay=0.1,
        )


def test_active_gate_full_forward_and_all_gradients_independent() -> None:
    """Use sigmoid activation and explicit matrix algebra with active gates/biases."""
    plan, original = specification()
    settings = run.settings_for(plan, original, "ball")
    torch.manual_seed(4)
    model = ProjectedNetwork(settings, 4, plan["variant"]).double()
    with torch.no_grad():
        model.gates.copy_(torch.linspace(0.2, 0.8, 4))
        for branch in model.branches:
            branch.even_gain.fill_(0.3)
            branch.hidden_bias.copy_(torch.linspace(-0.1, 0.1, 86))
            branch.output_bias.copy_(torch.linspace(-0.2, 0.2, 16))
    x = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
    actual = model(x)
    expected = x
    radius = 4
    for branch, gate in zip(model.branches, model.gates, strict=True):
        normed = expected / (expected.square().mean(-1, keepdim=True) + EPS).sqrt()
        g, v = F.linear(normed, branch.c_fc.weight, branch.hidden_bias).chunk(2, -1)
        hidden = (1 + g * g.sigmoid() + 0.5 * (branch.even_gain - 1) * g) * v
        raw = F.linear(hidden, branch.c_proj.weight, branch.output_bias)
        length = raw.square().sum(-1, keepdim=True).sqrt()
        projected = torch.where(length < radius, raw, raw * (radius / length))
        expected = expected + projected * gate / 2
    assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(actual)
    inputs = [x, *model.parameters()]
    a = torch.autograd.grad((actual * probe).sum(), inputs, retain_graph=True)
    b = torch.autograd.grad((expected * probe).sum(), inputs)
    for actual_grad, expected_grad in zip(a, b, strict=True):
        assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)
    assert all(torch.isfinite(g).all() for g in a)


def test_zero_interior_exterior_and_boundary_jacobians() -> None:
    """Do not use finite differences at the kink; check the chosen exterior VJP."""
    radius = 2.0
    for value in (
        torch.zeros(4),
        torch.tensor([0.1, -0.2, 0.3, -0.4]),
        torch.tensor([8.0, -6.0, 0.0, 0.0]),
        torch.tensor([2.0, 0.0, 0.0, 0.0]),
    ):
        x = value.double()
        length = x.norm().item()
        expected = torch.eye(4, dtype=torch.float64)
        if length >= radius:
            unit = x / length
            expected = radius / length * (expected - torch.outer(unit, unit))
        actual = torch.autograd.functional.jacobian(project_branch, x)
        assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        assert torch.linalg.matrix_norm(actual, ord=2) <= 1 + 1e-12
        if length <= radius:
            assert torch.equal(project_branch(x), x)
        else:
            assert_close(project_branch(x), radius * x / length, rtol=1e-12, atol=1e-12)


def test_gradcheck_away_from_boundary() -> None:
    """Independent finite differences check both ordinary and clipped rows."""
    x = torch.tensor(
        [[0.1, -0.2], [4.0, -3.0]], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(project_branch, (x,), eps=1e-6, atol=1e-6)


def test_training_loop_matches_parent_except_model_constructor() -> None:
    """The entire optimizer/sampler/observation AST must match the frozen parent."""
    source = ast.parse(inspect.getsource(train_projection_cell))
    parent = ast.parse(inspect.getsource(train_bias_cell))

    class Canonical(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == "ProjectedNetwork":
                node.id = "BiasedNetwork"
            return node

    source.body[0].name = parent.body[0].name
    assert ast.dump(Canonical().visit(source)) == ast.dump(parent)


def test_retain_every_control_and_reject_invalid_or_incomplete_grids() -> None:
    """Preserve the near-threshold odd failure and all six strong mixed misses."""
    plan, original = specification()
    rows = run.control_rows(plan)
    assert len(rows) == plan["reused_control_cells"] == CONTROL_CELLS
    assert sum(r["status"] == "passed" for r in rows) == CONTROL_PASSES
    assert all(r["projection_mode"] == "rms" for r in rows)
    result = run.summarize(rows, plan, original)
    assert not result["complete"]
    assert result["best_configuration"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        run.summarize([*rows, rows[0]], plan, original)
    with pytest.raises(ValueError, match="Undeclared"):
        run.summarize([dict(rows[0], seed=4)], plan, original)
    with pytest.raises(ValueError, match="Undeclared"):
        run.settings_for(plan, original, "other")


def test_complete_grid_selection_keeps_misses_and_both_teachers() -> None:
    """Faster successes cannot defeat complete pooled coverage."""
    plan, original = specification()
    rows = []
    for mode, teacher, depth, strength, seed in itertools.product(
        ("rms", "ball"), ("original_odd", "mixed"), (4, 16), (0.25, 1.0), (0, 1, 2)
    ):
        missed = (mode, teacher, depth, strength, seed) == ("ball", "mixed", 16, 1.0, 2)
        rows.append(
            {
                "projection_mode": mode,
                "teacher": teacher,
                "branches": depth,
                "strength": strength,
                "seed": seed,
                "status": "budget_miss" if missed else "passed",
                "first_threshold_updates": None
                if missed
                else (32 if mode == "ball" else 128),
                "final_heldout_relative_mse": 0.1 if missed else 0.01,
            }
        )
    assert run.summarize(rows[:-1], plan, original)["best_configuration"] is None
    result = run.summarize(rows, plan, original)
    assert result["complete"]
    assert result["best_configuration"] == "rms"
    assert (
        next(r for r in result["arms"] if r["projection_mode"] == "ball")[
            "geometric_mean_first_updates"
        ]
        is None
    )
