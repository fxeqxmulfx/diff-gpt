"""Verify the bias equation, unchanged protocol and complete paired evidence."""

import ast
import inspect
import itertools
import json

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from research.transformer_stress.common import digest
from research.transformer_stress_bias import run
from research.transformer_stress_bias.learning import train_bias_cell
from research.transformer_stress_bias.models import BiasedNetwork, is_new_bias
from research.transformer_stress_parity.learning import train_parity_cell
from research.transformer_stress_parity.models import ConfiguredNetwork

CONTROL_PASSES = 24
HIDDEN_WIDTH = 43


def specification() -> tuple[dict, dict]:
    """Read the fixed declaration and original protocol without modifying either."""
    return json.loads(run.PLAN.read_text()), run.read(run.PARENT / "manifest.json")[
        "parent_settings"
    ]


def test_initial_function_common_gradients_rng_and_declared_capacity() -> None:
    """Use active residual gates so initial identity cannot hide a changed branch."""
    plan, original = specification()
    x = torch.randn(
        8, original["dimension"], generator=torch.Generator().manual_seed(11)
    )
    for activation, depth, mode in itertools.product(
        plan["activation_variants"], original["branches"], ("none", "output", "all")
    ):
        settings = run.settings_for(plan, original, activation, mode)
        baseline = dict(
            original,
            activation_variant=activation,
            learning_rate=0.01,
            weight_decay=0.1,
        )
        assert {k: v for k, v in settings.items() if k != "bias_mode"} == baseline
        torch.manual_seed(2)
        parent = ConfiguredNetwork(baseline, depth, "peri_rezero")
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2)
        candidate = BiasedNetwork(settings, depth, "peri_rezero")
        assert torch.equal(torch.get_rng_state(), rng)
        common = {n: p for n, p in candidate.named_parameters() if not is_new_bias(n)}
        assert digest(common) == digest(dict(parent.named_parameters()))
        extra = (
            0
            if mode == "none"
            else depth * (16 + (2 * HIDDEN_WIDTH if mode == "all" else 0))
        )
        assert candidate.added_bias_parameters() == extra
        assert (
            sum(p.numel() for p in candidate.parameters())
            == sum(p.numel() for p in parent.parameters()) + extra
        )
        assert all(
            torch.count_nonzero(p) == 0
            for n, p in candidate.named_parameters()
            if is_new_bias(n)
        )
        with torch.no_grad():
            parent.gates.copy_(torch.linspace(0.1, 0.4, depth))
            candidate.gates.copy_(parent.gates)
        a, b = parent(x), candidate(x)
        assert_close(a, b, rtol=0, atol=0)
        a.square().sum().backward()
        b.square().sum().backward()
        for name, parameter in parent.named_parameters():
            assert_close(parameter.grad, common[name].grad, rtol=0, atol=0)
        assert all(torch.isfinite(p.grad).all() for p in candidate.parameters())
        if extra:
            assert all(
                torch.count_nonzero(p.grad) > 0
                for n, p in candidate.named_parameters()
                if is_new_bias(n)
            )


def test_nonzero_bias_equation_and_all_gradients_independently() -> None:
    """Use a sigmoid expression for the independent learned-even reference."""
    plan, original = specification()
    for activation, mode in itertools.product(
        plan["activation_variants"], plan["bias_modes"]
    ):
        settings = run.settings_for(plan, original, activation, mode)
        torch.manual_seed(3)
        branch = BiasedNetwork(settings, 4, "peri_rezero").double().branches[0]
        with torch.no_grad():
            branch.output_bias.copy_(torch.linspace(-0.1, 0.1, 16))
            if branch.hidden_bias is not None:
                branch.hidden_bias.copy_(torch.linspace(-0.2, 0.2, 2 * HIDDEN_WIDTH))
            if hasattr(branch, "even_gain"):
                branch.even_gain.fill_(0.3)
        x = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
        g, v = F.linear(x, branch.c_fc.weight, branch.hidden_bias).chunk(2, -1)
        factor = 1 + g * g.sigmoid()
        if hasattr(branch, "even_gain"):
            factor = factor + 0.5 * (branch.even_gain - 1) * g
        expected = F.linear(factor * v, branch.c_proj.weight, branch.output_bias)
        actual = branch(x)
        assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        probe = torch.randn_like(actual)
        inputs = [x, *branch.parameters()]
        first = torch.autograd.grad((actual * probe).sum(), inputs, retain_graph=True)
        second = torch.autograd.grad((expected * probe).sum(), inputs)
        for a, b in zip(first, second, strict=True):
            assert_close(a, b, rtol=1e-11, atol=1e-12)


def test_training_loop_matches_parent_except_constructor_and_capacity_metadata() -> (
    None
):
    """Compare the entire optimizer/sampler/observation AST, not a mock trainer."""
    source = ast.parse(inspect.getsource(train_bias_cell))
    parent = ast.parse(inspect.getsource(train_parity_cell))

    class Canonical(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == "BiasedNetwork":
                node.id = "ConfiguredNetwork"
            return node

        def visit_Call(self, node: ast.Call) -> ast.AST:
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "common_branch_parameters"
            ):
                return ast.parse(
                    "dict(model.branches.named_parameters())", mode="eval"
                ).body
            return self.generic_visit(node)

        def visit_Dict(self, node: ast.Dict) -> ast.Dict:
            pairs = [
                (k, v)
                for k, v in zip(node.keys, node.values, strict=True)
                if not (
                    isinstance(k, ast.Constant) and k.value == "added_bias_parameters"
                )
            ]
            node.keys, node.values = [k for k, _ in pairs], [v for _, v in pairs]
            return self.generic_visit(node)

    source.body[0].name = parent.body[0].name
    assert ast.dump(Canonical().visit(source)) == ast.dump(parent)


def test_retain_every_control_and_reject_incomplete_or_invalid_grids() -> None:
    """No failed teacher, duplicate seed or undeclared arm may disappear."""
    plan, original = specification()
    rows = run.control_rows(plan)
    assert len(rows) == plan["reused_control_cells"]
    assert sum(r["status"] == "passed" for r in rows) == CONTROL_PASSES
    assert all(
        r["weight_decay"] == plan["weight_decay"] and r["bias_mode"] == "none"
        for r in rows
    )
    assert len({run.key(r) for r in rows}) == len(rows)
    summary = run.summarize(rows, plan, original)
    assert not summary["complete"]
    assert summary["best_configuration"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        run.summarize([*rows, rows[0]], plan, original)
    with pytest.raises(ValueError, match="Undeclared"):
        run.summarize([dict(rows[0], seed=9)], plan, original)
    with pytest.raises(ValueError, match="Undeclared bias mode"):
        run.settings_for(plan, original, "shifted", "other")


def test_selection_keeps_misses_and_requires_all_teacher_cells() -> None:
    """A faster successful subset must not defeat complete pooled coverage."""
    plan, original = specification()
    rows = []
    for activation, mode, teacher, depth, strength, seed in itertools.product(
        ("shifted", "learned_even"),
        ("none", "output", "all"),
        ("original_odd", "mixed"),
        (4, 16),
        (0.25, 1.0),
        (0, 1, 2),
    ):
        missed = (mode, teacher, depth, strength, seed) == ("all", "mixed", 16, 1.0, 2)
        rows.append(
            {
                "activation_variant": activation,
                "bias_mode": mode,
                "teacher": teacher,
                "branches": depth,
                "strength": strength,
                "seed": seed,
                "status": "budget_miss" if missed else "passed",
                "first_threshold_updates": None
                if missed
                else (1 if mode == "all" else 64),
                "final_heldout_relative_mse": 0.1 if missed else 0.01,
            }
        )
    assert run.summarize(rows[:-1], plan, original)["best_configuration"] is None
    result = run.summarize(rows, plan, original)
    assert result["complete"]
    assert result["best_configuration"]["bias_mode"] != "all"
    assert all(
        r["geometric_mean_first_updates"] is None
        for r in result["arms"]
        if r["bias_mode"] == "all"
    )
