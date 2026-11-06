"""Check isolation and complete evidence in the residual decay follow-up."""

import itertools
import json

import pytest
import torch
from torch.testing import assert_close

from research.transformer_stress.common import digest
from research.transformer_stress_decay import run
from research.transformer_stress_parity.learning import train_parity_cell
from research.transformer_stress_parity.models import ConfiguredNetwork

CONTROL_PASSES = 19


def specification() -> tuple[dict, dict]:
    """Use the frozen plans and original complete protocol."""
    return (
        json.loads(run.PLAN.read_text()),
        json.loads((run.PARENT / "manifest.json").read_text())["parent_settings"],
    )


def test_only_decay_changes_with_identical_matrices_rng_and_function() -> None:
    """Check active residual outputs and gradients, not only the initial identity."""
    plan, original = specification()
    assert run.train_parity_cell is train_parity_cell
    x = torch.randn(
        8, original["dimension"], generator=torch.Generator().manual_seed(11)
    )
    for activation in plan["activation_variants"]:
        reference = None
        for decay in (0.01, 0.1, 1.0):
            settings = run.settings_for(plan, original, activation, decay)
            baseline = dict(original, activation_variant=activation, learning_rate=0.01)
            assert {k for k in settings if settings[k] != baseline[k]} == (
                {"weight_decay"} if decay != plan["control_weight_decay"] else set()
            )
            torch.manual_seed(2)
            model = ConfiguredNetwork(settings, 4, "peri_rezero")
            initial = digest(dict(model.named_parameters()))
            rng = torch.get_rng_state().clone()
            parameters = sum(p.numel() for p in model.parameters())
            with torch.no_grad():
                model.gates.copy_(torch.linspace(0.1, 0.4, 4))
            output = model(x)
            output.square().sum().backward()
            gradients = {n: p.grad for n, p in model.named_parameters()}
            if reference is None:
                reference = (initial, rng, parameters, output.detach(), gradients)
            else:
                assert initial == reference[0]
                assert parameters == reference[2]
                assert torch.equal(rng, reference[1])
                assert_close(output, reference[3], rtol=0, atol=0)
                for name, value in gradients.items():
                    assert_close(value, reference[4][name], rtol=0, atol=0)


def test_controls_keep_both_teachers_and_all_failed_cells() -> None:
    """The follow-up must retain the parents' actual mixed-teacher regressions."""
    plan, original = specification()
    rows = run.control_rows(plan)
    assert len(rows) == plan["reused_control_cells"]
    assert len({run.key(row) for row in rows}) == len(rows)
    assert {r["teacher"] for r in rows} == set(plan["teachers"])
    assert all(r["weight_decay"] == plan["control_weight_decay"] for r in rows)
    assert sum(r["status"] == "passed" for r in rows) == CONTROL_PASSES
    summary = run.summarize(rows, plan, original)
    assert not summary["complete"]
    assert summary["best_configuration"] is None


def test_reject_duplicate_or_undeclared_cells() -> None:
    """A correct row count cannot conceal a missing seed behind a duplicate."""
    plan, original = specification()
    rows = run.control_rows(plan)
    with pytest.raises(ValueError, match="Duplicate"):
        run.summarize([*rows, rows[0]], plan, original)
    with pytest.raises(ValueError, match="Undeclared"):
        run.summarize([dict(rows[0], seed=9)], plan, original)
    with pytest.raises(ValueError, match="Undeclared weight decay"):
        run.settings_for(plan, original, "shifted", 0.3)


def test_selection_requires_full_grid_and_never_filters_a_miss() -> None:
    """A fast successful subset cannot beat full coverage on both teachers."""
    plan, original = specification()
    rows = []
    for activation, decay, teacher, depth, strength, seed in itertools.product(
        ("shifted", "learned_even"),
        (0.01, 0.1, 1.0),
        ("original_odd", "mixed"),
        (4, 16),
        (0.25, 1.0),
        (0, 1, 2),
    ):
        missed = (decay, teacher, depth, strength, seed) == (1.0, "mixed", 16, 1.0, 2)
        rows.append(
            {
                "activation_variant": activation,
                "weight_decay": decay,
                "teacher": teacher,
                "branches": depth,
                "strength": strength,
                "seed": seed,
                "status": "budget_miss" if missed else "passed",
                "first_threshold_updates": None
                if missed
                else {0.01: 96, 0.1: 64, 1.0: 32}[decay],
                "final_heldout_relative_mse": 0.1 if missed else 0.02,
            }
        )
    partial = run.summarize(rows[:-1], plan, original)
    assert not partial["complete"]
    assert partial["best_configuration"] is None
    full = run.summarize(rows, plan, original)
    assert full["complete"]
    assert len(full["cells"]) == plan["total_comparison_cells"]
    assert full["best_configuration"] == {
        "activation_variant": "shifted",
        "weight_decay": 0.1,
    }
    assert all(
        r["geometric_mean_first_updates"] is None
        for r in full["arms"]
        if r["weight_decay"] == 1.0
    )
