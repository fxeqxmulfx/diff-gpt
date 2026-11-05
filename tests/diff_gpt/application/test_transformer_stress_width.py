"""Validate capacity disclosure, original columns and the unchanged learning loop."""

import ast
import inspect
import json

import torch

from research.transformer_stress.common import digest
from research.transformer_stress.residual import train_cell
from research.transformer_stress.run import PLAN
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork
from research.transformer_stress_width.learning import train_expanded_cell
from research.transformer_stress_width.models import ExpandedNetwork
from research.transformer_stress_width.run import PLAN as WIDTH_PLAN
from research.transformer_stress_width.run import summarize


def test_widening_preserves_original_columns_and_counts_extra_parameters() -> None:
    """The larger model retains all common draws and discloses its added capacity."""
    settings = json.loads(PLAN.read_text())["residual_learning"]
    width_plan = json.loads(WIDTH_PLAN.read_text())
    settings["expanded_hidden"] = width_plan["new_hidden_widths"][0]
    depth = settings["branches"][0]
    torch.manual_seed(2)
    original = ShiftedResidualNetwork(settings, depth, "peri_rezero")
    torch.manual_seed(2)
    expanded = ExpandedNetwork(settings, depth, "peri_rezero")
    for small, large in zip(original.branches, expanded.branches, strict=True):
        old, new = small.c_proj.in_features, large.c_proj.in_features
        assert torch.equal(small.c_fc.weight[:old], large.c_fc.weight[:old])
        assert torch.equal(small.c_fc.weight[old:], large.c_fc.weight[new : new + old])
        assert torch.equal(small.c_proj.weight, large.c_proj.weight[:, :old])
    added = depth * 3 * settings["dimension"] * (new - old)
    assert (
        sum(p.numel() for p in expanded.parameters())
        == sum(p.numel() for p in original.parameters()) + added
    )
    torch.manual_seed(2)
    repeated = ExpandedNetwork(settings, depth, "peri_rezero")
    assert digest(dict(repeated.named_parameters())) == digest(
        dict(expanded.named_parameters())
    )


def test_width_ablation_preserves_training_protocol() -> None:
    """Only the model constructor differs from the frozen v1 learning procedure."""
    original = inspect.getsource(train_cell)
    changed = (
        inspect.getsource(train_expanded_cell)
        .replace("train_expanded_cell", "train_cell", 1)
        .replace("ExpandedNetwork", "ResidualNetwork", 1)
    )
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(changed))


def test_width_report_retains_pending_conditions() -> None:
    """Never select from an incomplete capacity comparison."""
    plan = json.loads(WIDTH_PLAN.read_text())
    report = summarize([], plan)
    assert report["best_hidden_width"] is None
    assert report["complete"] is False
    assert all(row["completed"] == 0 for row in report["widths"])
