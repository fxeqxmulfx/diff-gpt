"""Verify the fix equations, unchanged learning protocol and cached full model."""

import ast
import inspect
import json

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.kv_cache import KVCache
from research.transformer_stress.common import EPS, digest
from research.transformer_stress.residual import ResidualNetwork, train_cell
from research.transformer_stress.run import PLAN
from research.transformer_stress_fixes.learning import (
    ShiftedResidualNetwork,
    train_shifted_cell,
)
from research.transformer_stress_fixes.models import (
    ShiftedSwiGLU,
    fixed_model,
    safe_rms,
)


@pytest.mark.parametrize("scale", [0, 1e-20, 1e-4, 1, 1e10, 1e20, 1e30])
def test_safe_rms_forward_and_radial_gradient(scale: float) -> None:
    """Use a closed-form radial VJP that does not cancel in the reference."""
    torch.manual_seed(4)
    x = (torch.randn(3, 32) * scale).requires_grad_()
    actual = safe_rms(x)
    wide = x.detach().double()
    energy = wide.square().mean(-1, keepdim=True)
    expected = wide / (energy + EPS).sqrt()
    gradient = torch.autograd.grad((actual * x.detach()).sum(), x)[0]
    expected_gradient = (EPS * wide / (energy + EPS).pow(1.5)).float()
    assert_close(actual, expected.float(), rtol=2e-6, atol=0)
    assert_close(gradient, expected_gradient, rtol=2e-6, atol=0)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(gradient).all()


def test_safe_rms_random_direction_finite_difference() -> None:
    """Check the full VJP away from the special radial witnesses."""
    torch.manual_seed(5)
    x = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(safe_rms, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_combined_fix_dense_cache_and_causality() -> None:
    """Retain FP32 cache values and global positions after widening internals."""
    settings = json.loads(PLAN.read_text())["components"]
    model = fixed_model(settings, 2, 8, "all_numeric").eval()
    tokens = torch.tensor([[2, 2, 4, 6, 1, 5, 7], [3, 3, 3, 4, 5, 7, 8]])
    with torch.no_grad():
        model.lerope.log_scales.fill_(0.3)
        dense = model(tokens)[0]
        state = KVCache(batch_size=2, num_heads=4, seq_len=8, head_dim=8, num_layers=2)
        cached = torch.cat(
            [
                model(tokens[:, i : i + 1], kv_cache=state)[0]
                for i in range(tokens.shape[1])
            ],
            1,
        )
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % settings["full_stack_vocab"]
        assert_close(dense[:, :4], model(changed)[0][:, :4], rtol=0, atol=0)
    assert_close(dense, cached, rtol=4e-5, atol=5e-6)
    assert model.token_embedding_table.weight is model.lm_head.weight
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_linear_path_and_parameter_pairing() -> None:
    """Zero gate projection yields a learned linear map, with no new parameters."""
    original = FeedForward(8)
    shifted = ShiftedSwiGLU(original)
    half = original.c_fc.out_features // 2
    with torch.no_grad():
        original.c_fc.weight[:half].zero_()
    x = torch.randn(2, 8)
    assert_close(
        shifted(x),
        F.linear(F.linear(x, original.c_fc.weight[half:]), original.c_proj.weight),
    )
    assert torch.count_nonzero(original(x)) == 0
    settings = json.loads(PLAN.read_text())["residual_learning"]
    torch.manual_seed(2)
    parent = ResidualNetwork(settings, 4, "peri_post_gate")
    torch.manual_seed(2)
    candidate = ShiftedResidualNetwork(settings, 4, "peri_post_gate")
    assert digest(dict(parent.named_parameters())) == digest(
        dict(candidate.named_parameters())
    )


def test_learning_protocol_changes_only_model_constructor() -> None:
    """AST comparison guards the inherited optimizer, data, budget and scoring code."""
    original = inspect.getsource(train_cell)
    changed = (
        inspect.getsource(train_shifted_cell)
        .replace("train_shifted_cell", "train_cell", 1)
        .replace("ShiftedResidualNetwork", "ResidualNetwork", 1)
    )
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(changed))


def test_bounded_frequency_is_a_declared_model_change() -> None:
    """Match initial values/derivatives and keep the extreme raw parameter finite."""
    settings = json.loads(PLAN.read_text())["components"]
    unbounded = fixed_model(settings, 2, 8, "all_numeric").lerope
    bounded = fixed_model(settings, 2, 8, "all_numeric_bounded_rope").lerope
    ordinary = sum(x.sum() for x in unbounded(0, 8))
    controlled = sum(x.sum() for x in bounded(0, 8))
    assert_close(ordinary, controlled, rtol=0, atol=0)
    assert_close(
        torch.autograd.grad(ordinary, unbounded.log_scales)[0],
        torch.autograd.grad(controlled, bounded.log_scales)[0],
        rtol=0,
        atol=0,
    )
    with torch.no_grad():
        bounded.log_scales.fill_(90)
    value = sum(x.sum() for x in bounded(0, 8))
    assert torch.isfinite(value)
    assert torch.isfinite(torch.autograd.grad(value, bounded.log_scales)[0]).all()
