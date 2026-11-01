"""Check coordinate changes against independent physical-query equations."""

import copy
import math
import unittest

import pytest
import torch
from torch.nn import functional as F

from diff_gpt.application.grokking.residual_attention_probe import dataset
from diff_gpt.application.residual_conditioning import DepthQueryConditioner, stable_rms
from diff_gpt.domain.model.conditioned_attention_residual import (
    ConditionedQKAttentionResidual,
)
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW


class ResidualConditioningTests(unittest.TestCase):
    """Audit outputs, true derivatives, averaging and complete saved states."""

    def setUp(self) -> None:
        """Keep independent CPU reductions deterministic."""
        torch.manual_seed(150)
        torch.set_num_threads(1)

    def test_coordinate_equations_and_all_gradients(self) -> None:
        """Autodiff in the chart equals the chain rule of the physical query."""
        for scale in (1e-4, 1.0, 1e6):
            model = ConditionedQKAttentionResidual(4).double()
            model.query_scale.fill_(scale)
            values = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
            actual = model(values)
            physical = model.query * scale
            q = F.rms_norm(physical, (4,), eps=model.eps)
            k = F.rms_norm(values, (4,), eps=model.eps)
            expected = (((k * q).sum(-1) / 2).softmax(-1)[..., None] * values).sum(-2)
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
            a = torch.autograd.grad(
                actual.square().sum(), (model.query, values), retain_graph=True
            )
            b = torch.autograd.grad(expected.square().sum(), (model.query, values))
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)

    def test_reconditioning_preserves_query_average_and_second_moment(self) -> None:
        """Both model modes and every accumulated gradient component transform."""
        model = ConditionedQKAttentionResidual(8).double()
        with torch.no_grad():
            model.query.mul_(1e6)
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01, betas=(0.9, 0.98))
        state = optimizer.state[model.query]
        state["base"].mul_(1.2).add_(torch.arange(8, dtype=torch.float64) * 100)
        gradients = [torch.randn(8, dtype=torch.float64) / 1e6 for _ in range(7)]
        for gradient in gradients:
            state["second"].mul_(0.98).addcmul_(gradient, gradient, value=0.02)
        optimizer.param_groups[0].update(step=7, maximum=0.01, total=0.0007)
        query = model.query.detach().clone()
        base = state["base"].clone()
        second = state["second"].clone()
        average = torch.lerp(query, base, 1 - 1 / 0.9)
        values = torch.randn(3, 5, 8, dtype=torch.float64)
        before = model(values).detach()
        factor = DepthQueryConditioner(model, optimizer)()
        torch.testing.assert_close(
            model.query * model.query_scale, query, rtol=1e-14, atol=1e-9
        )
        torch.testing.assert_close(
            state["base"] * model.query_scale, base, rtol=1e-14, atol=1e-9
        )
        torch.testing.assert_close(model(values), before, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            state["second"], second * factor.square(), rtol=1e-14, atol=1e-14
        )
        reconstructed = torch.zeros_like(second)
        for gradient in gradients:
            transformed = gradient * factor
            reconstructed.mul_(0.98).addcmul_(transformed, transformed, value=0.02)
        torch.testing.assert_close(
            state["second"], reconstructed, rtol=1e-14, atol=1e-14
        )
        optimizer.eval()
        torch.testing.assert_close(
            model.query * model.query_scale, average, rtol=1e-14, atol=1e-9
        )
        expected_q = F.rms_norm(average, (8,), eps=model.eps)
        scores = (F.rms_norm(values, (8,), eps=model.eps) * expected_q).sum(
            -1
        ) / math.sqrt(8)
        expected = (scores.softmax(-1)[..., None] * values).sum(-2)
        torch.testing.assert_close(model(values), expected, rtol=1e-12, atol=1e-12)
        group = optimizer.param_groups[0]
        assert (group["step"], group["maximum"], group["total"]) == (7, 0.01, 0.0007)

    def test_zero_tiny_and_ordinary_queries_are_unchanged(self) -> None:
        """The correction only shrinks oversized coordinates."""
        for scale in (0.0, 1e-8, 1.0):
            model = ConditionedQKAttentionResidual(8)
            with torch.no_grad():
                model.query.mul_(scale)
            optimizer = ScheduleFreeAdamW(model.parameters())
            saved = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
            assert float(DepthQueryConditioner(model, optimizer)()) == 1
            torch.testing.assert_close(model.state_dict(), saved[0], rtol=0, atol=0)
            torch.testing.assert_close(optimizer.state_dict(), saved[1], rtol=0, atol=0)

    def test_large_query_gets_usable_gradient_without_changing_forward(self) -> None:
        """Recover the coordinate gradient lost at the original huge query."""
        model = ConditionedQKAttentionResidual(16)
        with torch.no_grad():
            model.query.zero_()
            model.query[0], model.query[1] = -math.sqrt(8) * 1e6, math.sqrt(8) * 1e6
        values, targets, _ = dataset(123)
        before = model(values)
        old_gradient = torch.autograd.grad(F.mse_loss(before, targets), model.query)[0]
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01)
        factor = DepthQueryConditioner(model, optimizer)()
        after = model(values)
        gradient = torch.autograd.grad(F.mse_loss(after, targets), model.query)[0]
        torch.testing.assert_close(after, before, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(
            gradient, old_gradient * factor, rtol=2e-5, atol=1e-8
        )
        minimum_ratio = 1e5
        assert float(gradient.norm()) > minimum_ratio * float(old_gradient.norm())
        unit_rms_tolerance = 1.000001
        assert float(stable_rms(model.query.detach())) <= unit_rms_tolerance

    def test_saved_coordinate_and_optimizer_state_resume_exactly(self) -> None:
        """Saving preserves chart metadata, averaged weights and all optimizer state."""
        model = ConditionedQKAttentionResidual(8).double()
        with torch.no_grad():
            model.query.mul_(1e6)
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.1, betas=(0.9, 0.98))
        conditioner = DepthQueryConditioner(model, optimizer, maximum_rms=1.0)
        values = torch.randn(3, 5, 8, dtype=torch.float64)
        targets = values[:, 0].clone()

        def steps(
            net: ConditionedQKAttentionResidual,
            opt: ScheduleFreeAdamW,
            condition: DepthQueryConditioner,
            count: int,
        ) -> None:
            for _ in range(count):
                opt.zero_grad(set_to_none=True)
                condition()
                F.mse_loss(net(values), targets).backward()
                opt.step()

        steps(model, optimizer, conditioner, 7)
        optimizer.zero_grad(set_to_none=True)
        optimizer.eval()
        saved = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
        restored = ConditionedQKAttentionResidual(8).double()
        restored.load_state_dict(saved[0])
        resumed = ScheduleFreeAdamW(restored.parameters())
        resumed.load_state_dict(saved[1])
        optimizer.train()
        resumed.train()
        steps(model, optimizer, conditioner, 13)
        steps(
            restored,
            resumed,
            DepthQueryConditioner(restored, resumed, maximum_rms=1.0),
            13,
        )
        torch.testing.assert_close(
            model.state_dict(), restored.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            optimizer.state_dict(), resumed.state_dict(), rtol=0, atol=0
        )

    def test_wrong_state_or_call_order_is_rejected(self) -> None:
        """A chart change cannot silently consume an old-coordinate gradient."""
        model = ConditionedQKAttentionResidual(4)
        other = ConditionedQKAttentionResidual(4)
        optimizer = ScheduleFreeAdamW(model.parameters())
        with pytest.raises(ValueError, match="own"):
            DepthQueryConditioner(other, optimizer)
        with pytest.raises(ValueError, match="at least one"):
            DepthQueryConditioner(model, optimizer, maximum_rms=0.5)
        conditioner = DepthQueryConditioner(model, optimizer)
        optimizer.eval()
        with pytest.raises(RuntimeError, match="train"):
            conditioner()
        optimizer.train()
        model.query.grad = torch.ones_like(model.query)
        with pytest.raises(RuntimeError, match="zero_grad"):
            conditioner()
