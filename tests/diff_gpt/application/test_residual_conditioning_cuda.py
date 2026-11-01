"""Check the compiled repair against independent physical-query FP64 equations."""

import unittest

import torch

from diff_gpt.application.residual_conditioning import DepthQueryConditioner
from diff_gpt.domain.model.conditioned_attention_residual import (
    ConditionedQKAttentionResidual,
)
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW


class CompiledResidualConditioningTests(unittest.TestCase):
    """Exercise real CUDA compilation, the state kernel and both input gradients."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_compiled_physical_equations_and_all_gradients(self) -> None:
        """No model or conditioner implementation supplies the FP64 reference."""
        torch.manual_seed(150)
        torch.set_num_threads(1)
        model = ConditionedQKAttentionResidual(16).cuda()
        with torch.no_grad():
            model.query.mul_(1e6)
        optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.01, betas=(0.9, 0.98))
        conditioner = DepthQueryConditioner(model, optimizer, compiled=True)
        compiled = torch.compile(
            model, mode="max-autotune", fullgraph=True, dynamic=False
        )
        values = torch.randn(64, 8, 16, device="cuda", requires_grad=True)
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad(set_to_none=True)
            conditioner()
            actual = compiled(values)
            actual_gradients = torch.autograd.grad(
                actual.square().mean(), (model.query, values)
            )
            coordinate = model.query.detach().cpu().double().requires_grad_()
            inputs = values.detach().cpu().double().requires_grad_()
            physical = coordinate * model.query_scale.detach().cpu().double()
            query = physical / (physical.square().mean() + model.eps).sqrt()
            keys = inputs / (inputs.square().mean(-1, keepdim=True) + model.eps).sqrt()
            weights = ((keys * query).sum(-1) / 4).softmax(-1)
            reference = (weights[..., None] * inputs).sum(-2)
            expected_gradients = torch.autograd.grad(
                reference.square().mean(), (coordinate, inputs)
            )
            torch.testing.assert_close(
                actual.detach().cpu().double(), reference, rtol=3e-5, atol=3e-6
            )
            for actual_gradient, expected in zip(
                actual_gradients, expected_gradients, strict=True
            ):
                torch.testing.assert_close(
                    actual_gradient.cpu().double(), expected, rtol=3e-5, atol=3e-6
                )
