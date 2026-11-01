"""Independent finite-epsilon equations, gradients and depth aggregation."""

import math
import unittest

import torch
from torch.testing import assert_close

from diff_gpt.domain.model.attention_residual import (
    QKAttentionResidual,
    depth_attention,
    depth_scores,
)


class AttentionResidualTests(unittest.TestCase):
    """Check the mathematical operator before using it in a transformer."""

    def setUp(self) -> None:
        """Fix draws and CPU reduction order."""
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def test_equations_and_gradients(self) -> None:
        """Compare every query and value derivative with scalar equations."""
        eps = 1e-6
        values = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        query = torch.randn(4, dtype=torch.float64, requires_grad=True)
        q = query / (query.square().mean() + eps).sqrt()
        keys = values / (values.square().mean(-1, keepdim=True) + eps).sqrt()
        scores = torch.stack(
            [
                torch.stack([sum(q[k] * row[k] for k in range(4)) / 2 for row in batch])
                for batch in keys
            ]
        )
        expected = (scores.softmax(-1).unsqueeze(-1) * values).sum(-2)
        actual = depth_attention(values, query, eps)
        assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        a = torch.autograd.grad(
            actual.square().sum(), (query, values), retain_graph=True
        )
        b = torch.autograd.grad(expected.square().sum(), (query, values))
        assert_close(a, b, rtol=1e-10, atol=1e-12)
        assert torch.autograd.gradcheck(
            lambda v, q: depth_attention(v, q, eps), (values, query)
        )

    def test_zero_query_single_source_and_identical_values(self) -> None:
        """Zero query averages; one or identical sources are unchanged."""
        model = QKAttentionResidual(8).double()
        values = torch.randn(2, 3, 8, dtype=torch.float64)
        with torch.no_grad():
            model.query.zero_()
        assert_close(model(values), values.mean(-2), rtol=1e-14, atol=1e-14)
        assert_close(model(values[:, :1]), values[:, 0], rtol=0, atol=0)
        repeated = values[:, :1].expand(-1, 5, -1)
        assert_close(model(repeated), values[:, 0], rtol=1e-14, atol=1e-14)
        zero = torch.zeros_like(values, requires_grad=True)
        result = model(zero)
        result.sum().backward()
        assert torch.isfinite(zero.grad).all()
        assert torch.isfinite(model.query.grad).all()

    def test_score_and_value_norm_bounds_across_scales(self) -> None:
        """Unbounded input/query scales cannot create unbounded logits."""
        for width in (1, 4, 16):
            values = torch.randn(2, 7, width, dtype=torch.float64)
            query = torch.randn(width, dtype=torch.float64)
            for scale in (0.0, 1e-8, 1.0, 1e8):
                scores = depth_scores(values * scale, query * scale, 1e-6)
                assert float(scores.abs().max()) <= math.sqrt(width) + 1e-12
                output = depth_attention(values * scale, query * scale, 1e-6)
                bound = (values * scale).norm(dim=-1).amax(-1)
                assert bool((output.norm(dim=-1) <= bound + 1e-07).all())

    def test_small_epsilon_is_not_exact_scale_invariance(self) -> None:
        """Below sqrt(eps), the normalization deliberately attenuates scores."""
        values = torch.eye(4, dtype=torch.float64)
        query = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.float64)
        ordinary = depth_scores(values, query, 1e-6)
        tiny_scale = 1e-9
        tiny = depth_scores(values * tiny_scale, query * tiny_scale, 1e-6)
        assert float(ordinary.abs().max()) > 1.0
        assert float(tiny.abs().max()) < tiny_scale

    def test_initial_query_scale(self) -> None:
        """Initialize away from the steep zero-query region."""
        model = QKAttentionResidual(8)
        assert_close(
            model.query.square().mean(), torch.tensor(1.0), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
