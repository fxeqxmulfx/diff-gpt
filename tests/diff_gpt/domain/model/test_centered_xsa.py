"""Independent equations, gradients and repeated-value regressions for XSA."""

import copy
import unittest

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.centered_xsa import centered_projection
from diff_gpt.domain.model.mha import MultiHeadAttention
from diff_gpt.domain.model.rope import rope_tables


class CenteredXSATests(unittest.TestCase):
    """Preserve clipped real semantics while eliminating constant-value roundoff."""

    def setUp(self) -> None:
        """Use fixed CPU reductions and independent test inputs."""
        torch.set_num_threads(1)
        torch.manual_seed(147)

    def test_equations_and_all_gradients_including_clipping(self) -> None:
        """Check scores, values and own-value gradients against the source formula."""
        eps = 0.1
        for scale in (0.0, 0.5, 1.0, 2.0, 100.0):
            scores = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
            values = torch.randn(2, 8, 4, dtype=torch.float64, requires_grad=True)
            own = torch.zeros(2, 3, 4, dtype=torch.float64)
            own[..., 0] = scale * eps
            own.requires_grad_()
            weights = scores.softmax(-1)
            original_average = weights @ values
            unit = F.normalize(own, dim=-1, eps=eps)
            expected = (
                original_average
                - (original_average * unit).sum(-1, keepdim=True) * unit
            )
            anchor = values[:, :1]
            actual = centered_projection(
                weights @ (values - anchor), own - anchor, own, eps
            )
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            output_gradient = torch.randn_like(actual)
            expected_gradients = torch.autograd.grad(
                expected, (scores, values, own), output_gradient, retain_graph=True
            )
            actual_gradients = torch.autograd.grad(
                actual, (scores, values, own), output_gradient
            )
            torch.testing.assert_close(
                actual_gradients, expected_gradients, rtol=1e-10, atol=1e-10
            )

    def test_constant_values_are_exactly_zero(self) -> None:
        """A repeated token leaves exactly zero at every position in FP32."""
        model = MultiHeadAttention(32, 4, 0)
        values = torch.randn(2, 1, 32).expand(-1, 9, -1).clone()
        cos, sin = rope_tables(8, 9)
        output = model(values, cos, sin, None)
        assert torch.count_nonzero(output) == 0
        legacy = copy.deepcopy(model)
        legacy.centered_xsa = False
        assert torch.count_nonzero(legacy(values, cos, sin, None)[:, 1:]) > 0
        # Shared-token derivatives vanish too: the exact branch is identically zero.
        model = model.double()
        token = torch.randn(2, 1, 32, dtype=torch.float64, requires_grad=True)
        projected = model(token.expand(-1, 9, -1), cos.double(), sin.double(), None)
        direction = torch.randn_like(projected)
        (gradient,) = torch.autograd.grad((projected * direction).sum(), token)
        torch.testing.assert_close(
            gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0
        )

    def test_finite_difference_gradients_off_the_clipping_boundary(self) -> None:
        """Check both smooth clipping regions without relying on source autograd."""
        for scale in (0.01, 1.0):
            scores = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
            values = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)
            own = (torch.randn(1, 2, 2, dtype=torch.float64) * scale).requires_grad_()

            def function(
                scores: torch.Tensor, values: torch.Tensor, own: torch.Tensor
            ) -> torch.Tensor:
                anchor = values[:, :1]
                return centered_projection(
                    scores.softmax(-1) @ (values - anchor), own - anchor, own, 0.1
                )

            assert torch.autograd.gradcheck(function, (scores, values, own))

    def test_full_module_forward_and_parameter_gradients(self) -> None:
        """Centered and source modules agree on nonconstant float64 inputs."""
        centered = MultiHeadAttention(16, 2, 0).double()
        source = copy.deepcopy(centered)
        source.centered_xsa = False
        inputs = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
        cos, sin = (x.double() for x in rope_tables(8, 5))
        got = centered(inputs, cos, sin, None)
        want = source(inputs, cos, sin, None)
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-12)
        direction = torch.randn_like(got)
        got_grad = torch.autograd.grad(
            got, (inputs, *centered.parameters()), direction, retain_graph=True
        )
        want_grad = torch.autograd.grad(want, (inputs, *source.parameters()), direction)
        torch.testing.assert_close(got_grad, want_grad, rtol=1e-10, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
