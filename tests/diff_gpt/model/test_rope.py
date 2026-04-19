import pytest
import torch
from torch.testing import assert_close

from diff_gpt.model.rope import apply_rotary_emb


def _freqs_cis_from_theta(theta: torch.Tensor) -> torch.Tensor:
    """Pack (cos θ, sin θ) into the trailing dim — matches the
    real-valued shape `apply_rotary_emb` expects."""
    return torch.stack([theta.cos(), theta.sin()], dim=-1)


def test_rotation_geometry_pi_2():
    r"""
    Test rotation by \pi/2 on 4D input.
    x \in \mathbb{R}^{B \times H \times S \times D}
    """
    # B=1, H=1, S=1, D=2
    # x = (1, 0) \equiv 1 + 0i
    x = torch.tensor([[[[1.0, 0.0]]]])

    # \theta = \pi / 2
    # freqs_cis \in \mathbb{R}^{1 \times 1 \times 1 \times 1 \times 2}, last dim = (cos, sin)
    theta = torch.tensor([[[[torch.pi / 2]]]])
    freqs_cis = _freqs_cis_from_theta(theta)

    # z' = z \cdot i = (1)(i) = 0 + 1i
    out = apply_rotary_emb(x, freqs_cis)

    # Expected: (0, 1)
    expected = torch.tensor([[[[0.0, 1.0]]]])

    # y \approx expected
    assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_rotation_formula_explicit():
    r"""
    Verify 4D input against explicit matrix rotation.
    x \in \mathbb{R}^{B \times H \times S \times D}
    """
    B, H, S, D = 2, 4, 8, 16
    x = torch.randn(B, H, S, D)

    # \theta \in \mathbb{R}^{B \times H \times S \times D/2}
    theta = torch.randn(B, H, S, D // 2)
    freqs_cis = _freqs_cis_from_theta(theta)

    # y_{model}
    out = apply_rotary_emb(x, freqs_cis)

    # y_{manual}
    # x_{pair} \in \mathbb{R}^{... \times D/2 \times 2}
    x_reshaped = x.view(B, H, S, D // 2, 2)
    x_r, x_i = x_reshaped[..., 0], x_reshaped[..., 1]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # \begin{cases} x'_r = x_r \cos\theta - x_i \sin\theta \\ x'_i = x_r \sin\theta + x_i \cos\theta \end{cases}
    out_r = x_r * cos_theta - x_i * sin_theta
    out_i = x_r * sin_theta + x_i * cos_theta

    # Flatten last 2 dims: (..., D/2, 2) \to (..., D)
    expected = torch.stack([out_r, out_i], dim=-1).flatten(3)

    # y_{model} \equiv y_{manual}
    assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_backward(dtype):
    r"""
    Check \nabla_x L \neq 0 using 4D input.
    x \in \mathbb{R}^{B \times H \times S \times D}
    """
    # x \in \mathbb{R}^{1 \times 1 \times 2 \times 4} (4D to satisfy flatten(3))
    x = torch.randn(1, 1, 2, 4, dtype=dtype, requires_grad=True)

    # \theta \in \mathbb{R}^{1 \times 1 \times 2 \times 2}
    theta = torch.randn(1, 1, 2, 2, dtype=dtype)
    freqs_cis = _freqs_cis_from_theta(theta)

    out = apply_rotary_emb(x, freqs_cis)

    # L = \sum y
    loss = out.sum()
    loss.backward()

    # \exists \frac{\partial L}{\partial x}
    assert x.grad is not None
    assert torch.norm(x.grad) > 0


def test_broadcasting_shapes():
    r"""
    Test broadcasting of freqs_cis.
    x \in \mathbb{R}^{B \times H \times S \times D}
    freqs \in \mathbb{R}^{1 \times 1 \times S \times D/2 \times 2}
    """
    # x: [Batch=2, Heads=4, Seq=8, Dim=10]
    x = torch.randn(2, 4, 8, 10)

    # \theta: Broadcast over Batch and Heads
    theta = torch.randn(1, 1, 8, 5)
    freqs_cis = _freqs_cis_from_theta(theta)

    out = apply_rotary_emb(x, freqs_cis)

    # Shape(y) == Shape(x)
    assert out.shape == x.shape
    assert out.shape == (2, 4, 8, 10)
