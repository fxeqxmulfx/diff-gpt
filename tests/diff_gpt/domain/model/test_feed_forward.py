"""Tests for the position-wise FeedForward network."""

import pytest
import torch
from torch.testing import assert_close

from diff_gpt.domain.model.feed_forward import FeedForward


def test_ffn_shape_and_gradients() -> None:
    r"""
    Check the (B, T, C) output shape and gradient flow through both Linear layers.

    FFN maps $x \in \mathbb{R}^{B \times T \times C} \to
    y \in \mathbb{R}^{B \times T \times C}$.
    The hidden size is internal (about 8/3 x).
    """
    B, T, C = 2, 10, 32

    model = FeedForward(n_embd=C)

    x = torch.randn(B, T, C, requires_grad=True)

    # Forward
    y = model(x)

    # 1. Check Shape Preservation
    assert y.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {y.shape}"

    # 2. Check Gradient Flow
    loss = y.sum()
    loss.backward()

    # Check gradients on input
    assert x.grad is not None

    # Check gradients on weights
    # c_fc holds the gate and value projections (n -> 2 * hidden)
    assert model.c_fc.weight.grad is not None
    assert torch.norm(model.c_fc.weight.grad) > 0

    # c_proj is the output projection (hidden -> n)
    assert model.c_proj.weight.grad is not None
    assert torch.norm(model.c_proj.weight.grad) > 0


def test_ffn_token_independence() -> None:
    """
    Check the point-wise property: perturbing the last token changes only its output.

    Unlike Attention, FFN processes each token independently.
    $FFN(x)_t$ depends ONLY on $x_t$, not $x_{t-1}$ or $x_{t+1}$.
    """
    B, T, C = 1, 5, 16
    model = FeedForward(n_embd=C)
    model.eval()

    x = torch.randn(B, T, C)

    # 1. Original Forward
    with torch.no_grad():
        y_orig = model(x)

    # 2. Perturb the LAST token only
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 2.0  # Significant change

    with torch.no_grad():
        y_new = model(x_perturbed)

    # 3. Verify Isolation
    # All tokens except the last one must remain EXACTLY the same
    # (Testing for strictly 0 interaction between time steps)
    assert_close(
        y_orig[:, :-1, :],
        y_new[:, :-1, :],
        msg="FFN leaked info to previous tokens!",
    )

    # 4. Verify Change
    # The last token must change
    assert not torch.allclose(y_orig[:, -1, :], y_new[:, -1, :]), (
        "Perturbation did not affect the target token!"
    )


def test_ffn_batch_invariance() -> None:
    """
    Check batch independence: batched outputs match per-sample outputs.

    Processing a sample alone vs in a batch should yield identical results.
    """
    T, C = 5, 16
    model = FeedForward(n_embd=C)
    model.eval()

    # Create two different samples
    x1 = torch.randn(1, T, C)
    x2 = torch.randn(1, T, C)

    # Process separately
    with torch.no_grad():
        y1_ref = model(x1)
        y2_ref = model(x2)

    # Process as a batch
    x_batch = torch.cat([x1, x2], dim=0)  # (2, T, C)
    with torch.no_grad():
        y_batch = model(x_batch)

    # Check consistency
    assert_close(y_batch[0:1], y1_ref)
    assert_close(y_batch[1:2], y2_ref)


def test_ffn_is_the_swiglu_of_the_paper() -> None:
    r"""
    Check FFN_SwiGLU(x) = (Swish_1(x W) ⊗ x V) W_2, where Swish_1(a) = a · sigmoid(a).

    W is the first half of c_fc (the gate), V the second half and W_2 is
    c_proj (Shazeer, "GLU Variants Improve Transformer", eq. 6).
    """
    torch.manual_seed(0)
    C = 16
    model = FeedForward(n_embd=C).double()
    x = torch.randn(2, 5, C, dtype=torch.float64)

    with torch.no_grad():
        hidden = model.c_proj.in_features
        w, v = model.c_fc.weight[:hidden].T, model.c_fc.weight[hidden:].T
        a = x @ w
        expected = (a * torch.sigmoid(a) * (x @ v)) @ model.c_proj.weight.T
        assert_close(model(x), expected)


@pytest.mark.parametrize("n_embd", [16, 32, 64, 96])
def test_ffn_hidden_size_keeps_the_parameters_of_a_4x_mlp(n_embd: int) -> None:
    """
    Check the hidden size: 2/3 of 4 * n_embd, rounded to the nearest integer.

    The three matrices then hold the 8 * n_embd² parameters of the two
    matrices of a 4x MLP, give or take n_embd.
    """
    model = FeedForward(n_embd=n_embd)
    hidden = model.c_proj.in_features
    n_params = sum(p.numel() for p in model.parameters())

    assert abs(3 * hidden - 8 * n_embd) <= 1
    assert n_params == 3 * n_embd * hidden
    assert abs(n_params - 8 * n_embd**2) <= n_embd
