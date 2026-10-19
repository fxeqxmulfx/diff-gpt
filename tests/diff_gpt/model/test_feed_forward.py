import torch
from torch.testing import assert_close

from diff_gpt.model.feed_forward import FeedForward


def test_ffn_shape_and_gradients():
    r"""
    Verify dimensions and gradient flow.
    FFN maps $x \in \mathbb{R}^{B \times T \times C} \to y \in \mathbb{R}^{B \times T \times C}$.
    Expansion factor is internal (4x -> 1x).
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
    # net[0] is the expansion Linear layer (n -> 4n)
    assert model.net[0].weight.grad is not None
    assert torch.norm(model.net[0].weight.grad) > 0

    # net[2] is the projection Linear layer (4n -> n)
    assert model.net[2].weight.grad is not None
    assert torch.norm(model.net[2].weight.grad) > 0


def test_ffn_token_independence():
    r"""
    Verify point-wise property (Locality).
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
        y_orig[:, :-1, :], y_new[:, :-1, :], msg="FFN leaked info to previous tokens!"
    )

    # 4. Verify Change
    # The last token must change
    assert not torch.allclose(y_orig[:, -1, :], y_new[:, -1, :]), (
        "Perturbation did not affect the target token!"
    )


def test_ffn_batch_invariance():
    r"""
    Verify batch independence.
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
