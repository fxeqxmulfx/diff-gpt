"""Tests for the transformer Block."""

import torch
from torch.testing import assert_close

from diff_gpt.model.block import Block
from diff_gpt.model.kv_cache import KVCache
from diff_gpt.model.rope import rope_tables


def test_block_forward_and_gradients() -> None:
    """
    Check that Block maps (B, T, C) -> (B, T, C) and backpropagates fully.

    The gradient reaches the input and every parameter of attention and the
    MLP; the norms have no parameters.
    """
    B, T, C, H = 2, 8, 32, 4
    model = Block(n_embd=C, n_head=H, layer_idx=0)
    cos, sin = rope_tables(C // H, T)

    x = torch.randn(B, T, C, requires_grad=True)
    y = model(x, cos, sin, kv_cache=None)

    assert y.shape == (B, T, C)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.norm(p.grad) > 0, name


def test_block_is_identity_when_residual_branches_are_zeroed() -> None:
    """
    Check that the block returns its input unchanged when both branches are zeroed.

    Peri-RMSNorm: the norms sit inside the branches, so zeroing both
    output projections leaves only the identity path.
    """
    B, T, C, H = 2, 5, 16, 2
    model = Block(n_embd=C, n_head=H, layer_idx=0)
    with torch.no_grad():
        for proj in (model.sa.c_proj, model.ffwd.c_proj):
            proj.weight.zero_()
    cos, sin = rope_tables(C // H, T)

    x = torch.randn(B, T, C)
    with torch.no_grad():
        y = model(x, cos, sin, kv_cache=None)

    assert torch.equal(y, x)


def test_block_causality_preservation() -> None:
    """Perturbing the last token must leave every earlier output unchanged."""
    B, T, C, H = 1, 5, 16, 2
    model = Block(n_embd=C, n_head=H, layer_idx=0)
    model.eval()
    cos, sin = rope_tables(C // H, T)

    x = torch.randn(B, T, C)

    with torch.no_grad():
        y_orig = model(x, cos, sin, kv_cache=None)

    x_pert = x.clone()
    x_pert[:, -1, :] += 5.0
    with torch.no_grad():
        y_new = model(x_pert, cos, sin, kv_cache=None)

    assert_close(y_orig[:, :-1, :], y_new[:, :-1, :])
    assert not torch.allclose(y_orig[:, -1, :], y_new[:, -1, :])


def test_block_kv_cache_step_consistency() -> None:
    """Check that incremental (single-token) and batched Block forward match."""
    torch.manual_seed(0)
    B, T, C, H = 1, 6, 16, 2
    head_dim = C // H
    model = Block(n_embd=C, n_head=H, layer_idx=0)
    model.eval()
    cos, sin = rope_tables(head_dim, T)

    x = torch.randn(B, T, C)

    with torch.no_grad():
        y_batch = model(x, cos, sin, kv_cache=None)

    kv = KVCache(batch_size=B, num_heads=H, seq_len=20, head_dim=head_dim, num_layers=1)
    outputs = []
    for t in range(T):
        x_step = x[:, t : t + 1, :]
        with torch.no_grad():
            y_step = model(x_step, cos[t : t + 1], sin[t : t + 1], kv_cache=kv)
            outputs.append(y_step)
    y_serial = torch.cat(outputs, dim=1)

    assert_close(y_batch, y_serial, atol=1e-5, rtol=1e-5)
