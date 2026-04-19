import torch
from torch.testing import assert_close

from diff_gpt.model.block import Block, block_attn_res
from diff_gpt.model.kv_cache import KVCache


def get_freqs_cis(seq_len: int, head_dim: int) -> torch.Tensor:
    """Real-valued (cos, sin) RoPE table; shape (1, T, 1, D/2, 2)."""
    dims = head_dim // 2
    theta = torch.randn(1, seq_len, 1, dims)
    return torch.stack([theta.cos(), theta.sin()], dim=-1)


def test_block_attn_res_shape_and_weights_sum_to_one():
    """
    block_attn_res returns a per-token softmax mixture of (blocks + partial).
    Shape is (B, T, D); |h| stays bounded by the per-token max across blocks.
    """
    B, T, D, N = 2, 5, 16, 3
    blocks = tuple(torch.randn(B, T, D) for _ in range(N))
    partial = torch.randn(B, T, D)
    proj = torch.nn.Linear(D, 1, bias=False)

    h = block_attn_res(blocks, partial, proj)
    assert h.shape == (B, T, D)

    stacked = torch.stack([*blocks, partial], dim=0)
    upper = stacked.abs().max(dim=0).values
    assert (h.abs() <= upper + 1e-5).all()


def test_block_forward_and_gradients():
    """
    Block.forward(blocks, partial) -> (blocks', partial'). Shape preserved;
    gradient flows into attn, mlp, and both AttnRes projections.
    """
    B, T, C, H = 2, 8, 32, 4
    head_dim = C // H

    model = Block(
        n_embd=C,
        n_head=H,
        swiglu_alpha=1.0,
        swiglu_limit=10.0,
        layer_idx=0,
        layers_per_block=2,
    )

    x0 = torch.randn(B, T, C, requires_grad=True)
    x1 = torch.randn(B, T, C, requires_grad=True)
    freqs = get_freqs_cis(T, head_dim)

    new_blocks, new_partial = model((x0,), x1, freqs_cis=freqs, kv_cache=None)

    assert new_partial.shape == (B, T, C)
    assert len(new_blocks) == 1
    assert new_blocks[0] is x0

    loss = new_partial.sum()
    loss.backward()

    assert x0.grad is not None and x1.grad is not None
    for p in model.sa.parameters():
        assert p.grad is not None and torch.norm(p.grad) > 0
    for p in model.ffwd.parameters():
        assert p.grad is not None and torch.norm(p.grad) > 0
    assert torch.norm(model.attn_res_proj.weight.grad) > 0
    assert torch.norm(model.mlp_res_proj.weight.grad) > 0


def test_block_commits_at_layer_boundary():
    """
    With layers_per_block=1, every Block.forward commits partial_block into
    the returned blocks tuple and restarts partial fresh from attn_out+mlp_out.
    """
    B, T, C, H = 1, 4, 16, 2
    head_dim = C // H

    model = Block(
        n_embd=C,
        n_head=H,
        swiglu_alpha=1.0,
        swiglu_limit=10.0,
        layer_idx=0,
        layers_per_block=1,
    )
    x = torch.randn(B, T, C)
    freqs = get_freqs_cis(T, head_dim)

    with torch.no_grad():
        new_blocks, _ = model((x,), x, freqs_cis=freqs, kv_cache=None)
    assert len(new_blocks) == 2
    assert new_blocks[0] is x
    assert torch.equal(new_blocks[1], x)


def test_block_causality_preservation():
    """
    Block still obeys causal attention over time even with AttnRes over depth.
    """
    B, T, C, H = 1, 5, 16, 2
    head_dim = C // H
    model = Block(
        n_embd=C,
        n_head=H,
        swiglu_alpha=1.0,
        swiglu_limit=10.0,
        layer_idx=0,
        layers_per_block=2,
    )
    model.eval()

    x = torch.randn(B, T, C)
    freqs = get_freqs_cis(T, head_dim)

    with torch.no_grad():
        _, y_orig = model((x,), x, freqs_cis=freqs, kv_cache=None)

    x_pert = x.clone()
    x_pert[:, -1, :] += 5.0
    with torch.no_grad():
        _, y_new = model((x_pert,), x_pert, freqs_cis=freqs, kv_cache=None)

    assert_close(y_orig[:, :-1, :], y_new[:, :-1, :])
    assert not torch.allclose(y_orig[:, -1, :], y_new[:, -1, :])


def test_block_kv_cache_step_consistency():
    """
    Incremental (single-token) vs batched Block forward must yield the same
    output partial_block, even with AttnRes active.
    """
    B, T, C, H = 1, 6, 16, 2
    head_dim = C // H
    model = Block(
        n_embd=C,
        n_head=H,
        swiglu_alpha=1.0,
        swiglu_limit=10.0,
        layer_idx=0,
        layers_per_block=2,
    )
    model.eval()

    x = torch.randn(B, T, C)
    freqs_full = get_freqs_cis(T, head_dim)

    with torch.no_grad():
        _, y_batch = model((x,), x, freqs_cis=freqs_full, kv_cache=None)

    kv = KVCache(
        batch_size=B, num_heads=H, seq_len=20, head_dim=head_dim, num_layers=1
    )
    outputs = []
    for t in range(T):
        x_step = x[:, t : t + 1, :]
        freqs_step = freqs_full[:, t : t + 1, :, :]
        with torch.no_grad():
            _, y_step = model(
                (x_step,), x_step, freqs_cis=freqs_step, kv_cache=kv
            )
            outputs.append(y_step)
    y_serial = torch.cat(outputs, dim=1)

    assert_close(y_batch, y_serial, atol=1e-5, rtol=1e-5)
