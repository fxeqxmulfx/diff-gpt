import torch
from torch.testing import assert_close

from diff_gpt.model.mha import MultiHeadAttention
from diff_gpt.model.kv_cache import KVCache


def get_freqs_cis(seq_len: int, head_dim: int, dtype=torch.float32) -> torch.Tensor:
    r"""
    Generate the (cos, sin) RoPE table consumed by `apply_rotary_emb`.
    Shape: (1, T, 1, D/2, 2) where the last dim packs (cos θ, sin θ) —
    real-valued, broadcasts over batch and heads.
    """
    dims = head_dim // 2
    theta = torch.randn(1, seq_len, 1, dims)
    return torch.stack([theta.cos(), theta.sin()], dim=-1)


def test_shape_and_gating_gradients():
    r"""
    Verify dimensions and gradient flow through Gate.
    $x \in \mathbb{R}^{B \times T \times C} \implies y \in \mathbb{R}^{B \times T \times C}$
    $\nabla_{W_g} L \neq \emptyset$
    """
    B, T, C, H = 2, 10, 32, 4
    head_dim = C // H

    # layer_idx=0
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)

    x = torch.randn(B, T, C, requires_grad=True)
    freqs_cis = get_freqs_cis(T, head_dim)

    # kv_cache = None (Training mode)
    y = model(x, freqs_cis, kv_cache=None)

    # Check Shape
    assert y.shape == (B, T, C)

    # Check Gate Gradients
    loss = y.sum()
    loss.backward()

    assert model.c_g.weight.grad is not None
    assert torch.norm(model.c_g.weight.grad) > 0


def test_causal_masking_property():
    r"""
    Verify autoregressive property.
    $y_t = f(x_0, \dots, x_t)$
    $\frac{\partial y_t}{\partial x_{t+k}} = 0 \quad \forall k > 0$
    """
    B, T, C, H = 1, 5, 16, 2
    head_dim = C // H
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    model.eval()

    x = torch.randn(B, T, C)
    freqs_cis = get_freqs_cis(T, head_dim)

    # 1. Forward original
    with torch.no_grad():
        y_orig = model(x, freqs_cis, kv_cache=None)

    # 2. Perturb LAST token ($x_{T-1}$)
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 1.0

    with torch.no_grad():
        y_new = model(x_perturbed, freqs_cis, kv_cache=None)

    # 3. Verify prefix stability ($t < T-1$)
    assert_close(y_orig[:, :-1, :], y_new[:, :-1, :])

    # 4. Verify causality breach (last token changed)
    assert not torch.allclose(y_orig[:, -1, :], y_new[:, -1, :])


def test_kv_cache_equivalence_step_by_step():
    r"""
    Verify Real KVCache step-by-step inference.
    $KV$ Cache mode ($T_q=1$) vs Standard mode ($T_q=T$)
    $y_{step}[t] \equiv y_{batch}[t]$
    """
    B, T, C, H = 1, 4, 16, 2
    head_dim = C // H

    # Important: layer_idx=0 matches num_layers=1 in KVCache to trigger auto-pos update
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    model.eval()

    x = torch.randn(B, T, C)
    freqs_full = get_freqs_cis(T, head_dim)

    # --- 1. Batch Forward ---
    with torch.no_grad():
        y_batch = model(x, freqs_full, kv_cache=None)

    # --- 2. Step-by-Step Forward ---
    # Init Real KVCache (num_layers=1 for unit test)
    kv = KVCache(batch_size=B, num_heads=H, seq_len=10, head_dim=head_dim, num_layers=1)

    outputs = []
    for t in range(T):
        # x_step \in \mathbb{R}^{B \times 1 \times C}
        x_step = x[:, t : t + 1, :]

        # freqs_step \in \mathbb{R}^{1 \times 1 \times 1 \times D/2}
        freqs_step = freqs_full[:, t : t + 1, :, :]

        with torch.no_grad():
            # KVCache automatically increments pos because layer_idx (0) == num_layers-1 (0)
            y_step = model(x_step, freqs_step, kv_cache=kv)
            outputs.append(y_step)

    y_sequential = torch.cat(outputs, dim=1)

    assert_close(y_sequential, y_batch, atol=1e-5, rtol=1e-5)


def test_chunked_prefill_inference():
    r"""
    Verify chunked inference (Prefill + Generation).
    Scenario: Prefill $t=0..1$, Generate $t=2$.
    """
    B, T, C, H = 1, 3, 16, 2
    head_dim = C // H

    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    model.eval()

    x = torch.randn(B, T, C)
    freqs_full = get_freqs_cis(T, head_dim)

    # --- 1. Full Pass ---
    with torch.no_grad():
        y_full = model(x, freqs_full, None)

    # --- 2. Chunked Pass ---
    # Real KVCache
    kv = KVCache(batch_size=B, num_heads=H, seq_len=10, head_dim=head_dim, num_layers=1)

    # Chunk 1: Prefill (Tokens 0, 1). KVCache pos $0 \to 2$
    with torch.no_grad():
        y_chunk1 = model(x[:, :2, :], freqs_full[:, :2, :, :], kv)

    assert kv.get_pos() == 2

    # Chunk 2: Decode (Token 2). KVCache pos $2 \to 3$
    with torch.no_grad():
        y_chunk2 = model(x[:, 2:3, :], freqs_full[:, 2:3, :, :], kv)

    assert kv.get_pos() == 3

    y_reconstructed = torch.cat([y_chunk1, y_chunk2], dim=1)

    assert_close(y_reconstructed, y_full, atol=1e-5, rtol=1e-5)
