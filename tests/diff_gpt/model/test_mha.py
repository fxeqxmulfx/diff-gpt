"""Tests for causal multi-head self-attention, with and without the KV cache."""

import torch
from torch.testing import assert_close

from diff_gpt.model.kv_cache import KVCache
from diff_gpt.model.mha import MultiHeadAttention

# Chunk boundaries for test_chunked_prefill_inference: prefill [0, 2), then a
# chunk [2, 5) on top of the cached prefix; the last token is decoded alone.
PREFILL_END = 2
CHUNK_END = 5


def test_shape_and_gradients() -> None:
    r"""
    Check the (B, T, C) output shape and gradient flow through both projections.

    $x \in \mathbb{R}^{B \times T \times C} \implies
    y \in \mathbb{R}^{B \times T \times C}$
    $\nabla_{W_{attn}} L \neq 0, \quad \nabla_{W_{proj}} L \neq 0$
    """
    B, T, C, H = 2, 10, 32, 4

    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)

    x = torch.randn(B, T, C, requires_grad=True)

    # kv_cache = None (Training mode)
    y = model(x, kv_cache=None)

    # Check Shape
    assert y.shape == (B, T, C)

    # Check Gradients
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.norm(p.grad) > 0, name


def test_causal_masking_property() -> None:
    r"""
    Check causality: perturbing the last token changes only its output.

    Autoregressive property: $y_t = f(x_0, \dots, x_t)$
    $\frac{\partial y_t}{\partial x_{t+k}} = 0 \quad \forall k > 0$
    """
    B, T, C, H = 1, 5, 16, 2
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    model.eval()

    x = torch.randn(B, T, C)

    # 1. Forward original
    with torch.no_grad():
        y_orig = model(x, kv_cache=None)

    # 2. Perturb LAST token ($x_{T-1}$)
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 1.0

    with torch.no_grad():
        y_new = model(x_perturbed, kv_cache=None)

    # 3. Verify prefix stability ($t < T-1$)
    assert_close(y_orig[:, :-1, :], y_new[:, :-1, :])

    # 4. Verify causality breach (last token changed)
    assert not torch.allclose(y_orig[:, -1, :], y_new[:, -1, :])


def test_kv_cache_equivalence_step_by_step() -> None:
    r"""
    Check that step-by-step inference with a real KVCache matches the batch forward.

    $KV$ Cache mode ($T_q=1$) vs Standard mode ($T_q=T$)
    $y_{step}[t] \equiv y_{batch}[t]$
    """
    B, T, C, H = 1, 4, 16, 2
    head_dim = C // H

    # Important: layer_idx=0 matches num_layers=1 in KVCache to trigger auto-pos update
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    model.eval()

    x = torch.randn(B, T, C)

    # --- 1. Batch Forward ---
    with torch.no_grad():
        y_batch = model(x, kv_cache=None)

    # --- 2. Step-by-Step Forward ---
    # Init Real KVCache (num_layers=1 for unit test)
    kv = KVCache(batch_size=B, num_heads=H, seq_len=10, head_dim=head_dim, num_layers=1)

    outputs = []
    for t in range(T):
        # x_step \in \mathbb{R}^{B \times 1 \times C}
        x_step = x[:, t : t + 1, :]

        with torch.no_grad():
            # KVCache automatically increments pos
            # because layer_idx (0) == num_layers-1 (0)
            y_step = model(x_step, kv_cache=kv)
            outputs.append(y_step)

    y_sequential = torch.cat(outputs, dim=1)

    assert_close(y_sequential, y_batch, atol=1e-5, rtol=1e-5)


def test_chunked_prefill_inference() -> None:
    """
    Check that chunked inference matches a full pass and pos advances per chunk.

    The chunks go through all three masking branches.
    Scenario: prefill $t=0..1$ ($T_q = T_k$, causal), chunk $t=2..4$
    (cached prefix + causal within the chunk), decode $t=5$ ($T_q = 1$).
    """
    B, T, C, H = 1, 6, 16, 2
    head_dim = C // H

    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    model.eval()

    x = torch.randn(B, T, C)

    # --- 1. Full Pass ---
    with torch.no_grad():
        y_full = model(x, kv_cache=None)

    # --- 2. Chunked Pass ---
    # Real KVCache
    kv = KVCache(batch_size=B, num_heads=H, seq_len=10, head_dim=head_dim, num_layers=1)

    with torch.no_grad():
        # Chunk 1: Prefill (Tokens 0, 1). KVCache pos $0 \to 2$
        y_chunk1 = model(x[:, :PREFILL_END, :], kv_cache=kv)
        assert kv.get_pos() == PREFILL_END
        # Chunk 2: Tokens 2..4 against the cached prefix. KVCache pos $2 \to 5$
        y_chunk2 = model(x[:, PREFILL_END:CHUNK_END, :], kv_cache=kv)
        assert kv.get_pos() == CHUNK_END
        # Chunk 3: Decode (Token 5). KVCache pos $5 \to 6$
        y_chunk3 = model(x[:, CHUNK_END:T, :], kv_cache=kv)
        assert kv.get_pos() == T

    y_reconstructed = torch.cat([y_chunk1, y_chunk2, y_chunk3], dim=1)

    assert_close(y_reconstructed, y_full, atol=1e-5, rtol=1e-5)
