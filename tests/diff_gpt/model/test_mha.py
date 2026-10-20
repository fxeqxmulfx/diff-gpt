"""Tests for causal multi-head self-attention, with and without the KV cache."""

import math

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.model.kv_cache import KVCache
from diff_gpt.model.mha import MultiHeadAttention
from diff_gpt.model.rope import apply_rope, rope_tables

# Chunk boundaries for test_chunked_prefill_inference: prefill [0, 2), then a
# chunk [2, 5) on top of the cached prefix; the last token is decoded alone.
PREFILL_END = 2
CHUNK_END = 5
# Norm a head's XSA output stays above once a token has context to attend to.
MIN_CONTEXT_NORM = 1e-3


def test_shape_and_gradients() -> None:
    r"""
    Check the (B, T, C) output shape and gradient flow to every parameter.

    $x \in \mathbb{R}^{B \times T \times C} \implies
    y \in \mathbb{R}^{B \times T \times C}$
    $\nabla_{W_{attn}} L \neq 0, \quad \nabla_{W_{proj}} L \neq 0$.
    """
    B, T, C, H = 2, 10, 32, 4

    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    cos, sin = rope_tables(C // H, T)

    x = torch.randn(B, T, C, requires_grad=True)

    # kv_cache = None (Training mode)
    y = model(x, cos, sin, kv_cache=None)

    # Check Shape
    assert y.shape == (B, T, C)

    # Check Gradients
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.norm(p.grad) > 0, name


def test_xsa_output_is_orthogonal_to_the_tokens_own_value() -> None:
    """
    Check XSA: each head's output is orthogonal to the token's own value.

    With an identity output projection the module returns the heads' outputs
    side by side; head by head, their dot products with the values vanish,
    while past the first token the context still leaves an output.
    """
    torch.manual_seed(0)
    B, T, C, H = 2, 7, 16, 2
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0).double()
    with torch.no_grad():
        model.c_proj.weight.copy_(torch.eye(C))
    cos, sin = (table.double() for table in rope_tables(C // H, T))
    x = torch.randn(B, T, C, dtype=torch.float64)

    with torch.no_grad():
        z = model(x, cos, sin, kv_cache=None).view(B, T, H, C // H)
        v = model.c_attn(x)[..., 2 * C :].view(B, T, H, C // H)

    zeros = torch.zeros(B, T, H, dtype=torch.float64)
    assert_close((z * v).sum(dim=-1), zeros, atol=1e-12, rtol=0)
    assert (z[:, 1:].norm(dim=-1) > MIN_CONTEXT_NORM).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_xsa_first_position_outputs_zero(dtype: torch.dtype) -> None:
    """
    The first token attends only to itself, so XSA leaves it no output.

    Its attention output is its own value, all of which lies along that value.
    """
    torch.manual_seed(0)
    B, T, C, H = 2, 5, 16, 2
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0).to(dtype)
    cos, sin = (table.to(dtype) for table in rope_tables(C // H, T))
    x = torch.randn(B, T, C, dtype=dtype)

    with torch.no_grad():
        y = model(x, cos, sin, kv_cache=None)

    assert torch.count_nonzero(y[:, 0]) == 0


def test_without_rope_matches_identity_rotation_and_ignores_positions() -> None:
    """Disabling RoPE equals identity rotations, regardless of supplied positions."""
    torch.manual_seed(0)
    B, T, C, H = 2, 7, 16, 2
    positional = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    content = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0, use_rope=False)
    content.load_state_dict(positional.state_dict())
    x = torch.randn(B, T, C)
    cos, sin = rope_tables(C // H, T + 3)
    identity_cos = torch.ones_like(cos[:T])
    identity_sin = torch.zeros_like(sin[:T])
    with torch.no_grad():
        expected = positional(x, identity_cos, identity_sin, kv_cache=None)
        actual = content(x, cos[:T], sin[:T], kv_cache=None)
        shifted = content(x, cos[3:], sin[3:], kv_cache=None)
    assert_close(actual, expected, atol=0, rtol=0)
    assert_close(shifted, expected, atol=0, rtol=0)


def test_rejects_odd_head_size() -> None:
    """RoPE rotates channel pairs, so a head of odd size is rejected."""
    with pytest.raises(ValueError, match="must be even"):
        MultiHeadAttention(n_embd=12, n_head=4, layer_idx=0)


@pytest.mark.parametrize("head_dim", [4, 8, 16])
def test_qk_norm_logits_are_scaled_cosines(head_dim: int) -> None:
    r"""
    With QK-norm, logit scale follows from head size without a temperature parameter.

    $\ell_{ts} = \sqrt{d} \cos(R_t q_t, R_s k_s)$: causal softmax, the values,
    XSA and the output projection rebuild the module output from it.
    """
    torch.manual_seed(0)
    B, T, H = 2, 7, 2
    C = H * head_dim
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    model = model.double()
    cos, sin = (table.double() for table in rope_tables(C // H, T))
    # The float32 tables, renormalized in float64, rotate without changing
    # norms; then normalizing before RoPE, as the module does, or after agree
    radius = torch.hypot(cos, sin)
    cos, sin = cos / radius, sin / radius
    x = torch.randn(B, T, C, dtype=torch.float64)

    with torch.no_grad():
        y = model(x, cos, sin, kv_cache=None)
        q, k, v = (
            t.view(B, T, H, C // H).transpose(1, 2)
            for t in model.c_attn(x).split(C, dim=2)
        )
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cosines = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        p = (math.sqrt(head_dim) * cosines).masked_fill(~causal, -torch.inf).softmax(-1)
        z = p @ v
        unit = F.normalize(v, dim=-1)
        z = z - (z * unit).sum(dim=-1, keepdim=True) * unit
        rebuilt = model.c_proj(z.transpose(1, 2).reshape(B, T, C))

    assert_close(y, rebuilt, atol=1e-12, rtol=0)


def test_qk_norm_ignores_the_scale_of_query_and_key_weights() -> None:
    """
    QK-norm makes attention blind to the norm of the query and key weights.

    Shrinking them, as weight decay does, leaves the output unchanged
    up to the numerical epsilon in RMSNorm.
    """
    torch.manual_seed(0)
    B, T, C, H = 2, 7, 16, 2
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0).double()
    cos, sin = (table.double() for table in rope_tables(C // H, T))
    x = torch.randn(B, T, C, dtype=torch.float64)

    with torch.no_grad():
        y = model(x, cos, sin, kv_cache=None)
        model.c_attn.weight[: 2 * C] *= 0.1
        y_shrunk = model(x, cos, sin, kv_cache=None)

    assert_close(y, y_shrunk, atol=1e-12, rtol=0)


def test_causal_masking_property() -> None:
    r"""
    Check causality: perturbing the last token changes only its output.

    Autoregressive property: $y_t = f(x_0, \dots, x_t)$
    $\frac{\partial y_t}{\partial x_{t+k}} = 0 \quad \forall k > 0$.
    """
    B, T, C, H = 1, 5, 16, 2
    model = MultiHeadAttention(n_embd=C, n_head=H, layer_idx=0)
    model.eval()
    cos, sin = rope_tables(C // H, T)

    x = torch.randn(B, T, C)

    # 1. Forward original
    with torch.no_grad():
        y_orig = model(x, cos, sin, kv_cache=None)

    # 2. Perturb LAST token ($x_{T-1}$)
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 1.0

    with torch.no_grad():
        y_new = model(x_perturbed, cos, sin, kv_cache=None)

    # 3. Verify prefix stability ($t < T-1$)
    assert_close(y_orig[:, :-1, :], y_new[:, :-1, :])

    # 4. Verify causality breach (last token changed)
    assert not torch.allclose(y_orig[:, -1, :], y_new[:, -1, :])


def test_kv_cache_equivalence_step_by_step() -> None:
    r"""
    Check that step-by-step inference with a real KVCache matches the batch forward.

    $KV$ Cache mode ($T_q=1$) vs Standard mode ($T_q=T$)
    $y_{step}[t] \equiv y_{batch}[t]$; step $t$ takes the RoPE row of position $t$.
    """
    B, T, C, H = 1, 4, 16, 2
    head_dim = C // H

    # Important: layer_idx=0 matches num_layers=1 in KVCache to trigger auto-pos update
    model = MultiHeadAttention(
        n_embd=C,
        n_head=H,
        layer_idx=0,
    )
    model.eval()
    cos, sin = rope_tables(head_dim, T)

    x = torch.randn(B, T, C)

    # --- 1. Batch Forward ---
    with torch.no_grad():
        y_batch = model(x, cos, sin, kv_cache=None)

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
            y_step = model(x_step, cos[t : t + 1], sin[t : t + 1], kv_cache=kv)
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

    model = MultiHeadAttention(
        n_embd=C,
        n_head=H,
        layer_idx=0,
    )
    model.eval()
    cos, sin = rope_tables(head_dim, T)

    x = torch.randn(B, T, C)

    # --- 1. Full Pass ---
    with torch.no_grad():
        y_full = model(x, cos, sin, kv_cache=None)

    # --- 2. Chunked Pass ---
    # Real KVCache
    kv = KVCache(batch_size=B, num_heads=H, seq_len=10, head_dim=head_dim, num_layers=1)

    with torch.no_grad():
        # Chunk 1: Prefill (Tokens 0, 1). KVCache pos $0 \to 2$
        p = slice(0, PREFILL_END)
        y_chunk1 = model(x[:, p, :], cos[p], sin[p], kv_cache=kv)
        assert kv.get_pos() == PREFILL_END
        # Chunk 2: Tokens 2..4 against the cached prefix. KVCache pos $2 \to 5$
        c = slice(PREFILL_END, CHUNK_END)
        y_chunk2 = model(x[:, c, :], cos[c], sin[c], kv_cache=kv)
        assert kv.get_pos() == CHUNK_END
        # Chunk 3: Decode (Token 5). KVCache pos $5 \to 6$
        d = slice(CHUNK_END, T)
        y_chunk3 = model(x[:, d, :], cos[d], sin[d], kv_cache=kv)
        assert kv.get_pos() == T

    y_reconstructed = torch.cat([y_chunk1, y_chunk2, y_chunk3], dim=1)

    assert_close(y_reconstructed, y_full, atol=1e-5, rtol=1e-5)
