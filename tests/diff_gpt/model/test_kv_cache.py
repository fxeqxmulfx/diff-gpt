# Copyright 2025 Mikhail Borisov
"""Tests for the KV cache: allocation, insertion, growth, prefill and reset."""

import torch
from torch.testing import assert_close

from diff_gpt.model.kv_cache import KVCache

# insert_kv grows the capacity to a multiple of this many positions.
GROWTH_MULTIPLE = 1024


def test_kv_cache_resize() -> None:
    """
    Check that a full cache grows on insert and keeps the tokens it already stored.

    The KV cache was not resized correctly, more information here:
    https://github.com/karpathy/nanochat/pull/186
    This test reproduces the issue and will be merged alongside the fix.
    """
    batch_size = 2
    num_heads = 3
    seq_len = 4
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
    )

    # Insert a single token with a distinct fill value to all layers
    def insert_token(token_idx: int) -> None:
        for layer_idx in range(num_layers):
            k = torch.full(
                (batch_size, num_heads, 1, head_dim),
                fill_value=float(token_idx),
                dtype=torch.float32,
            )
            v = torch.full(
                (batch_size, num_heads, 1, head_dim),
                fill_value=float(token_idx * 100),
                dtype=torch.float32,
            )
            kv_cache.insert_kv(layer_idx, k, v)

    # Insert 4 tokens (fills the initial seq_len=4)
    for i in range(4):
        insert_token(i)

    # Record the original state of the cache
    original_cache = kv_cache.kv_cache.clone()
    original_seq_len = original_cache.shape[4]

    # Insert the 5th token, which will trigger a resize
    insert_token(4)
    # Verify that the cache actually resized
    new_seq_len = kv_cache.kv_cache.shape[4]
    assert new_seq_len > original_seq_len, (
        f"Cache did not resize: original seq_len={original_seq_len}, "
        f"new seq_len={new_seq_len}"
    )

    # Verify that the original 4 tokens are still intact after resize
    for layer_idx in range(num_layers):
        for token_idx in range(4):
            # Check that resized cache matches expected values
            expected_k = float(token_idx)
            expected_v = float(token_idx * 100)
            actual_k = kv_cache.kv_cache[layer_idx, 0, :, :, token_idx, :]
            actual_v = kv_cache.kv_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == expected_k).all(), (
                f"Layer {layer_idx}, token {token_idx}: key corrupted, "
                f"expected {expected_k}"
            )
            assert (actual_v == expected_v).all(), (
                f"Layer {layer_idx}, token {token_idx}: value corrupted, "
                f"expected {expected_v}"
            )
            # And that the original cache matches resized cache
            original_k = original_cache[layer_idx, 0, :, :, token_idx, :]
            original_v = original_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == original_k).all(), (
                f"Layer {layer_idx}, token {token_idx}: key doesn't match original"
            )
            assert (actual_v == original_v).all(), (
                f"Layer {layer_idx}, token {token_idx}: value doesn't match original"
            )


def test_initialization_state() -> None:
    r"""
    Check that a new cache is empty at pos 0 and records shape (L, 2, B, H, T, D).

    Initial state $S_0$: pos = 0, Cache = \emptyset
    """
    B, H, T, D, L = 2, 4, 128, 64, 3
    cache = KVCache(B, H, T, D, L)

    assert cache.get_pos() == 0
    assert cache.kv_cache.size(0) == 0
    # \text{Shape} = (L, 2, B, H, T, D)
    assert cache.kv_shape == (L, 2, B, H, T, D)


def test_insert_kv_lazy_init_and_pos_update() -> None:
    r"""
    Check lazy allocation, the last-layer-only pos update and the returned keys.

    $pos_{new} = pos_{old} + \Delta t \iff layer == L-1$
    """
    B, H, T, D, L = 1, 2, 10, 8, 2
    cache = KVCache(B, H, T, D, L)

    # Input: k, v \in \mathbb{R}^{B \times H \times 1 \times D}
    k = torch.randn(B, H, 1, D)
    v = torch.randn(B, H, 1, D)

    # --- Layer 0 ---
    _k_out, _v_out = cache.insert_kv(0, k, v)

    # Cache allocated?
    assert cache.kv_cache is not None
    # $pos$ should NOT update yet (wait for last layer)
    assert cache.get_pos() == 0

    # --- Layer 1 (Last) ---
    k_out_2, _v_out_2 = cache.insert_kv(1, k, v)

    # $pos$ updated now
    assert cache.get_pos() == 1

    # Data consistency: Output view should match input for t=0
    # k_{out} \in \mathbb{R}^{B \times H \times 1 \times D}
    assert_close(k_out_2[..., 0:1, :], k)


def test_insert_sequence_chunk() -> None:
    r"""
    Check that inserting a chunk ($T_{add} > 1$) advances pos and returns the chunk.

    $t_0 = 0, t_1 = 5 \implies pos \to 5$
    """
    B, H, T_init, D, L = 1, 1, 20, 4, 1
    cache = KVCache(B, H, T_init, D, L)

    # k \in \mathbb{R}^{1 \times 1 \times 5 \times 4}
    seq_len_add = 5
    k = torch.randn(B, H, seq_len_add, D)
    v = torch.randn(B, H, seq_len_add, D)

    out_k, _out_v = cache.insert_kv(0, k, v)

    # $pos = 5$
    assert cache.get_pos() == seq_len_add
    # Output view shape: $... \times 5 \times D$
    assert out_k.shape[-2] == seq_len_add
    assert_close(out_k, k)


def test_dynamic_growth_rounding() -> None:
    r"""
    Check that an overflowing insert grows the capacity to a multiple of 1024.

    Condition: $t_1 > T_{capacity}$
    New Size: $\lceil \frac{t_1 + 1024}{1024} \rceil \times 1024$
    """
    # Initialize small capacity: T=10
    B, H, T_init, D, L = 1, 1, 10, 4, 1
    cache = KVCache(B, H, T_init, D, L)

    # First insert to init cache
    k0 = torch.randn(B, H, 1, D)
    cache.insert_kv(0, k0, k0)  # pos -> 1, Capacity = 10

    assert cache.kv_cache.shape[4] == T_init

    # Insert big chunk to force resize
    # Current pos=1. Add 20. New pos=21. 21 > 10.
    # Needed = 21 + 1024 = 1045. Round up to 1024 multiple -> 2048.
    k_big = torch.randn(B, H, 20, D)
    cache.insert_kv(0, k_big, k_big)

    current_capacity = cache.kv_cache.shape[4]

    # Verify growth
    assert current_capacity >= GROWTH_MULTIPLE
    assert current_capacity % GROWTH_MULTIPLE == 0
    assert cache.get_pos() == k0.shape[-2] + k_big.shape[-2]


def test_prefill_broadcasting() -> None:
    r"""
    Check that prefill copies pos and broadcasts a batch-1 cache to every batch row.

    Batch broadcasting: $B_{src}=1 \to B_{dst}=N$
    $KV_{dst}[i] \equiv KV_{src}[0] \quad \forall i$
    """
    # Source: Batch=1, Seq=5
    src_cache = KVCache(1, 2, 10, 4, 1)
    k = torch.randn(1, 2, 5, 4)
    src_cache.insert_kv(0, k, k)  # pos=5

    # Dest: Batch=3, Seq=20
    dst_cache = KVCache(3, 2, 20, 4, 1)

    dst_cache.prefill(src_cache)

    # Check pos transfer
    assert dst_cache.get_pos() == k.shape[-2]

    # Check broadcasting
    # dst memory: [L, 2, B=3, H, T, D]
    # Check batch index 0, 1, 2 all contain the source data
    for b in range(3):
        stored_k = dst_cache.kv_cache[0, 0, b, :, :5, :]
        assert_close(stored_k, k.squeeze(0))  # remove source batch dim


def test_reset() -> None:
    r"""
    Check that reset rewinds pos to 0, so the next insert starts at t=0.

    $pos \to 0$
    """
    cache = KVCache(1, 1, 10, 4, 1)
    k = torch.randn(1, 1, 5, 4)
    cache.insert_kv(0, k, k)

    assert cache.get_pos() == k.shape[-2]

    cache.reset()
    assert cache.get_pos() == 0

    # Next insert at t=0
    k_new = torch.ones(1, 1, 1, 4)
    out_k, _ = cache.insert_kv(0, k_new, k_new)

    # Compare full tensors to avoid rank mismatch
    # out_k \in \mathbb{R}^{1 \times 1 \times 1 \times 4}
    assert_close(out_k, k_new)
