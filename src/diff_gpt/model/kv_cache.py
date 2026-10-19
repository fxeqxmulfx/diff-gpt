# Copyright 2025 Mikhail Borisov
"""Per-layer key/value cache for incremental decoding."""

import torch


class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.

    Note that the .pos advances automatically after the last layer of the
    Transformer inserts.
    """

    __slots__ = (
        "kv_cache",
        "kv_shape",
        "pos",
    )

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
    ) -> None:
        """
        Record the cache shape; the storage is allocated lazily.

        Allocation happens on the first `insert_kv` or `prefill`, which supply
        the dtype and device. `seq_len` is the initial capacity in positions;
        `insert_kv` grows the cache past it when needed.
        """
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the
        # Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = torch.zeros(0)
        self.pos = 0  # current position in time in the cache

    def reset(self) -> None:
        """Rewind the position to 0, keeping the allocated storage."""
        self.pos = 0

    def get_pos(self) -> int:
        """Return the current position in time, i.e. how many steps are cached."""
        return self.pos

    def prefill(self, other: KVCache) -> None:
        """
        Prefill given another KV cache, optionally expanding along the batch dim.

        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        if self.kv_cache.size(0) != 0:
            msg = "Cannot prefill a non-empty KV cache"
            raise RuntimeError(msg)
        if other.kv_cache.size(0) == 0:
            msg = "Cannot prefill with a None KV cache"
            raise ValueError(msg)

        # Extract dimensions explicitly
        self_layers, self_kv, self_batch, self_heads, self_seq, self_head_dim = (
            self.kv_shape
        )
        other_layers, other_kv, other_batch, other_heads, other_seq, other_head_dim = (
            other.kv_shape
        )

        # Validate dimensions
        if self_layers != other_layers:
            msg = f"Layer count mismatch: {self_layers} != {other_layers}"
            raise ValueError(msg)
        if self_kv != other_kv:
            msg = f"K/V dimension mismatch: {self_kv} != {other_kv}"
            raise ValueError(msg)
        if self_heads != other_heads:
            msg = f"Head count mismatch: {self_heads} != {other_heads}"
            raise ValueError(msg)
        if self_head_dim != other_head_dim:
            msg = f"Head dim mismatch: {self_head_dim} != {other_head_dim}"
            raise ValueError(msg)

        # Batch size can be expanded (other can be 1, self can be larger)
        if other_batch not in (self_batch, 1):
            msg = (
                f"Batch size mismatch: {self_batch} vs {other_batch} "
                "(other must be 1 or equal)"
            )
            raise ValueError(msg)

        # Sequence length: self must be longer than other
        if self_seq < other_seq:
            msg = f"Sequence length mismatch: {self_seq} < {other_seq}"
            raise ValueError(msg)

        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, : other.pos, :] = other.kv_cache[..., : other.pos, :]
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Insert `k`, `v` of layer `layer_idx` and return its keys/values so far.

        `k` and `v` are (B, H, T_add, D). The returned tensors are views of the
        cache up to the new position; the cache grows when it runs out of room.
        """
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache.size(0) == 0:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        _B, _H, T_add, _D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024  # as much as we need plus buffer of 1024
            t_needed = (
                t_needed + 1023
            ) & ~1023  # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(
                additional_shape,
                dtype=k.dtype,
                device=k.device,
            )
            self.kv_cache = torch.cat(
                [self.kv_cache, additional_cache],
                dim=4,
            ).contiguous()
            self.kv_shape = self.kv_cache.shape
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view
