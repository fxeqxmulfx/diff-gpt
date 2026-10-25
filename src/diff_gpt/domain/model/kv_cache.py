"""Per-layer key/value cache for incremental decoding."""

import torch


class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.

    Note that the .pos advances automatically after the last layer of the
    Transformer inserts.

    A cache can continue the positions of another without copying them
    (`share_prefix`): groups of its rows then share one copy of their
    prompt's keys and values, and the cache itself holds only the positions
    inserted after it.
    """

    __slots__ = (
        "kv_cache",
        "kv_shape",
        "pos",
        "prefix",
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
        self.pos = 0  # positions in this cache's own storage
        # Keys/values (L, 2, prompts, H, T, D) that groups of rows share
        # before their own positions (share_prefix), or None
        self.prefix: torch.Tensor | None = None

    def reset(self) -> None:
        """Rewind to the end of the shared prefix (0 without one), keeping storage."""
        self.pos = 0

    def get_pos(self) -> int:
        """Return the current position in time, i.e. how many steps are cached."""
        if self.prefix is None:
            return self.pos
        return self.prefix.size(4) + self.pos

    def _check_rows_of(self, other: KVCache) -> None:
        """Raise ValueError unless every row of `other` can lead rows of self."""
        # Extract dimensions explicitly
        self_layers, self_kv, self_batch, self_heads, _, self_head_dim = self.kv_shape
        # other's capacity is irrelevant: only its cached positions count
        other_layers, other_kv, other_batch, other_heads, _, other_head_dim = (
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

        # Batch size can be expanded: every row of other leads as many rows
        if self_batch % other_batch != 0:
            msg = (
                f"Batch size mismatch: {self_batch} vs {other_batch} "
                "(self must hold every row of other the same number of times)"
            )
            raise ValueError(msg)

    def prefill(self, other: KVCache) -> None:
        """
        Prefill given another KV cache, optionally expanding along the batch dim.

        Each row of `other` fills self_batch / other_batch consecutive rows,
        so a batch-1 prefill fills them all: a copy of each prompt's
        positions for every row of its samples. `share_prefix` continues
        them without the copies.
        """
        # 1) validate the shapes
        if self.kv_cache.size(0) != 0 or self.prefix is not None:
            msg = "Cannot prefill a non-empty KV cache"
            raise RuntimeError(msg)
        if other.kv_cache.size(0) == 0:
            msg = "Cannot prefill with a None KV cache"
            raise ValueError(msg)
        if other.prefix is not None:
            msg = "Cannot prefill from a KV cache that shares a prefix"
            raise ValueError(msg)
        self._check_rows_of(other)
        # Sequence length: self must hold every position cached in other (its
        # capacity may be larger, after it grew)
        self_seq = self.kv_shape[4]
        if self_seq < other.pos:
            msg = (
                f"Sequence length mismatch: capacity {self_seq} < "
                f"{other.pos} cached positions"
            )
            raise ValueError(msg)

        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over: row i of other fills rows i·r to i·r + r - 1,
        # r = self_batch / other_batch, broadcast from a view of other, so no
        # copy of the whole cache is made on the way
        other_batch = other.kv_shape[2]
        cached = other.kv_cache[..., : other.pos, :].unsqueeze(3)
        self.kv_cache[..., : other.pos, :].unflatten(2, (other_batch, -1)).copy_(cached)
        # 4) update the pos
        self.pos = other.pos

    def share_prefix(self, other: KVCache) -> None:
        """
        Continue the positions cached in `other` without copying them.

        Row i of `other` leads rows i·r to i·r + r - 1 of self, r = self_batch
        / other_batch, as after `prefill`, but stays one copy: attention reads
        it once for all of them (`prefix_kv`). Self holds only the positions
        inserted after it, so its capacity counts those alone. Self keeps a
        view of `other`'s keys and values, so `other` must not change after.
        """
        if self.kv_cache.size(0) != 0 or self.prefix is not None:
            msg = "Cannot share a prefix with a non-empty KV cache"
            raise RuntimeError(msg)
        if other.kv_cache.size(0) == 0:
            msg = "Cannot share the prefix of an empty KV cache"
            raise ValueError(msg)
        if other.prefix is not None:
            msg = "Cannot share the prefix of a KV cache that shares one"
            raise ValueError(msg)
        self._check_rows_of(other)
        self.prefix = other.kv_cache[..., : other.pos, :]

    def prefix_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Return the shared keys and values of layer `layer_idx`, or None.

        Each is (prompts, H, T, D): the T positions of each prompt, which
        batch / prompts consecutive rows of this cache continue.
        """
        if self.prefix is None:
            return None
        return self.prefix[layer_idx, 0], self.prefix[layer_idx, 1]

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
        With a shared prefix they hold the positions after it.
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
