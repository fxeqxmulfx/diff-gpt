"""Causal multi-head self-attention with an optional KV cache."""

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from diff_gpt.model.kv_cache import KVCache


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_idx: int,
    ) -> None:
        """
        Create the fused query/key/value and output projections, without biases.

        `n_embd` must be divisible by `n_head`. `layer_idx` is the index of the
        owning block; it selects this layer's slot in the KV cache.
        """
        super().__init__()
        if n_embd % n_head != 0:
            msg = f"n_embd {n_embd} must be divisible by n_head {n_head}"
            raise ValueError(msg)
        self.n_embd = n_embd
        self.n_head = n_head
        self.layer_idx = layer_idx
        self.head_dim = self.n_embd // self.n_head
        # Query, key and value projections for all heads in one matmul
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        """
        Apply causal self-attention to `x` of shape (B, T, C).

        With `kv_cache`, the new keys/values are appended to this layer's slot
        and the queries also attend to all cached positions before them.
        """
        B, T, _ = x.size()
        # Project the input to get queries, keys, and values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (
            q.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
            k.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
            v.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)  # number of queries in this forward pass
        Tk = k.size(
            2,
        )  # number of keys/values in total (in the cache + current forward pass)
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version
            # when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros(
                (Tq, Tk),
                dtype=torch.bool,
                device=q.device,
            )  # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device),
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)
