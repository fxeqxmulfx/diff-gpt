"""Causal multi-head exclusive self-attention with an optional KV cache."""

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.model.rope import apply_rope

if TYPE_CHECKING:
    from diff_gpt.model.kv_cache import KVCache


class MultiHeadAttention(nn.Module):
    """
    Causal multi-head exclusive self-attention (XSA).

    Zhai, "Exclusive Self Attention", arXiv:2603.09078: each head's output
    loses its component along the token's own value, so attention carries
    only the context; the token itself reaches the next layer through the
    residual stream. XSA adds no parameters.

    QK-norm RMS-normalizes each head's queries and keys, without parameters,
    before RoPE. Standard 1/sqrt(head_dim) attention scaling makes a logit
    sqrt(head_dim) times their cosine, up to the norm's numerical epsilon.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_idx: int,
        *,
        use_rope: bool = True,
    ) -> None:
        """
        Create the fused query/key/value and output projections, without biases.

        `n_embd` must split into `n_head` heads of even size, since RoPE
        rotates channel pairs. `layer_idx` is the index of the owning block; it
        selects this layer's slot in the KV cache. Each head's queries and
        keys are RMS-normalized before the standard q·k / sqrt(head_dim).
        `use_rope=False` leaves queries and keys unrotated while retaining
        the full causal attention mask.
        """
        super().__init__()
        if n_embd % n_head != 0:
            msg = f"n_embd {n_embd} must be divisible by n_head {n_head}"
            raise ValueError(msg)
        self.n_embd = n_embd
        self.n_head = n_head
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        self.head_dim = self.n_embd // self.n_head
        if self.head_dim % 2 != 0:
            msg = f"head size {self.head_dim} must be even: RoPE rotates pairs"
            raise ValueError(msg)
        # Query, key and value projections for all heads in one matmul
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        """
        Apply causal exclusive self-attention to `x` of shape (B, T, C).

        `cos` and `sin` are the RoPE tables of the T positions of `x`, shape
        (T, head_dim / 2); they rotate the queries and keys when `use_rope`
        is True. With `kv_cache`, the new keys/values are appended to this
        layer's slot and the queries also attend to all cached positions
        before them. Each head's output is then made orthogonal to the token's
        own value; the first position attends only to itself, so its output is zero.
        """
        B, T, _ = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (
            q.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
            k.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
            v.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
        # RoPE keeps the norms, and the cache stores normalized keys.
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        # Positional layers cache rotated keys; content layers cache raw keys.
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # XSA removes the direction of each token's own value, so keep these
        # values: the cache returns those of every position so far
        v_unit = F.normalize(v, dim=-1)
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
        # XSA: drop each head's component along the token's own value
        y = y - (y * v_unit).sum(dim=-1, keepdim=True) * v_unit
        if Tq == Tk:
            # Position 0 has no context: its XSA output is exactly zero.
            # Avoid amplifying cancellation roundoff in the branch's RMSNorm.
            y[:, :, 0, :] = 0
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)
