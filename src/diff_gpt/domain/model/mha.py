"""Causal multi-head exclusive self-attention with an optional KV cache."""

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.domain.model.centered_xsa import XSA_EPS, centered_projection
from diff_gpt.domain.model.rms_norm import rms_norm
from diff_gpt.domain.model.rope import apply_rope

if TYPE_CHECKING:
    from diff_gpt.domain.model.kv_cache import KVCache


def one_query_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Attend from one query per row to the cached keys and values of the row.

    `q` is (B, H, 1, D), and `k`, `v` are the row's own positions, (B, H, T,
    D). `prefix`, the keys and values (N, H, P, D) of N prompts, comes before
    them: rows i·S to i·S + S - 1 continue prompt i, S = B / N. The S queries
    of a prompt score its prefix together in one matrix product, so a decode
    step reads each prefix once rather than once per row, and one softmax
    spans the prefix and the row's own positions. Matrix products and a
    softmax, in place of scaled_dot_product_attention: for one query per row
    they read the cache 1.4 to 1.7 times as fast as its best backend on a
    GTX 1050 (from 20 rows of 576 positions up) and at 0.92 to 1.26 times
    the speed of flash attention on a Xeon E5-2680 v4, and they copy no
    keys. The softmax runs in float32; the scale is 1/sqrt(D), as in
    scaled_dot_product_attention.
    """
    rows, heads, _, dim = q.shape
    q = q * dim**-0.5
    scores = q @ k.transpose(-1, -2)  # (B, H, 1, T)
    if prefix is None:
        return scores.softmax(dim=-1, dtype=torch.float32).to(v.dtype) @ v
    prefix_k, prefix_v = prefix
    prompts, _, prefix_len, _ = prefix_k.shape
    samples = rows // prompts
    # (N, H, S, D): the queries of the rows that continue each prompt
    grouped = q.reshape(prompts, samples, heads, dim).transpose(1, 2)
    scores = torch.cat(
        (
            grouped @ prefix_k.transpose(-1, -2),  # (N, H, S, P)
            scores.reshape(prompts, samples, heads, -1).transpose(1, 2),
        ),
        dim=-1,
    )  # (N, H, S, P + T)
    weights = scores.softmax(dim=-1, dtype=torch.float32).to(v.dtype)
    y = weights[..., :prefix_len] @ prefix_v  # (N, H, S, D)
    own = weights[..., prefix_len:].transpose(1, 2).reshape(rows, heads, 1, -1)
    return y.transpose(1, 2).reshape(rows, heads, 1, dim) + own @ v


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
    ) -> None:
        """
        Create the fused query/key/value and output projections, without biases.

        `n_embd` must split into `n_head` heads of even size, since RoPE
        rotates channel pairs. `layer_idx` is the index of the owning block; it
        selects this layer's slot in the KV cache. Each head's queries and
        keys are RMS-normalized before the standard q·k / sqrt(head_dim).
        Centered XSA evaluates the same real projection while avoiding
        constant-value cancellation. Every layer applies RoPE.
        """
        super().__init__()
        if n_embd % n_head != 0:
            msg = f"n_embd {n_embd} must be divisible by n_head {n_head}"
            raise ValueError(msg)
        self.n_embd = n_embd
        self.n_head = n_head
        self.layer_idx = layer_idx
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
        *,
        return_guard: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply causal exclusive self-attention to `x` of shape (B, T, C).

        `cos` and `sin` are the RoPE tables of the T positions of `x`, shape
        (T, head_dim / 2); they rotate the queries and keys. With `kv_cache`,
        the new keys/values are appended to this
        layer's slot and the queries also attend to all cached positions
        before them, a shared prefix among them (one_query_attention): rows
        that share one take one position (T = 1) per call. Each head's output
        is then made orthogonal to the token's own value; the first position
        attends only to itself, so its output is zero. `return_guard` also
        returns whether all first-position value heads are above clipping;
        the full model uses that predicate to stabilize repeated prefixes.
        """
        B, T, _ = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (
            q.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
            k.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
            v.view(B, T, self.n_head, self.head_dim).transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
        # RoPE keeps the norms, and the cache stores normalized keys.
        q = rms_norm(q)
        k = rms_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # XSA removes the direction of each token's own value, so keep these
        # values: the cache returns those of every position so far
        own_value = v
        # Keys/values that groups of rows share before their own, or None
        prefix = None if kv_cache is None else kv_cache.prefix_kv(self.layer_idx)
        if prefix is not None and T != 1:
            msg = f"rows that share a cached prefix take one position, got {T}"
            raise ValueError(msg)
        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        if prefix is None:
            anchor = v[:, :, :1]
        else:
            prefix_k, prefix_v = prefix
            prefix_anchor = prefix_v[:, :, :1]
            anchor = prefix_anchor.repeat_interleave(B // prefix_v.size(0), dim=0)
            prefix = (prefix_k, prefix_v - prefix_anchor)
        # Cache storage retains raw values; only this attention call is centered.
        v = v - anchor
        Tq = q.size(2)  # number of queries in this forward pass
        # number of keys/values in total (shared prefix + cache + this pass)
        Tk = k.size(2) + (0 if prefix is None else prefix[0].size(2))
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version
            # when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = one_query_attention(q, k, v, prefix)
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
        y = centered_projection(y, own_value - anchor, own_value)
        if Tq == Tk:
            # Position 0 has no context: its XSA output is exactly zero.
            # Avoid amplifying cancellation roundoff in the branch's RMSNorm.
            y[:, :, 0, :] = 0
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.c_proj(y)
        if return_guard:
            # Reuse the values from the actual fused projection: no extra matmul.
            valid = (own_value[:, :, 0].norm(dim=-1) > XSA_EPS).all(dim=1)
            return output, valid
        return output
