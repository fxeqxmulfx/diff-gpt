"""
Historical GPTMini blocks, retained as the associative-recall baseline.

Source: transformer/experiments/archive/gpt_mini/gpt_mini.py and the
seeded Normal(0.02) builder used by transformer/experiments/basis.
"""

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.mha import one_query_attention
from diff_gpt.domain.model.rope import apply_rope

if TYPE_CHECKING:
    from diff_gpt.domain.model.kv_cache import KVCache


class MiniRMSNorm(nn.Module):
    """Parameter-free native RMSNorm with GPTMini's explicit epsilon."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize the channel axis, using epsilon 1e-5 in every dtype."""
        return F.rms_norm(value, (value.shape[-1],), eps=1e-5)


class MiniAttention(nn.Module):
    """Unit Q/K, learned per-head temperature, ordinary XSA and RoPE."""

    def __init__(self, n_embd: int, n_head: int, layer_idx: int) -> None:
        """Construct the bias-free fused QKV and output matrices."""
        super().__init__()
        if n_head < 1 or n_embd % n_head:
            msg = f"n_embd {n_embd} must be divisible by positive n_head {n_head}"
            raise ValueError(msg)
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.layer_idx = layer_idx
        if self.head_dim % 2:
            msg = f"head size {self.head_dim} must be even: RoPE rotates pairs"
            raise ValueError(msg)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.log_alpha = nn.Parameter(
            torch.full((n_head,), 0.5 * math.log(self.head_dim))
        )

    def forward(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Evaluate historical dense attention, or continue a cached prefix."""
        batch, length, width = value.shape
        q, k, v = (
            part.reshape(batch, length, self.n_head, self.head_dim).transpose(1, 2)
            for part in self.c_attn(value).chunk(3, dim=-1)
        )
        q, k = F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        own_value = v
        alpha = self.log_alpha.exp().view(1, self.n_head, 1, 1)
        prefix = None if kv_cache is None else kv_cache.prefix_kv(self.layer_idx)
        if prefix is not None and length != 1:
            msg = "rows sharing a cached prefix take one position per call"
            raise ValueError(msg)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        if length == 1 and kv_cache is not None:
            attended = one_query_attention(q * alpha * self.head_dim**0.5, k, v, prefix)
        else:
            scores = (q @ k.transpose(-2, -1)) * alpha
            keys = k.shape[-2]
            mask = torch.triu(
                torch.full((length, keys), -torch.inf, device=q.device, dtype=q.dtype),
                diagonal=keys - length + 1,
            )
            attended = (scores + mask).softmax(dim=-1) @ v
        unit = F.normalize(own_value, dim=-1, eps=1e-6)
        attended = attended - (attended * unit).sum(dim=-1, keepdim=True) * unit
        return self.c_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, width)
        )


class MiniFeedForward(nn.Module):
    """Historical bias-free ReLU-squared MLP, with hidden width 4d."""

    def __init__(self, n_embd: int) -> None:
        """Create the two matrices in historical random-draw order."""
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply W_out ReLU(W_in x)^2 independently at every position."""
        return self.c_proj(F.relu(self.c_fc(value)).square())


class MiniBlock(nn.Module):
    """PreNorm residual updates with unnormalized, unscaled branch outputs."""

    def __init__(self, n_embd: int, n_head: int, layer_idx: int) -> None:
        """Build attention, MLP and parameter-free input norms."""
        super().__init__()
        self.ln_1 = MiniRMSNorm()
        self.ln_2 = MiniRMSNorm()
        self.sa = MiniAttention(n_embd, n_head, layer_idx)
        self.ffwd = MiniFeedForward(n_embd)

    def forward(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Add the attention output, then the ReLU-squared MLP output."""
        value = value + self.sa(self.ln_1(value), cos, sin, kv_cache)
        return value + self.ffwd(self.ln_2(value))


def initialize_swiglu(blocks: nn.ModuleList, n_embd: int) -> None:
    """Install the passing SwiGLU feature without shifting shared random draws."""
    replacements = [FeedForward(n_embd) for _ in blocks]
    for block, replacement in zip(blocks, replacements, strict=True):
        for parameter in replacement.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        block.ffwd = replacement


class RMSNormEmbedding(nn.Embedding):
    """Native parameter-free RMS normalization after an existing embedding."""

    def __init__(self, embedding: nn.Embedding) -> None:
        """Reuse the exact weight object, retaining the tied output-head gradient."""
        super().__init__(
            embedding.num_embeddings,
            embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
            _weight=embedding.weight,
        )
        self.weight = embedding.weight
        self.norm = nn.RMSNorm(embedding.embedding_dim, elementwise_affine=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Normalize the embedding with native dtype epsilon and native backward."""
        return self.norm(super().forward(tokens))
