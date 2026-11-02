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
from diff_gpt.domain.model.rms_norm import RMSNorm as RepairedRMSNorm
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

    @staticmethod
    def normalize_qk(
        q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the historical unit-vector query/key normalization."""
        return F.normalize(q, dim=-1, eps=1e-6), F.normalize(k, dim=-1, eps=1e-6)

    def attention_gain(self) -> torch.Tensor:
        """Return each head's learned inverse temperature."""
        return self.log_alpha.exp().view(1, self.n_head, 1, 1)

    @staticmethod
    def attend_dense(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the historical explicit scores, causal softmax and values."""
        length, keys = q.shape[-2], k.shape[-2]
        scores = (q @ k.transpose(-2, -1)) * alpha
        mask = torch.triu(
            torch.full((length, keys), -torch.inf, device=q.device, dtype=q.dtype),
            diagonal=keys - length + 1,
        )
        return (scores + mask).softmax(dim=-1) @ v

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
        q, k = self.normalize_qk(q, k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        own_value = v
        alpha = self.attention_gain()
        prefix = None if kv_cache is None else kv_cache.prefix_kv(self.layer_idx)
        if prefix is not None and length != 1:
            msg = "rows sharing a cached prefix take one position per call"
            raise ValueError(msg)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        if length == 1 and kv_cache is not None:
            attended = one_query_attention(q * alpha * self.head_dim**0.5, k, v, prefix)
        else:
            attended = self.attend_dense(q, k, v, alpha)
        unit = F.normalize(own_value, dim=-1, eps=1e-6)
        attended = attended - (attended * unit).sum(dim=-1, keepdim=True) * unit
        return self.c_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, width)
        )


class RMSQKAttention(MiniAttention):
    """Per-head repaired RMS Q/K with the learned cosine temperature retained."""

    def __init__(self, attention: MiniAttention) -> None:
        """Reuse all projections and temperatures without drawing new weights."""
        nn.Module.__init__(self)
        self.n_head = attention.n_head
        self.head_dim = attention.head_dim
        self.layer_idx = attention.layer_idx
        self.log_alpha = attention.log_alpha
        self.c_attn = attention.c_attn
        self.c_proj = attention.c_proj
        self.q_norm = RepairedRMSNorm(self.head_dim)
        self.k_norm = RepairedRMSNorm(self.head_dim)

    def normalize_qk(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize each head using native forward and the repaired backward."""
        return self.q_norm(q), self.k_norm(k)

    def attention_gain(self) -> torch.Tensor:
        """Compensate the two sqrt(head_dim) factors, preserving cosine gain."""
        return super().attention_gain() / self.head_dim


class SDPAQKAttention(RMSQKAttention):
    """Use fused dense attention while retaining Q/K RMS and its exact gain."""

    def attend_dense(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Keep the cosine gain and handle the offset of a cached query chunk."""
        q = q * (alpha * self.head_dim**0.5)
        length, keys = q.shape[-2], k.shape[-2]
        if length == keys:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        mask = torch.ones((length, keys), device=q.device, dtype=torch.bool).tril(
            diagonal=keys - length
        )
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def fix_qk_temperature(attention: RMSQKAttention) -> None:
    """Keep the supplied head gains as fixed buffers outside optimizer/state."""
    log_alpha = attention.log_alpha.detach().clone()
    del attention.log_alpha
    attention.register_buffer("log_alpha", log_alpha, persistent=False)


def use_native_rms_epsilon(model: nn.Module) -> list[str]:
    """Use dtype epsilon at repaired RMS sites that had an explicit epsilon."""
    changed = []
    for name, module in model.named_modules():
        if isinstance(module, RepairedRMSNorm) and module.eps is not None:
            module.eps = None
            changed.append(name)
    return changed


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


class RMSNormProjection(nn.Linear):
    """Normalize an existing branch output without redrawing its projection."""

    def __init__(self, projection: nn.Linear) -> None:
        """Reuse weight and bias objects and add a parameter-free output norm."""
        # Linear.__init__ would draw unused weights and change the RNG state.
        nn.Module.__init__(self)
        self.in_features = projection.in_features
        self.out_features = projection.out_features
        self.weight = projection.weight
        self.bias = projection.bias
        self.norm = nn.RMSNorm(projection.out_features, elementwise_affine=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the original projection, then native RMSNorm before addition."""
        return self.norm(super().forward(value))


def repair_rms_backward(model: nn.Module, n_embd: int) -> int:
    """Install the passing RMS backward repair without changing site epsilons."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, MiniRMSNorm):
            epsilon, width = 1e-5, n_embd
        elif isinstance(module, nn.RMSNorm):
            epsilon, width = module.eps, module.normalized_shape[-1]
        else:
            continue
        replacement = RepairedRMSNorm(width)
        replacement.eps = epsilon
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attribute, replacement)
        count += 1
    return count


class ResidualScaledProjection(nn.Linear):
    """Scale a residual branch's output while keeping its initial matrix intact."""

    def __init__(self, projection: nn.Linear, scale: float) -> None:
        """Reuse the exact parameters and draw no random numbers."""
        nn.Module.__init__(self)
        self.in_features = projection.in_features
        self.out_features = projection.out_features
        self.weight = projection.weight
        self.bias = projection.bias
        self.residual_scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Multiply after projection, before residual addition."""
        return super().forward(value) * self.residual_scale
