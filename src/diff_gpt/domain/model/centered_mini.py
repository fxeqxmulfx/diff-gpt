"""Centered evaluation of the passing GPTMini attention with repaired Q/K RMS."""

from typing import TYPE_CHECKING

import torch
from torch import nn

from diff_gpt.domain.model.centered_xsa import centered_projection
from diff_gpt.domain.model.mha import one_query_attention
from diff_gpt.domain.model.mini_block import LowEpsilonQKAttention, MiniBlock
from diff_gpt.domain.model.rope import apply_rope

if TYPE_CHECKING:
    from diff_gpt.domain.model.kv_cache import KVCache


class CenteredQKAttention(LowEpsilonQKAttention):
    """Evaluate ordinary XSA using values centered around the first key."""

    def forward(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        *,
        return_guard: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Preserve real score/projection equations and raw cached values."""
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
        if prefix is None:
            anchor = v[:, :, :1]
        else:
            prefix_k, prefix_v = prefix
            anchor = prefix_v[:, :, :1].repeat_interleave(
                batch // prefix_v.size(0), dim=0
            )
            prefix = (prefix_k, prefix_v - prefix_v[:, :, :1])
        centered_values = v - anchor
        if length == 1 and kv_cache is not None:
            attended = one_query_attention(
                q * alpha * self.head_dim**0.5, k, centered_values, prefix
            )
        else:
            attended = self.attend_dense(q, k, centered_values, alpha)
        attended = centered_projection(
            attended, own_value - anchor, own_value, eps=self.xsa_epsilon
        )
        output = self.c_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, width)
        )
        if return_guard:
            valid = (own_value[:, :, 0].norm(dim=-1) > self.xsa_epsilon).all(dim=1)
            return output, valid
        return output


class FirstZeroQKAttention(CenteredQKAttention):
    """Force the first global attention output and its direct gradient to zero."""

    def forward(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        *,
        return_guard: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Zero the first global position, including a cached prefill."""
        starts_at_zero = kv_cache is None or kv_cache.get_pos() == 0
        result = super().forward(value, cos, sin, kv_cache, return_guard=return_guard)
        if return_guard:
            output, guard = result
        else:
            output = result
        if starts_at_zero:
            # The bias-free finite linear projection commutes with zeroing.
            output[:, 0] = 0
        return (output, guard) if return_guard else output


class PrefixStableBlock(MiniBlock):
    """Reuse a prepared block and share its still-valid constant prefix states."""

    def __init__(self, block: MiniBlock) -> None:
        """Borrow existing modules without drawing or changing any parameters."""
        nn.Module.__init__(self)
        self.ln_1, self.ln_2 = block.ln_1, block.ln_2
        self.sa, self.ffwd = block.sa, block.ffwd

    def forward(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        *,
        constant: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rewrite a prefix only while every layer's value heads stay unclipped."""
        if constant is None:
            return super().forward(value, cos, sin, kv_cache)
        branch, valid = self.sa(self.ln_1(value), cos, sin, kv_cache, return_guard=True)
        constant = constant & valid[:, None]
        value = value + torch.where(constant[..., None], branch[:, :1], branch)
        branch = self.ffwd(self.ln_2(value))
        value = value + torch.where(constant[..., None], branch[:, :1], branch)
        return value, constant
