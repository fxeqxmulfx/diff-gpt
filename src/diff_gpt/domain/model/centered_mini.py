"""Centered evaluation of the passing GPTMini attention with repaired Q/K RMS."""

from typing import TYPE_CHECKING

from diff_gpt.domain.model.centered_xsa import centered_projection
from diff_gpt.domain.model.mha import one_query_attention
from diff_gpt.domain.model.mini_block import LowEpsilonQKAttention
from diff_gpt.domain.model.rope import apply_rope

if TYPE_CHECKING:
    import torch

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
