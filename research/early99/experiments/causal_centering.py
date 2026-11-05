"""Causal zero-sum attention mixing followed by the existing XSA projection."""

from typing import TYPE_CHECKING

import torch
from torch import nn

from diff_gpt.domain.model.centered_mini import FirstZeroQKAttention
from diff_gpt.domain.model.mha import one_query_attention
from diff_gpt.domain.model.rope import apply_rope

if TYPE_CHECKING:
    from diff_gpt.domain.model.gpt import GPT
    from diff_gpt.domain.model.kv_cache import KVCache


class CausalCenteredAttention(FirstZeroQKAttention):
    """
    Subtract the visible-prefix mean before the unchanged own-direction XSA.

    Sources: arXiv:2306.01610 Section 3.2 (gamma=-1 outside softmax) and
    arXiv:2410.07799v3 Eqs. (7)-(9), (15)-(18). Adapt their unmasked uniform
    mean to each causal prefix; retain the parent's Q/K RMS, RoPE, fixed
    temperature, XSA and output projection. The papers' asymptotic spectral
    conclusions do not automatically apply to this causal trained model.
    """

    def __init__(self, attention: FirstZeroQKAttention) -> None:
        """Borrow every tensor and module without an initialization draw."""
        nn.Module.__init__(self)
        self.n_head = attention.n_head
        self.head_dim = attention.head_dim
        self.layer_idx = attention.layer_idx
        self.c_attn, self.c_proj = attention.c_attn, attention.c_proj
        self.q_norm, self.k_norm = attention.q_norm, attention.k_norm
        self.register_buffer("log_alpha", attention.log_alpha, persistent=False)
        self.train(attention.training)

    def forward(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        *,
        return_guard: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Use the same causal mean in dense, chunked and shared-prefix decoding."""
        starts_at_zero = kv_cache is None or kv_cache.get_pos() == 0
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
            message = "rows sharing a cached prefix take one position per call"
            raise ValueError(message)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        prefix_length = 0
        prefix_sum = None
        if prefix is None:
            anchor = v[:, :, :1]
        else:
            prefix_k, prefix_v = prefix
            copies = batch // prefix_v.size(0)
            prefix_length = prefix_v.size(-2)
            anchor = prefix_v[:, :, :1].repeat_interleave(copies, dim=0)
            centered_prefix = prefix_v - prefix_v[:, :, :1]
            prefix_sum = centered_prefix.sum(dim=-2, keepdim=True).repeat_interleave(
                copies, dim=0
            )
            prefix = (prefix_k, centered_prefix)
        centered_values = v - anchor
        if length == 1 and kv_cache is not None:
            attended = one_query_attention(
                q * alpha * self.head_dim**0.5, k, centered_values, prefix
            )
        else:
            attended = self.attend_dense(q, k, centered_values, alpha)
        # Both averages use the same anchor. Their difference therefore equals
        # (A - U_causal) V; do not pass zero-sum weights to centered_projection,
        # whose own-value correction assumes a row sum of one.
        total = centered_values.cumsum(dim=-2)[:, :, -length:]
        if prefix_sum is not None:
            total = total + prefix_sum
        keys = prefix_length + centered_values.size(-2)
        counts = torch.arange(
            keys - length + 1, keys + 1, device=value.device, dtype=value.dtype
        ).view(1, 1, length, 1)
        attended = attended - total / counts
        attended = self.project_exclusive(attended, own_value)
        output = self.c_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, width)
        )
        if starts_at_zero:
            output[:, 0] = 0
        if return_guard:
            valid = (own_value[:, :, 0].norm(dim=-1) > self.xsa_epsilon).all(dim=1)
            return output, valid
        return output


def install_causal_centering(model: GPT) -> None:
    """Replace attention mixing only, preserving all existing Peri-RMS modules."""
    for block in model.blocks:
        block.sa = CausalCenteredAttention(block.sa)
