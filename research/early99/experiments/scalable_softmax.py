"""Parameter-free SSMax with matched Q/K normalization ablation."""

from typing import TYPE_CHECKING

import torch
from torch import nn

from diff_gpt.domain.model.centered_mini import FirstZeroQKAttention
from diff_gpt.domain.model.centered_xsa import centered_projection
from diff_gpt.domain.model.mha import one_query_attention
from diff_gpt.domain.model.rope import apply_rope

if TYPE_CHECKING:
    from diff_gpt.domain.model.gpt import GPT
    from diff_gpt.domain.model.kv_cache import KVCache


class ScalableSoftmaxAttention(FirstZeroQKAttention):
    """
    Multiply causal attention scores by log(visible keys), retaining XSA.

    Source: arXiv:2501.19399v1, Eqs. (2), (11), Section 3 variant (c): s=1,
    no learned scale or bias. Keep the benchmark's fixed Q/K temperature and
    RoPE base. The separate no-QK arm bypasses only parameter-free Q/K RMS.
    Neither arm subtracts a uniform attention mean.
    """

    def __init__(self, attention: FirstZeroQKAttention, *, normalize_qk: bool) -> None:
        """Borrow all original tensors without consuming RNG or adding parameters."""
        nn.Module.__init__(self)
        self.n_head = attention.n_head
        self.head_dim = attention.head_dim
        self.layer_idx = attention.layer_idx
        self.c_attn, self.c_proj = attention.c_attn, attention.c_proj
        self.q_norm, self.k_norm = attention.q_norm, attention.k_norm
        self.use_qk_norm = normalize_qk
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
        """Scale finite queries after normalization, using full cached positions."""
        start = 0 if kv_cache is None else kv_cache.get_pos()
        batch, length, width = value.shape
        q, k, v = (
            part.reshape(batch, length, self.n_head, self.head_dim).transpose(1, 2)
            for part in self.c_attn(value).chunk(3, dim=-1)
        )
        if self.use_qk_norm:
            q, k = self.normalize_qk(q, k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        counts = torch.arange(
            start + 1, start + length + 1, device=value.device, dtype=value.dtype
        )
        # Scale before masking: log(1)=0 must never multiply a masked -infinity.
        q = q * counts.log().view(1, 1, length, 1)
        own_value = v
        alpha = self.attention_gain()
        prefix = None if kv_cache is None else kv_cache.prefix_kv(self.layer_idx)
        if prefix is not None and length != 1:
            message = "rows sharing a cached prefix take one position per call"
            raise ValueError(message)
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
        # SSMax rows still sum to one, so the original clipped XSA identity holds.
        attended = centered_projection(
            attended, own_value - anchor, own_value, eps=self.xsa_epsilon
        )
        output = self.c_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, width)
        )
        if start == 0:
            output[:, 0] = 0
        if return_guard:
            valid = (own_value[:, :, 0].norm(dim=-1) > self.xsa_epsilon).all(dim=1)
            return output, valid
        return output


def install_scalable_softmax(model: GPT, *, normalize_qk: bool = True) -> None:
    """Install one explicitly declared SSMax arm on the existing parent model."""
    for block in model.blocks:
        block.sa = ScalableSoftmaxAttention(block.sa, normalize_qk=normalize_qk)
