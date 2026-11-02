"""Dense reference equations for clipped XSA and the scaled residual stream."""

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.rope import apply_rope

if TYPE_CHECKING:
    from diff_gpt.domain.model.mha import MultiHeadAttention
    from diff_gpt.domain.model.normalized_gpt import NormalizedGPT as GPT


def source_attention(
    model: MultiHeadAttention, value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Evaluate ordinary dense averaging and the uncentered source projection."""
    batch, length, width = value.shape
    q, k, v = F.linear(value, model.c_attn.weight).chunk(3, -1)
    q, k, v = (
        part.reshape(batch, length, model.n_head, model.head_dim).transpose(1, 2)
        for part in (q, k, v)
    )
    q, k = F.rms_norm(q, (model.head_dim,)), F.rms_norm(k, (model.head_dim,))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = q @ k.transpose(-1, -2) / model.head_dim**0.5
    future = ~torch.ones(length, length, dtype=torch.bool, device=value.device).tril()
    average = scores.masked_fill(future, -torch.inf).softmax(-1) @ v
    unit = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    projected = average - (average * unit).sum(-1, keepdim=True) * unit
    projected = torch.cat(
        (torch.zeros_like(projected[:, :, :1]), projected[:, :, 1:]), 2
    )
    return F.linear(
        projected.transpose(1, 2).reshape(batch, length, width), model.c_proj.weight
    )


def source_gpt(model: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """Evaluate the source function without the repeated-prefix gradient rewrite."""
    value = model.embedding_norm(model.token_embedding_table(tokens))
    cos, sin = model.rope_cos[: tokens.shape[1]], model.rope_sin[: tokens.shape[1]]
    for block in model.blocks:
        branch = source_attention(block.sa, block.ln_1(value), cos, sin)
        value = value + model.branch_scale * block.attn_out_norm(branch)
        value = value + model.branch_scale * block.ffn_out_norm(
            block.ffwd(block.ln_2(value))
        )
    return model.lm_head(model.ln_f(value))
