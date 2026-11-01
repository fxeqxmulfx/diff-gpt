"""Transformer block: causal self-attention and an MLP on a residual stream."""

from typing import TYPE_CHECKING

import torch
from torch import nn

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.mha import MultiHeadAttention
from diff_gpt.domain.model.rms_norm import RMSNorm

if TYPE_CHECKING:
    from diff_gpt.domain.model.kv_cache import KVCache


class Block(nn.Module):
    """Peri-RMSNorm transformer block with a residual stream."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_idx: int,
        *,
        n_layer: int,
    ) -> None:
        """
        Normalize the input and output of each attention and MLP branch.

        `layer_idx` is the block's index in the stack; the attention uses it to
        address its layer in the KV cache.
        Every layer applies RoPE. Each branch output is RMS-normalized
        before residual addition: the Peri-LN update of
        arXiv:2510.22026v2, Table 1, adapted to parameter-free RMSNorm.
        The owner supplies its total depth `n_layer`; each branch receives
        fixed coefficient 1/sqrt(2*n_layer) after normalization.
        """
        super().__init__()
        self.residual_scale = (2 * n_layer) ** -0.5
        self.ln_1 = RMSNorm(n_embd)
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            layer_idx=layer_idx,
        )
        self.ln_2 = RMSNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd)
        self.attn_out_norm = RMSNorm(n_embd)
        self.ffn_out_norm = RMSNorm(n_embd)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None,
        *,
        constant: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention, then the MLP, normalizing each branch before addition.

        `cos` and `sin` are the RoPE tables of the positions of `x`.
        A non-cached model supplies its still-constant token-prefix mask.
        Heads below clipping permanently disable that prefix rewrite.
        """
        if constant is None:
            branch = self.sa(self.ln_1(x), cos, sin, kv_cache=kv_cache)
        else:
            branch, valid = self.sa(
                self.ln_1(x), cos, sin, kv_cache=kv_cache, return_guard=True
            )
            constant = constant & valid[:, None]
        branch = self.attn_out_norm(branch)
        if constant is not None:
            branch = torch.where(constant[..., None], branch[:, :1], branch)
        x = x + self.residual_scale * branch
        branch = self.ffn_out_norm(self.ffwd(self.ln_2(x)))
        if constant is not None:
            branch = torch.where(constant[..., None], branch[:, :1], branch)
        output = x + self.residual_scale * branch
        if constant is not None:
            return output, constant
        return output
