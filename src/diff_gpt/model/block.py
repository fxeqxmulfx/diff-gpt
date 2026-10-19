# Copyright 2025 Mikhail Borisov
"""Transformer block: causal self-attention and an MLP on a residual stream."""

from typing import TYPE_CHECKING

from torch import nn

from diff_gpt.model.feed_forward import FeedForward
from diff_gpt.model.mha import MultiHeadAttention

if TYPE_CHECKING:
    import torch

    from diff_gpt.model.kv_cache import KVCache


class Block(nn.Module):
    """Pre-LayerNorm transformer block with a residual stream."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_idx: int,
    ) -> None:
        """
        Create the attention and MLP sublayers, each behind a bias-free LayerNorm.

        `layer_idx` is the block's index in the stack; the attention uses it to
        address its layer in the KV cache.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            layer_idx=layer_idx,
        )
        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.ffwd = FeedForward(n_embd=n_embd)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        """Apply attention, then the MLP, each as a pre-LayerNorm residual update."""
        x = x + self.sa(self.ln_1(x), kv_cache=kv_cache)
        return x + self.ffwd(self.ln_2(x))
