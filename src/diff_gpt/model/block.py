import torch
from torch import nn

from diff_gpt.model.mha import MultiHeadAttention
from diff_gpt.model.feed_forward import FeedForward
from diff_gpt.model.kv_cache import KVCache


class Block(nn.Module):
    """Pre-LayerNorm transformer block with a residual stream, as in GPT-2."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            layer_idx=layer_idx,
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd=n_embd)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        x = x + self.sa(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.ffwd(self.ln_2(x))
        return x
