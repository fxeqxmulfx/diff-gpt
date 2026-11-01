"""Transformer block: causal self-attention and an MLP on a residual stream."""

from typing import TYPE_CHECKING

from torch import nn

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.mha import MultiHeadAttention
from diff_gpt.domain.model.rms_norm import RMSNorm

if TYPE_CHECKING:
    import torch

    from diff_gpt.domain.model.kv_cache import KVCache


class Block(nn.Module):
    """Peri-RMSNorm transformer block with a residual stream."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        layer_idx: int,
        *,
        use_rope: bool = True,
        centered_xsa: bool = False,
    ) -> None:
        """
        Normalize the input and output of each attention and MLP branch.

        `layer_idx` is the block's index in the stack; the attention uses it to
        address its layer in the KV cache.
        `use_rope` selects whether this layer rotates its queries and keys by
        position. Each branch output is RMS-normalized
        before residual addition: the Peri-LN update of
        arXiv:2510.22026v2, Table 1, adapted to parameter-free RMSNorm.
        """
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            layer_idx=layer_idx,
            use_rope=use_rope,
            centered_xsa=centered_xsa,
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
        return_guard: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention, then the MLP, normalizing each branch before addition.

        `cos` and `sin` are the RoPE tables of the positions of `x`.
        """
        if return_guard:
            branch, valid = self.sa(
                self.ln_1(x), cos, sin, kv_cache=kv_cache, return_guard=True
            )
        else:
            branch = self.sa(self.ln_1(x), cos, sin, kv_cache=kv_cache)
        x = x + self.attn_out_norm(branch)
        output = x + self.ffn_out_norm(self.ffwd(self.ln_2(x)))
        if return_guard:
            return output, valid
        return output
