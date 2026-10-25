"""Position-wise feed-forward network of a transformer block."""

from typing import TYPE_CHECKING

from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    import torch


class FeedForward(nn.Module):
    """
    SwiGLU MLP: FFN_SwiGLU(x) = (Swish_1(x W) ⊗ x V) W_2, without biases.

    Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202. Swish_1 is
    SiLU, x · sigmoid(x); the gate x W scales the value x V elementwise.
    """

    def __init__(self, n_embd: int) -> None:
        """
        Build the bias-free gate/value and output projections.

        Three matrices instead of the two of a 4x MLP, so the hidden size is
        2/3 of 4 * n_embd, to the nearest integer, as in the paper: the
        parameters and FLOPs stay those of the 4x MLP.
        """
        super().__init__()
        hidden = (8 * n_embd + 1) // 3
        # The gate W and the value V in one matmul
        self.c_fc = nn.Linear(n_embd, 2 * hidden, bias=False)
        self.c_proj = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to each position of `x` independently."""
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(F.silu(gate) * value)
