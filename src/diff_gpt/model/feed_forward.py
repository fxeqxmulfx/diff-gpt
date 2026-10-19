"""Position-wise feed-forward network of a transformer block."""

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    import torch


class FeedForward(nn.Module):
    """MLP: expand 4x, GELU, project back."""

    def __init__(self, n_embd: int) -> None:
        """Build the bias-free `n_embd -> 4 * n_embd -> n_embd` MLP."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to each position of `x` independently."""
        return self.net(x)
