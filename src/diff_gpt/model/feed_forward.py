import torch
from torch import nn


class FeedForward(nn.Module):
    """GPT-2 MLP: expand 4x, GELU, project back."""

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.net(x)
        return result
