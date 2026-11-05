"""Separate SwiGLU's odd and even paths without forbidding either function class."""

import torch
from torch import nn

from research.transformer_stress.common import require
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork


class LearnedEvenSwiGLU(nn.Module):
    """Use an odd initial branch with a learnable scalar for its quadratic path."""

    def __init__(self, original: nn.Module) -> None:
        """Preserve every matrix and add exactly one zero-initialized scalar."""
        super().__init__()
        self.c_fc, self.c_proj = original.c_fc, original.c_proj
        self.even_gain = nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """At gain one, recover shifted SwiGLU in real arithmetic."""
        gate, values = self.c_fc(value).chunk(2, -1)
        factor = 1 + 0.5 * gate * (0.5 * gate).tanh()
        factor = factor + 0.5 * self.even_gain * gate
        return self.c_proj(factor * values)


class ConfiguredNetwork(ShiftedResidualNetwork):
    """Use either the exact shifted parent or the declared learnable even path."""

    def __init__(self, settings: dict, depth: int, kind: str) -> None:
        """Retain the original matrix draws, residual gates, norms and depth."""
        super().__init__(settings, depth, kind)
        arm = settings["activation_variant"]
        require(arm in {"shifted", "learned_even"}, "Unknown activation arm")
        if arm == "learned_even":
            self.branches = nn.ModuleList(
                LearnedEvenSwiGLU(branch) for branch in self.branches
            )
