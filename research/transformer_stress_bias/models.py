"""Add zero-initialized bias vectors around the existing FFN linear modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from torch.nn import functional as F

from research.transformer_stress.common import require
from research.transformer_stress_parity.models import ConfiguredNetwork

if TYPE_CHECKING:
    import torch

BIAS_NAMES = ("hidden_bias", "output_bias")


def is_new_bias(name: str) -> bool:
    """Separate declared added vectors from the original common parameters."""
    return name.rsplit(".", 1)[-1] in BIAS_NAMES


class BiasedSwiGLU(nn.Module):
    """Retain shifted or learned-even algebra, adding the selected bias vectors."""

    def __init__(self, original: nn.Module, mode: str) -> None:
        """Borrow every matrix/coefficient and draw no new random values."""
        super().__init__()
        require(mode in {"none", "output", "all"}, "Undeclared bias mode")
        self.c_fc, self.c_proj = original.c_fc, original.c_proj
        if hasattr(original, "even_gain"):
            self.even_gain = original.even_gain
        self.register_parameter(
            "hidden_bias",
            nn.Parameter(self.c_fc.weight.new_zeros(self.c_fc.out_features))
            if mode == "all"
            else None,
        )
        self.register_parameter(
            "output_bias",
            nn.Parameter(self.c_proj.weight.new_zeros(self.c_proj.out_features))
            if mode in {"output", "all"}
            else None,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Place hidden biases before activation and output bias before post-RMS."""
        projected = self.c_fc(value)
        if self.hidden_bias is not None:
            projected = projected + self.hidden_bias
        gate, values = projected.chunk(2, -1)
        if hasattr(self, "even_gain"):
            factor = 1 + 0.5 * gate * (0.5 * gate).tanh()
            factor = factor + 0.5 * self.even_gain * gate
        else:
            factor = 1 + F.silu(gate)
        output = self.c_proj(factor * values)
        if self.output_bias is not None:
            output = output + self.output_bias
        return output


class BiasedNetwork(ConfiguredNetwork):
    """Keep the complete residual forward path and replace only its FFN wrapper."""

    def __init__(self, settings: dict, depth: int, kind: str) -> None:
        """Construct the complete parent first, then append deterministic biases."""
        super().__init__(settings, depth, kind)
        self.branches = nn.ModuleList(
            BiasedSwiGLU(branch, settings["bias_mode"]) for branch in self.branches
        )

    def common_branch_parameters(self) -> dict[str, torch.Tensor]:
        """Hash the original matrices and even coefficients in their original order."""
        return {n: p for n, p in self.branches.named_parameters() if not is_new_bias(n)}

    def added_bias_parameters(self) -> int:
        """Report the actual extra vector capacity from registered parameters."""
        return sum(p.numel() for n, p in self.named_parameters() if is_new_bias(n))
