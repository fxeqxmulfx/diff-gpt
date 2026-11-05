"""Widen the shifted SwiGLU while preserving every original matrix entry."""

import torch
from torch import nn

from research.transformer_stress_fixes.learning import ShiftedResidualNetwork
from research.transformer_stress_fixes.models import ShiftedSwiGLU


class ExpandedNetwork(ShiftedResidualNetwork):
    """Keep the original feature columns and append declared random features."""

    def __init__(self, settings: dict, depth: int, kind: str) -> None:
        """Record increased capacity; do not imply matched parameter counts."""
        super().__init__(settings, depth, kind)
        width, hidden = settings["dimension"], settings["expanded_hidden"]
        rng = torch.Generator().manual_seed(torch.initial_seed() + 271828)
        expanded = []
        for original in self.branches:
            old_hidden = original.c_proj.in_features
            large = nn.Module()
            large.c_fc = nn.Linear(width, 2 * hidden, bias=False)
            large.c_proj = nn.Linear(hidden, width, bias=False)
            with torch.no_grad():
                large.c_fc.weight.normal_(
                    std=settings["initial_weight_std"], generator=rng
                )
                large.c_proj.weight.normal_(
                    std=settings["initial_weight_std"], generator=rng
                )
                large.c_fc.weight[:old_hidden].copy_(original.c_fc.weight[:old_hidden])
                large.c_fc.weight[hidden : hidden + old_hidden].copy_(
                    original.c_fc.weight[old_hidden:]
                )
                large.c_proj.weight[:, :old_hidden].copy_(original.c_proj.weight)
            expanded.append(ShiftedSwiGLU(large))
        self.branches = nn.ModuleList(expanded)
