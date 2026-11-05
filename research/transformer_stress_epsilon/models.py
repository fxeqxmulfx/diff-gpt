"""Change only the output RMS epsilon in the fixed shifted-SwiGLU residual."""

import math
from typing import TYPE_CHECKING

from diff_gpt.domain.model.rms_norm import rms_norm
from research.transformer_stress.common import EPS, require
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork

if TYPE_CHECKING:
    import torch


class SoftPeriNetwork(ShiftedResidualNetwork):
    """Preserve input epsilon and learned matrices; soften branch normalization."""

    def __init__(self, settings: dict, depth: int, kind: str) -> None:
        """Use the exact same draws and zero-initialized scalar gates."""
        require(kind == "peri_rezero", "Epsilon study declares only the selected gate")
        super().__init__(settings, depth, kind)
        self.branch_epsilon = settings["branch_epsilon"]
        require(self.branch_epsilon > 0, "Branch epsilon must be positive")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Keep the residual coefficient after output RMS at the declared epsilon."""
        for branch, gate in zip(self.branches, self.gates, strict=True):
            raw = branch(rms_norm(value, EPS))
            value = value + rms_norm(raw, self.branch_epsilon) * gate / math.sqrt(
                self.depth
            )
        return value
