"""Replace only output RMS by Euclidean ball projection at the same RMS radius."""

from __future__ import annotations

import math

import torch

from diff_gpt.domain.model.rms_norm import rms_norm
from research.transformer_stress.common import EPS, require
from research.transformer_stress_bias.models import BiasedNetwork


def project_branch(value: torch.Tensor) -> torch.Tensor:
    """
    Leave RMS <= 1 unchanged; bound larger rows using FP64 intermediates.

    This changes the real function from RMS normalization. At the boundary,
    an explicit branch selects the exterior derivative; differentiability is
    not claimed there. vector_norm supplies a finite zero derivative at zero.
    """
    wide = value.double()
    radius = math.sqrt(value.shape[-1])
    relative_norm = torch.linalg.vector_norm(wide, dim=-1, keepdim=True) / radius
    denominator = torch.where(
        relative_norm < 1, torch.ones_like(relative_norm), relative_norm
    )
    return (wide / denominator).to(value.dtype)


class ProjectedNetwork(BiasedNetwork):
    """Keep every parameter, pre-RMS, residual gate and depth scaling intact."""

    def __init__(self, settings: dict, depth: int, kind: str) -> None:
        """Draw the original model first without adding parameters or RNG use."""
        super().__init__(settings, depth, kind)
        require(kind == "peri_rezero", "Only the declared residual gate is supported")
        require(settings["projection_mode"] == "ball", "Undeclared projection")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Keep the gate after projection, retaining the original 1/sqrt(D) scale."""
        for branch, gate in zip(self.branches, self.gates, strict=True):
            raw = branch(rms_norm(value, EPS))
            value = value + project_branch(raw) * gate / math.sqrt(self.depth)
        return value
