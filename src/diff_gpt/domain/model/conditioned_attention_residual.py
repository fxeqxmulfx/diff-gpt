"""
Q/K depth attention in explicit, rescalable query coordinates.

Source: Attention Residuals, arXiv:2603.15031v1 Section 3.1, modified as in
attention_residual.py at e13cf13. Transformer.AttentionResiduals.QueryCoordinates
in the companion Lean repository proves the coordinate identity. If physical
q = scale * u, RMS(q; eps) = RMS(u; eps / scale**2) for scale > 0. Keeping
that epsilon correction preserves the finite-epsilon model, not just its
large-norm approximation. Keys retain their original epsilon; values are raw.

Scale is coordinate metadata, not a learned gain or temperature. Gradients
are ordinary derivatives in the current coordinates. Rescaling the stored
query alone would invalidate Schedule-Free averaging and gradient moments;
use the application-level DepthQueryConditioner to change coordinates.
"""

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.attention_residual import QKAttentionResidual

if TYPE_CHECKING:
    from torch import Tensor


class ConditionedQKAttentionResidual(QKAttentionResidual):
    """Retain a physical query through positive coordinate changes."""

    query_scale: Tensor

    def __init__(self, width: int, *, eps: float = 1e-6) -> None:
        """Start with the original unit-RMS query and identity coordinates."""
        super().__init__(width, eps=eps)
        self.register_buffer("query_scale", self.query.new_ones(()))

    def scores(self, values: Tensor) -> Tensor:
        """Use exact transformed epsilon and an ordinary differentiable graph."""
        query_epsilon = self.eps / self.query_scale / self.query_scale
        q = self.query * torch.rsqrt(self.query.square().mean() + query_epsilon)
        keys = F.rms_norm(values, (self.width,), eps=self.eps)
        return (keys * q).sum(dim=-1) / self.width**0.5

    def forward(self, values: Tensor) -> Tensor:
        """Aggregate the original values with parameter-free normalized scores."""
        if values.ndim <= 1 or values.shape[-1] != self.width:
            msg = "values must have shape (..., sources, width)"
            raise ValueError(msg)
        if values.shape[-2] == 0:
            msg = "at least one source, normally the embedding, is required"
            raise ValueError(msg)
        weights = self.scores(values).softmax(dim=-1)
        return (weights.unsqueeze(-1) * values).sum(dim=-2)
