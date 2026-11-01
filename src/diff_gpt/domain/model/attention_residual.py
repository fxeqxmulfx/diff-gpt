"""
Query/key normalized attention over earlier depth outputs.

Adapted from Attention Residuals, arXiv:2603.15031v1, Section 3.1.
Unlike that key-only formula, both operands have parameter-free RMSNorm,
there is no affine key gain, and scores divide by sqrt(width). Raw source
values are retained. Transformer.AttentionResiduals.Normalized in the
companion Lean repository bounds scores by sqrt(width) and their finite-
epsilon sensitivity by 2/sqrt(eps). Neither bound guarantees learning.
"""

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import Tensor


def depth_scores(values: Tensor, query: Tensor, eps: float) -> Tensor:
    """
    Score (..., sources, width) values with one learned depth query.

    Normalize query and keys for scoring. A zero query produces uniform weights;
    positive epsilon keeps zero keys finite. Finite epsilon means scaling
    invariance is approximate, and derivatives near zero can be large.
    """
    width = query.shape[-1]
    q = F.rms_norm(query, (width,), eps=eps)
    k = F.rms_norm(values, (width,), eps=eps)
    return (k * q).sum(dim=-1) / math.sqrt(width)


def depth_attention(values: Tensor, query: Tensor, eps: float) -> Tensor:
    """Return the convex combination of the original, unnormalized values."""
    weights = depth_scores(values, query, eps).softmax(dim=-1)
    return (weights.unsqueeze(-1) * values).sum(dim=-2)


class QKAttentionResidual(nn.Module):
    """
    A depth aggregator with one learned query and no learned temperature.

    Sources include the embedding and completed branch/block outputs. The
    caller determines that list and its causality. This module alone does
    not replace a transformer's residual recurrence or cache semantics.
    """

    def __init__(self, width: int, *, eps: float = 1e-6) -> None:
        """Initialize the query at unit RMS, away from the steep zero region."""
        super().__init__()
        if type(width) is not int or width <= 0:
            msg = "width must be a positive integer"
            raise ValueError(msg)
        if not math.isfinite(eps) or eps <= 0:
            msg = "eps must be finite and positive"
            raise ValueError(msg)
        self.width = width
        self.eps = eps
        self.query = nn.Parameter(torch.empty(width))
        with torch.no_grad():
            self.query.normal_()
            self.query.div_(self.query.square().mean().sqrt().clamp_min(1e-12))

    def forward(self, values: Tensor) -> Tensor:
        """Aggregate a nonempty source axis; preserve leading batch axes."""
        if values.ndim <= 1 or values.shape[-1] != self.width:
            msg = "values must have shape (..., sources, width)"
            raise ValueError(msg)
        if values.shape[-2] == 0:
            msg = "at least one source, normally the embedding, is required"
            raise ValueError(msg)
        return depth_attention(values, self.query, self.eps)
