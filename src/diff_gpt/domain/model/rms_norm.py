"""
Parameter-free RMSNorm with a separated radial and tangent backward.

Source: transformer lab task155, Eqs. (1)-(3), and the Lean theorems
Transformer.Residual.rms_vjp_decomposition and rms_vjp_parallel.
Forward uses the native operation. The backward projects in FP64 before
returning the input dtype, retaining small radial derivatives lost by the
native FP32 subtraction. This does not enlarge a mathematically tiny
derivative or guarantee faster transformer grokking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx


class _RMSBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, value: torch.Tensor, eps: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.eps = eps
        return F.rms_norm(value, (value.shape[-1],), eps=eps)

    @staticmethod
    def backward(ctx: FunctionCtx, upstream: torch.Tensor) -> tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors
        value64, upstream64 = value.double(), upstream.double()
        energy = value64.square().mean(-1, keepdim=True)
        denominator = torch.where(energy > 0, energy, torch.ones_like(energy))
        parallel = value64 * (value64 * upstream64).mean(-1, keepdim=True) / denominator
        inverse = (energy + ctx.eps).rsqrt()
        gradient = (
            inverse * (upstream64 - parallel) + ctx.eps * inverse.pow(3) * parallel
        )
        return gradient.to(value.dtype), None


def rms_norm(value: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    """Normalize the last axis, retaining native opmath epsilon for each dtype."""
    if eps is None:
        # Native RMSNorm accumulates half and bfloat16 inputs in float32.
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        eps = torch.finfo(dtype).eps
    return _RMSBackward.apply(value, eps)


class RMSNorm(nn.RMSNorm):
    """A parameter-free native forward with the separated RMS backward."""

    def __init__(self, width: int) -> None:
        """Normalize one channel axis without a trainable gain or bias."""
        super().__init__(width, elementwise_affine=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Preserve native epsilon behavior, including an explicit test epsilon."""
        return rms_norm(value, self.eps)
