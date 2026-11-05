"""Preserve small bounded-frequency derivatives lost by subtracting tanh squared."""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx


class _BoundedMultiplier(torch.autograd.Function):
    """The same bounded multiplier with a stable first derivative."""

    @staticmethod
    def forward(ctx: FunctionCtx, value: torch.Tensor, bound: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.bound = bound
        return (bound * (value / bound).tanh()).exp()

    @staticmethod
    def backward(ctx: FunctionCtx, upstream: torch.Tensor) -> tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors
        decay = (-2 * (value / ctx.bound).abs()).exp()
        sech_squared = 4 * decay / (1 + decay).square()
        # Reuse the custom function so differentiating this VJP retains the
        # corrected derivative of the frequency multiplier as well.
        multiplier = _BoundedMultiplier.apply(value, ctx.bound)
        return upstream * multiplier * sech_squared, None


def stable_multiplier(value: torch.Tensor, bound: float = 4.0) -> torch.Tensor:
    """Evaluate exp(bound*tanh(value/bound)) without a 1-tanh² backward subtraction."""
    return _BoundedMultiplier.apply(value, bound)


def decimal_reference(value: float, bound: float = 4.0) -> tuple[float, float]:
    """Independently evaluate the real function and derivative at 80-digit precision."""
    with localcontext() as context:
        context.prec = 80
        raw, cap = Decimal(str(value)), Decimal(str(bound))
        decay = (-2 * abs(raw) / cap).exp()
        tangent = (1 - decay) / (1 + decay)
        if raw < 0:
            tangent = -tangent
        multiplier = (cap * tangent).exp()
        derivative = multiplier * 4 * decay / (1 + decay) ** 2
        return float(multiplier), float(derivative)


class StableFrequencies(nn.Module):
    """Reuse the tables and parameter; change only multiplier differentiation."""

    def __init__(self, original: nn.Module) -> None:
        """Keep initial values, learned weights, dtype and global cached positions."""
        super().__init__()
        self.log_scales = original.log_scales
        self.bound = original.bound
        self.output_dtype = original.output_dtype
        self.register_buffer("base_angles", original.base_angles, persistent=False)

    def forward(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Change the VJP without changing the computed forward angles or tables."""
        angles = self.base_angles[start:end] * stable_multiplier(
            self.log_scales.double(), self.bound
        )
        return angles.cos().to(self.output_dtype), angles.sin().to(self.output_dtype)
