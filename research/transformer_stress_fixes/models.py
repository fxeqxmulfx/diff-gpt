"""Preserve real equations while widening vulnerable numerical intermediates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.domain.model.centered_xsa import centered_projection
from diff_gpt.domain.model.mha import one_query_attention
from diff_gpt.domain.model.rms_norm import RMSNorm
from diff_gpt.domain.model.rope import apply_rope
from research.early99.experiments.scalable_softmax import ScalableSoftmaxAttention
from research.transformer_stress.common import EPS, full_stack

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx


class _SafeRMS(torch.autograd.Function):
    """Range-scaled forward and pairwise FP64 VJP for the same RMS function."""

    @staticmethod
    def forward(ctx: FunctionCtx, value: torch.Tensor, eps: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.eps = eps
        wide = value.double()
        scale = wide.abs().amax(-1, keepdim=True).clamp_min(math.sqrt(eps))
        unit = wide / scale
        denominator = (
            unit.square().mean(-1, keepdim=True) + (math.sqrt(eps) / scale).square()
        ).sqrt()
        return (unit / denominator).to(value.dtype)

    @staticmethod
    def backward(ctx: FunctionCtx, upstream: torch.Tensor) -> tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors
        x, g = value.double(), upstream.double()
        energy = x.square().mean(-1, keepdim=True)
        safe_energy = torch.where(energy > 0, energy, torch.ones_like(energy))
        inverse = (energy + ctx.eps).rsqrt()
        parts = []
        # x_j*g_i - x_i*g_j is exactly zero for identical FP32 vectors,
        # avoiding the projection/subtraction cancellation of a radial VJP.
        # Chunk the output axis to limit temporary storage. This correctness
        # prototype intentionally trades arithmetic for radial accuracy.
        for start in range(0, x.shape[-1], 16):
            xi, gi = x[..., start : start + 16, None], g[..., start : start + 16, None]
            wedge = x[..., None, :] * gi - xi * g[..., None, :]
            tangent = (wedge * x[..., None, :]).mean(-1) / safe_energy
            parts.append(tangent)
        tangent = torch.cat(parts, -1)
        gradient = inverse * tangent + ctx.eps * inverse.pow(3) * (g - tangent)
        return gradient.to(value.dtype), None


def safe_rms(value: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Retain the actual finite epsilon; do not clip or cap the normalized input."""
    return _SafeRMS.apply(value, eps)


class SafeRMSNorm(nn.Module):
    """Parameter-free substitute preserving the declared normalization equation."""

    def __init__(self, eps: float = EPS) -> None:
        """Store the original explicit epsilon without a learned scale."""
        super().__init__()
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the range-safe forward and radial-safe backward."""
        return safe_rms(value, self.eps)


def replace_rms(module: nn.Module) -> None:
    """Replace normalization modules while preserving every learned parameter."""
    for name, child in list(module.named_children()):
        if isinstance(child, RMSNorm):
            setattr(module, name, SafeRMSNorm(EPS if child.eps is None else child.eps))
        else:
            replace_rms(child)


class WideFrequencies(nn.Module):
    """Construct p*log(p+1)*frequency*exp(alpha) in FP64 before sin/cos."""

    def __init__(self, original: nn.Module, bound: float | None = None) -> None:
        """Reuse the learned vector and rebuild the intended angle in FP64."""
        super().__init__()
        self.log_scales = original.log_scales
        self.bound = bound
        count, bands = original.base_angles.shape
        positions = torch.arange(
            count, device=original.base_angles.device, dtype=torch.float64
        )
        frequencies = 10000 ** -(
            torch.arange(bands, device=positions.device).double() / bands
        )
        self.register_buffer(
            "base_angles",
            (positions * positions.log1p())[:, None] * frequencies,
            persistent=False,
        )
        self.output_dtype = original.base_angles.dtype

    def forward(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the cache's global position map and differentiable log frequencies."""
        logs = self.log_scales.double()
        if self.bound is not None:
            logs = self.bound * (logs / self.bound).tanh()
        angles = self.base_angles[start:end] * logs.exp()
        return angles.cos().to(self.output_dtype), angles.sin().to(self.output_dtype)


class WidePeriSwiGLU(nn.Module):
    """Widen products and output projection through RMS; keep weights in FP32."""

    def __init__(self, original: nn.Module) -> None:
        """Borrow both original matrices and the declared scaled Peri projection."""
        super().__init__()
        self.c_fc, self.c_proj = original.c_fc, original.c_proj

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Avoid overflowing products before a subsequent bounded normalization."""
        gate, values = F.linear(value.double(), self.c_fc.weight.double()).chunk(2, -1)
        raw = F.linear(F.silu(gate) * values, self.c_proj.weight.double())
        return (safe_rms(raw) * self.c_proj.residual_scale).to(value.dtype)


class WideXSAAttention(ScalableSoftmaxAttention):
    """Keep SSMax/cache logic, widening XSA and output projection through Peri-RMS."""

    def forward(
        self,
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: object = None,
        *,
        return_guard: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Retain raw cached values, singleton zeroing and causal counts."""
        start = 0 if kv_cache is None else kv_cache.get_pos()
        batch, length, width = value.shape
        q, k, v = (
            part.reshape(batch, length, self.n_head, self.head_dim).transpose(1, 2)
            for part in self.c_attn(value).chunk(3, dim=-1)
        )
        if self.use_qk_norm:
            q, k = self.normalize_qk(q, k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q = q * torch.arange(
            start + 1, start + length + 1, device=value.device, dtype=value.dtype
        ).log().view(1, 1, length, 1)
        own_value = v
        prefix = None if kv_cache is None else kv_cache.prefix_kv(self.layer_idx)
        if prefix is not None and length != 1:
            message = "rows sharing a cached prefix take one position per call"
            raise ValueError(message)
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        if prefix is None:
            anchor = v[:, :, :1]
        else:
            prefix_k, prefix_v = prefix
            anchor = prefix_v[:, :, :1].repeat_interleave(
                batch // prefix_v.size(0), dim=0
            )
            prefix = (prefix_k, prefix_v - prefix_v[:, :, :1])
        if length == 1 and kv_cache is not None:
            attended = one_query_attention(
                q * self.attention_gain() * self.head_dim**0.5, k, v - anchor, prefix
            )
        else:
            attended = self.attend_dense(q, k, v - anchor, self.attention_gain())
        exclusive = centered_projection(
            attended.double(),
            (own_value - anchor).double(),
            own_value.double(),
            eps=self.xsa_epsilon,
        )
        raw = F.linear(
            exclusive.transpose(1, 2).contiguous().view(batch, length, width),
            self.c_proj.weight.double(),
        )
        output = (self.c_proj.norm(raw) * self.c_proj.residual_scale).to(value.dtype)
        if start == 0:
            output[:, 0] = 0
        if return_guard:
            valid = (own_value[:, :, 0].norm(dim=-1) > self.xsa_epsilon).all(dim=1)
            return output, valid
        return output


def fixed_model(settings: dict, layers: int, context: int, variant: str) -> nn.Module:
    """Build one declared numerical fix or the combined fix, preserving FP32 weights."""
    model = full_stack(settings, layers, context)
    combined = variant.startswith("all_numeric")
    if variant == "wide_xsa" or combined:
        for block in model.blocks:
            block.sa = WideXSAAttention(block.sa, normalize_qk=True)
    if variant == "range_rms" or combined:
        replace_rms(model)
    if variant == "wide_rotary" or combined:
        model.lerope = WideFrequencies(
            model.lerope, bound=4.0 if variant == "all_numeric_bounded_rope" else None
        )
    if variant == "wide_swiglu" or combined:
        for block in model.blocks:
            block.ffwd = WidePeriSwiGLU(block.ffwd)
    return model


class ShiftedSwiGLU(nn.Module):
    """Test an explicit linear path: (1 + SiLU(gate)) * value, same matrices."""

    def __init__(self, original: nn.Module) -> None:
        """Retain initialization and capacity; intentionally change the function."""
        super().__init__()
        self.c_fc, self.c_proj = original.c_fc, original.c_proj

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Add the declared linear path without adding teacher-specific features."""
        gate, values = self.c_fc(value).chunk(2, -1)
        return self.c_proj((1 + F.silu(gate)) * values)
