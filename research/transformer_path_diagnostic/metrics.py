"""Finite descriptive tensor statistics and checked additive gradient paths."""

from __future__ import annotations

import torch

from research.early99_reporting.round2_audit import require

PATH_TOLERANCE = 1e-5


def stats(value: torch.Tensor) -> dict:
    """Summarize finite entries; retain nonfinite counts, including causal masks."""
    value = value.detach().double()
    finite = torch.isfinite(value)
    kept = value[finite]
    return {
        "shape": list(value.shape),
        "elements": value.numel(),
        "nonfinite": int((~finite).sum()),
        "rms": kept.square().mean().sqrt().item() if kept.numel() else None,
        "mean": kept.mean().item() if kept.numel() else None,
        "abs_max": kept.abs().max().item() if kept.numel() else None,
        "zero_fraction": (kept == 0).double().mean().item() if kept.numel() else None,
    }


def norm(value: torch.Tensor) -> float:
    """Compute a detached Euclidean/Frobenius norm in FP64."""
    return value.detach().double().norm().item()


def ratio(numerator: float, denominator: float) -> float | None:
    """Keep zero-denominator comparisons undefined instead of adding an epsilon."""
    return numerator / denominator if denominator else None


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Measure alignment, retaining undefined zero-vector angles."""
    return ratio(
        (left.detach().double() * right.detach().double()).sum().item(),
        norm(left) * norm(right),
    )


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Use relative L2 for nonzero references and absolute L2 at exact zero."""
    distance = norm(actual.double() - expected.double())
    return distance / norm(expected) if norm(expected) else distance


def decomposition(
    first: torch.Tensor, second: torch.Tensor, total: torch.Tensor
) -> dict:
    """Check an additive path split and quantify amplification or cancellation."""
    error = relative_error(first + second, total)
    require(
        error <= PATH_TOLERANCE, f"Gradient path does not reconstruct total: {error}"
    )
    return {
        "first_l2": norm(first),
        "second_l2": norm(second),
        "total_l2": norm(total),
        "second_to_first_l2": ratio(norm(second), norm(first)),
        "cosine": cosine(first, second),
        "sum_to_separate_norms": ratio(norm(total), norm(first) + norm(second)),
        "reconstruction_relative_l2": error,
    }


def probe_like(logits: torch.Tensor, seed: int) -> torch.Tensor:
    """Draw a reproducible centered unit cotangent independent of confidence."""
    require(logits.shape[-1] > 1, "A centered probe requires at least two logits")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(64):
        probe = torch.randint(0, 2, logits.shape, generator=generator).double() * 2 - 1
        probe -= probe.mean(dim=-1, keepdim=True)
        if probe.norm() > 0:
            break
    require(probe.norm() > 0, "No nonzero centered Rademacher probe in 64 draws")
    probe /= probe.norm()
    return probe.to(device=logits.device, dtype=logits.dtype)
