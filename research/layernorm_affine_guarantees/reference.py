"""Independent Decimal formulas for local RMS and population LayerNorm VJPs."""

from __future__ import annotations

from decimal import Decimal

RMS = "rms_existing_repaired_backward"
LAYER = "layernorm_native"


def evaluate(
    values: list[Decimal],
    epsilon: Decimal,
    normalizer: str,
    upstream: list[Decimal],
    gamma: list[Decimal] | None = None,
    beta: list[Decimal] | None = None,
) -> dict[str, list[Decimal]]:
    """Compute outputs and analytic VJPs using only scalar decimal arithmetic."""
    if normalizer not in {RMS, LAYER} or not values or epsilon <= 0:
        message = "Invalid local normalizer or positive-width/epsilon condition"
        raise ValueError(message)
    if (gamma is None) != (beta is None):
        message = "Affine scale and bias must both be supplied or both omitted"
        raise ValueError(message)
    width = len(values)
    if any(len(v) != width for v in [upstream, *([gamma, beta] if gamma else [])]):
        message = "Mismatched vector sizes"
        raise ValueError(message)
    divisor = Decimal(width)
    mean = sum(values) / divisor if normalizer == LAYER else Decimal(0)
    centered = [x - mean for x in values]
    energy = sum(x * x for x in centered) / divisor
    denominator = (energy + epsilon).sqrt()
    normalized = [x / denominator for x in centered]
    weighted = (
        [g * a for g, a in zip(upstream, gamma, strict=True)]
        if gamma is not None
        else upstream
    )
    average_upstream = sum(weighted) / divisor if normalizer == LAYER else Decimal(0)
    product_mean = sum(x * g for x, g in zip(centered, weighted, strict=True)) / divisor
    gradient = [
        (g - average_upstream - x * product_mean / (energy + epsilon)) / denominator
        for x, g in zip(centered, weighted, strict=True)
    ]
    output = (
        [a * x + b for a, x, b in zip(gamma, normalized, beta, strict=True)]
        if gamma is not None
        else normalized
    )
    result = {"output": output, "input_gradient": gradient}
    if gamma is not None:
        result["gamma_gradient"] = [
            g * x for g, x in zip(upstream, normalized, strict=True)
        ]
        result["beta_gradient"] = list(upstream)
    return result
