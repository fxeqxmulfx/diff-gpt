"""
Stationary linear Gaussian AR and its known quantized conditional law.

For one observed value of the relevant phase, integrate the bivariate-normal
rectangle probabilities. Innovation entropy alone is not the token Bayes floor:
the previous value is also quantized, and the outer bins include clipping tails.
Legendre quadrature integrates over the previous bin's uniform Gaussian CDF
coordinate. Positive bins reflect into the left tail to avoid cancellation.
"""

from __future__ import annotations

import math
from functools import cache, lru_cache

import numpy as np
import torch

COEFFICIENT = 0.75
INNOVATION_STD = 1.0
PROBABILITY_TOLERANCE = 1e-6
QUADRATURE_ORDERS = (64, 128, 256, 512)


def generate(length: int, seed: int, period: int = 1) -> np.ndarray:
    """Start every phase in its stationary Gaussian law, without a burn-in."""
    if length < 1 or period < 1:
        msg = "Gaussian AR length and period must be positive"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    stationary_std = INNOVATION_STD / math.sqrt(1 - COEFFICIENT**2)
    values = np.empty(length + period, dtype=np.float64)
    values[:period] = rng.normal(0, stationary_std, period)
    innovations = rng.normal(0, INNOVATION_STD, length)
    for index in range(period, len(values)):
        values[index] = (
            COEFFICIENT * values[index - period] + innovations[index - period]
        )
    return values[period:, None]


def bin_edges(mean: float, std: float, scale: float, vocab: int) -> np.ndarray:
    """Return raw-value bin edges for the actual order-zero DiffGPT codec."""
    internal = mean + (np.arange(1, vocab) - vocab // 2) * std / scale
    return np.concatenate(([-np.inf], internal, [np.inf]))


def normal_cdf(value: np.ndarray) -> np.ndarray:
    """Evaluate the Gaussian CDF accurately in the small left tail."""
    return np.array([0.5 * math.erfc(-float(x) / math.sqrt(2)) for x in value])


@cache
def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


def _integrate(
    edges: np.ndarray, coefficient: float, noise_std: float, order: int
) -> np.ndarray:
    stationary_std = noise_std / math.sqrt(1 - coefficient**2)
    lower, upper = edges[:-1].copy(), edges[1:].copy()
    reflect = lower >= 0
    lower[reflect], upper[reflect] = -upper[reflect].copy(), -lower[reflect].copy()
    low = normal_cdf(lower / stationary_std)
    high = normal_cdf(upper / stationary_std)
    if np.any(high <= low):
        msg = "Gaussian bins have no resolvable probability mass"
        raise ValueError(msg)
    nodes, weights = _quadrature(order)
    u = low[:, None] + (high - low)[:, None] * (nodes + 1) / 2
    x = torch.special.ndtri(torch.as_tensor(u)) * stationary_std
    x[torch.as_tensor(reflect)] *= -1
    boundaries = torch.as_tensor(edges)
    lo = (boundaries[:-1] - coefficient * x[..., None]) / noise_std
    hi = (boundaries[1:] - coefficient * x[..., None]) / noise_std
    root_two = math.sqrt(2)
    # Use survival probabilities when both bounds are positive. Subtracting
    # two CDF values near one would discard the small transition probability.
    probability = torch.where(
        lo >= 0,
        0.5 * (torch.erfc(lo / root_two) - torch.erfc(hi / root_two)),
        0.5 * (torch.erfc(-hi / root_two) - torch.erfc(-lo / root_two)),
    )
    result = 0.5 * np.einsum("q,iqj->ij", weights, probability.numpy(), optimize=False)
    np.maximum(result, 0, out=result)
    return result / result.sum(axis=1, keepdims=True)


@lru_cache(maxsize=32)
def transition(
    edges: tuple[float, ...],
    coefficient: float = COEFFICIENT,
    noise_std: float = INNOVATION_STD,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return the analytically specified token law with checked numerical error."""
    if not -1 < coefficient < 1 or not 0 < noise_std < np.inf:
        msg = "stationary Gaussian AR needs |coefficient| < 1 and noise_std > 0"
        raise ValueError(msg)
    boundaries = np.asarray(edges)
    if not np.all(np.diff(boundaries) > 0):
        msg = "quantization bin edges must increase strictly"
        raise ValueError(msg)
    previous = _integrate(boundaries, coefficient, noise_std, QUADRATURE_ORDERS[0])
    for order in QUADRATURE_ORDERS[1:]:
        current = _integrate(boundaries, coefficient, noise_std, order)
        error = float(np.max(np.abs(current - previous)))
        if error <= PROBABILITY_TOLERANCE:
            current.setflags(write=False)
            return current, {
                "quadrature_order": order,
                "max_probability_delta": error,
                "tolerance": PROBABILITY_TOLERANCE,
            }
        previous = current
    msg = f"Gaussian token oracle quadrature has not converged: {error:g}"
    raise ValueError(msg)
