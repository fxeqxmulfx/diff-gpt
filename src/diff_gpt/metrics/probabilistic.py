"""
Proper scoring rules and standard probabilistic forecasting metrics.

All functions take `samples` shaped (N, ...) with N independent Monte-Carlo
trajectories on axis 0 and truth of shape (...) matching samples[0].
"""

from __future__ import annotations

import numpy as np


def crps_empirical(samples: np.ndarray, truth: np.ndarray) -> float:
    """Empirical CRPS from N samples, averaged over trailing dimensions.

    Uses the closed-form identity
        CRPS(F, y) = E|X − y| − ½·E|X − X′|
    where X, X′ ∼ F independently. For an empirical distribution from
    sorted samples X_(1) ≤ … ≤ X_(N), the second term reduces to
    ½·E|X−X′| = (1/N²) · Σ_k (2k − N − 1) · X_(k), computed via a single
    sort per axis instead of an O(N²) pairwise matrix.

    `samples`: shape (N, ...); `truth`: shape (...). Returns scalar mean.
    """
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    N = samples.shape[0]
    assert N >= 2, "CRPS requires at least 2 samples"
    assert samples.shape[1:] == truth.shape, (
        f"samples trailing shape {samples.shape[1:]} must match truth {truth.shape}"
    )
    term1 = np.mean(np.abs(samples - truth[None]), axis=0)
    sorted_samples = np.sort(samples, axis=0)
    ranks = np.arange(1, N + 1, dtype=np.float64)
    weights = (2.0 * ranks - N - 1.0) / (N * N)
    weights_shape = [N] + [1] * (sorted_samples.ndim - 1)
    term2 = np.sum(
        sorted_samples * weights.reshape(weights_shape), axis=0
    )
    return float(np.mean(term1 - term2))


def pinball_loss(pred: np.ndarray, truth: np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss at level α ∈ (0, 1).

    A proper scoring rule for the α-quantile: minimized when `pred` is
    the true α-quantile of the predictive distribution. Returns scalar.
    """
    assert 0.0 < alpha < 1.0, f"alpha must be in (0, 1), got {alpha}"
    err = np.asarray(truth, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.maximum(alpha * err, (alpha - 1.0) * err)))


def empirical_coverage(
    samples: np.ndarray, truth: np.ndarray, level: float = 0.8
) -> float:
    """Fraction of `truth` values that fall inside the central `level`
    prediction interval of `samples`.

    A well-calibrated 80% interval should cover ~0.8 of truths on held-
    out data. Systematic under- or over-coverage indicates miscalibration.
    """
    assert 0.0 < level < 1.0
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    lo = np.quantile(samples, lo_q, axis=0)
    hi = np.quantile(samples, hi_q, axis=0)
    inside = (truth >= lo) & (truth <= hi)
    return float(np.mean(inside))


def msis(
    samples: np.ndarray,
    truth: np.ndarray,
    insample: np.ndarray,
    season: int,
    alpha: float = 0.05,
) -> float:
    """Mean Scaled Interval Score (M4's probabilistic metric).

    MSIS = mean((U − L) + (2/α)·(L − y)·𝟙{y<L} + (2/α)·(y − U)·𝟙{y>U}) / naive,
    where the scaling `naive` is the mean absolute seasonal-naive error
    on the in-sample history (same as MASE's denominator).

    Lower is better. Reduces to the Winkler score divided by a
    scale-free denominator.
    """
    assert 0.0 < alpha < 1.0
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    insample = np.asarray(insample, dtype=np.float64)
    lo = np.quantile(samples, alpha / 2.0, axis=0)
    hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    width = hi - lo
    penalty_lo = (2.0 / alpha) * (lo - truth) * (truth < lo)
    penalty_hi = (2.0 / alpha) * (truth - hi) * (truth > hi)
    score = width + penalty_lo + penalty_hi
    if len(insample) <= season:
        return float("nan")
    naive = float(np.mean(np.abs(insample[season:] - insample[:-season])))
    if naive == 0.0:
        return float("nan")
    return float(np.mean(score) / naive)
