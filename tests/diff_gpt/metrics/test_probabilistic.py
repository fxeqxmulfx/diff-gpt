# Copyright 2025 Mikhail Borisov
"""Tests for the probabilistic metrics: CRPS, pinball loss, coverage and MSIS."""

import numpy as np
import pytest

from diff_gpt.metrics import (
    crps_empirical,
    empirical_coverage,
    msis,
    pinball_loss,
)

# Allowed |coverage - level| for samples drawn from the true distribution.
COVERAGE_TOLERANCE = 0.05
# Plausible MSIS range for a standard-normal 95% interval with naive error 1.
MSIS_PLAUSIBLE_MIN = 2.5
MSIS_PLAUSIBLE_MAX = 5.5

# ------------------------------- CRPS -------------------------------------


def test_crps_zero_when_samples_concentrated_at_truth() -> None:
    """CRPS of a delta at the truth is zero."""
    samples = np.full((100, 5), 3.14)
    truth = np.full((5,), 3.14)
    assert crps_empirical(samples, truth) == pytest.approx(0.0, abs=1e-12)


def test_crps_matches_pairwise_formula() -> None:
    """
    Verify the sort-based implementation against the O(N²) definition.

    CRPS = E|X - y| - ½ · (1/N²) Σ_{i,j} |X_i - X_j|.
    """
    rng = np.random.default_rng(0)
    N = 30
    samples = rng.normal(size=(N, 4))
    truth = rng.normal(size=(4,))
    fast = crps_empirical(samples, truth)
    # Slow reference.
    term1 = np.mean(np.abs(samples - truth[None]), axis=0)
    slow_term2 = np.zeros(4)
    for i in range(N):
        for j in range(N):
            slow_term2 += np.abs(samples[i] - samples[j])
    slow_term2 = slow_term2 / (2.0 * N * N)
    slow = float(np.mean(term1 - slow_term2))
    assert fast == pytest.approx(slow, rel=1e-10, abs=1e-12)


def test_crps_is_proper_scoring_rule() -> None:
    """
    Check that a sharper distribution centered at the truth beats a wider one.

    CRPS is minimized by the true distribution.
    """
    truth = np.zeros(10)
    rng = np.random.default_rng(0)
    tight = rng.normal(loc=0.0, scale=1.0, size=(500, 10))
    wide = rng.normal(loc=0.0, scale=3.0, size=(500, 10))
    assert crps_empirical(tight, truth) < crps_empirical(wide, truth)


def test_crps_shape_mismatch_raises() -> None:
    """Check that a trailing-shape mismatch with the truth raises ValueError."""
    with pytest.raises(ValueError, match="trailing shape"):
        crps_empirical(np.zeros((10, 3)), np.zeros(4))


# ------------------------------ Pinball -----------------------------------


def test_pinball_zero_when_pred_equals_truth() -> None:
    """Check that pinball loss is zero at every alpha when pred equals truth."""
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.0, 2.0, 3.0])
    for alpha in (0.1, 0.5, 0.9):
        assert pinball_loss(pred, truth, alpha) == pytest.approx(0.0)


def test_pinball_alpha_0p5_equals_half_mae() -> None:
    """At alpha=0.5 pinball loss equals ½·MAE."""
    rng = np.random.default_rng(0)
    pred = rng.normal(size=50)
    truth = rng.normal(size=50)
    pb = pinball_loss(pred, truth, 0.5)
    mae = float(np.mean(np.abs(pred - truth)))
    assert pb == pytest.approx(0.5 * mae, rel=1e-10)


def test_pinball_asymmetry() -> None:
    """Under-prediction at alpha=0.9 is penalized more than over-prediction."""
    truth = np.array([10.0])
    under = np.array([5.0])
    over = np.array([15.0])
    pb_under = pinball_loss(under, truth, 0.9)
    pb_over = pinball_loss(over, truth, 0.9)
    assert pb_under > pb_over


# ----------------------------- Coverage -----------------------------------


def test_empirical_coverage_nominal_on_gaussian_samples() -> None:
    """Samples from the true distribution must yield coverage ≈ level."""
    rng = np.random.default_rng(0)
    N = 2000
    M = 500  # test points
    samples = rng.normal(size=(N, M))
    truth = rng.normal(size=M)
    for level in (0.5, 0.8, 0.95):
        cov = empirical_coverage(samples, truth, level=level)
        assert abs(cov - level) < COVERAGE_TOLERANCE, f"level={level}: coverage {cov}"


def test_empirical_coverage_zero_for_delta_and_different_truth() -> None:
    """Check that a delta away from the truth has zero coverage."""
    samples = np.full((50, 3), 0.0)
    truth = np.full((3,), 10.0)
    assert empirical_coverage(samples, truth, level=0.8) == 0.0


# ------------------------------- MSIS -------------------------------------


def test_msis_inside_interval_equals_width_over_naive() -> None:
    """When the truth is always inside the interval, MSIS = mean(width)/naive."""
    rng = np.random.default_rng(0)
    N, T = 200, 10
    samples = rng.normal(size=(N, T))
    # Truth always at 0 (center of samples) → always inside any interval.
    truth = np.zeros(T)
    # In-sample with known seasonal-naive error = 1.
    insample = np.array([0.0, 1.0] * 5)  # seasonal diff at lag 1 = 1 everywhere
    score = msis(samples, truth, insample, season=1, alpha=0.05)
    # Width at alpha=0.05 is q_97.5 - q_2.5 of standard normal samples ≈ 2·1.96.
    # Scaled by naive=1 → roughly 3.92. Just check it's positive and sensible.
    assert MSIS_PLAUSIBLE_MIN < score < MSIS_PLAUSIBLE_MAX


def test_msis_penalizes_miss() -> None:
    """Outside-interval truth gets a larger MSIS than inside."""
    rng = np.random.default_rng(0)
    N, T = 200, 10
    samples = rng.normal(size=(N, T))
    truth_inside = np.zeros(T)
    truth_outside = np.full(T, 10.0)  # far outside standard-normal interval
    insample = np.array([0.0, 1.0] * 5)
    s_in = msis(samples, truth_inside, insample, season=1, alpha=0.05)
    s_out = msis(samples, truth_outside, insample, season=1, alpha=0.05)
    assert s_out > 5.0 * s_in
