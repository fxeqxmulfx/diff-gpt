import numpy as np
import pytest

from diff_gpt.metrics import (
    crps_empirical,
    empirical_coverage,
    msis,
    pinball_loss,
)


# ------------------------------- CRPS -------------------------------------


def test_crps_zero_when_samples_concentrated_at_truth():
    """CRPS of a delta at the truth is zero."""
    samples = np.full((100, 5), 3.14)
    truth = np.full((5,), 3.14)
    assert crps_empirical(samples, truth) == pytest.approx(0.0, abs=1e-12)


def test_crps_matches_pairwise_formula():
    """Verify the sort-based implementation against the O(N²) definition
    CRPS = E|X − y| − ½ · (1/N²) Σ_{i,j} |X_i − X_j|."""
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


def test_crps_is_proper_scoring_rule():
    """Sharper distribution centered at the truth must beat a wider one.
    CRPS is minimized by the true distribution."""
    truth = np.zeros(10)
    rng = np.random.default_rng(0)
    tight = rng.normal(loc=0.0, scale=1.0, size=(500, 10))
    wide = rng.normal(loc=0.0, scale=3.0, size=(500, 10))
    assert crps_empirical(tight, truth) < crps_empirical(wide, truth)


def test_crps_shape_mismatch_raises():
    with pytest.raises(AssertionError, match="trailing shape"):
        crps_empirical(np.zeros((10, 3)), np.zeros(4))


# ------------------------------ Pinball -----------------------------------


def test_pinball_zero_when_pred_equals_truth():
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.0, 2.0, 3.0])
    for alpha in (0.1, 0.5, 0.9):
        assert pinball_loss(pred, truth, alpha) == pytest.approx(0.0)


def test_pinball_alpha_0p5_equals_half_mae():
    """At α=0.5 pinball loss equals ½·MAE."""
    rng = np.random.default_rng(0)
    pred = rng.normal(size=50)
    truth = rng.normal(size=50)
    pb = pinball_loss(pred, truth, 0.5)
    mae = float(np.mean(np.abs(pred - truth)))
    assert pb == pytest.approx(0.5 * mae, rel=1e-10)


def test_pinball_asymmetry():
    """Under-prediction at α=0.9 is penalized more than over-prediction."""
    truth = np.array([10.0])
    under = np.array([5.0])
    over = np.array([15.0])
    pb_under = pinball_loss(under, truth, 0.9)
    pb_over = pinball_loss(over, truth, 0.9)
    assert pb_under > pb_over


# ----------------------------- Coverage -----------------------------------


def test_empirical_coverage_nominal_on_gaussian_samples():
    """Samples from the true distribution must yield coverage ≈ level."""
    rng = np.random.default_rng(0)
    N = 2000
    M = 500  # test points
    samples = rng.normal(size=(N, M))
    truth = rng.normal(size=M)
    for level in (0.5, 0.8, 0.95):
        cov = empirical_coverage(samples, truth, level=level)
        assert abs(cov - level) < 0.05, f"level={level}: coverage {cov}"


def test_empirical_coverage_zero_for_delta_and_different_truth():
    samples = np.full((50, 3), 0.0)
    truth = np.full((3,), 10.0)
    assert empirical_coverage(samples, truth, level=0.8) == 0.0


# ------------------------------- MSIS -------------------------------------


def test_msis_inside_interval_equals_width_over_naive():
    """When the truth is always inside the interval, MSIS = mean(width)/naive."""
    rng = np.random.default_rng(0)
    N, T = 200, 10
    samples = rng.normal(size=(N, T))
    # Truth always at 0 (center of samples) → always inside any interval.
    truth = np.zeros(T)
    # In-sample with known seasonal-naive error = 1.
    insample = np.array([0.0, 1.0] * 5)  # seasonal diff at lag 1 = 1 everywhere
    score = msis(samples, truth, insample, season=1, alpha=0.05)
    # Width at α=0.05 is q_97.5 − q_2.5 of standard normal samples ≈ 2·1.96.
    # Scaled by naive=1 → roughly 3.92. Just check it's positive and sensible.
    assert 2.5 < score < 5.5


def test_msis_penalizes_miss():
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
