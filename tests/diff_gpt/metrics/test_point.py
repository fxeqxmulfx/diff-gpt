import numpy as np
import pytest

from diff_gpt.metrics import bayes_point


def test_mae_is_median_and_mse_is_mean():
    s = np.array([[1.0, 10.0], [3.0, 20.0], [8.0, 60.0]])  # (N=3, 2)
    assert np.array_equal(bayes_point(s, "mae"), np.array([3.0, 20.0]))
    assert np.allclose(bayes_point(s, "mse"), np.array([4.0, 30.0]))


def test_smape_is_concave_so_bimodal_optimum_is_a_mode():
    """Two-point distribution {1, 3}: E sMAPE is 53.3 at the median 2 but 50.0
    at either sample, so the sMAPE-Bayes point is a mode, not the median."""
    s = np.array([[1.0], [3.0]])
    y = bayes_point(s, "smape")[0]
    assert y in (1.0, 3.0)
    from diff_gpt.metrics.point import _expected_smape
    assert _expected_smape(s[:, 0], np.array([y]))[0] < _expected_smape(s[:, 0], np.array([2.0]))[0]


def test_smape_optimum_moves_above_the_median_when_spread_is_large_vs_level():
    rng = np.random.default_rng(0)
    s = rng.normal(2.0, 1.0, size=(4000, 1))  # spread comparable to the level
    med = bayes_point(s, "mae")[0]
    sm = bayes_point(s, "smape")[0]
    assert sm > med * 1.01


def test_owa_point_lies_between_median_and_smape_point():
    rng = np.random.default_rng(0)
    s = rng.normal(2.0, 1.0, size=(4000, 1))
    med = bayes_point(s, "mae")[0]
    sm = bayes_point(s, "smape")[0]
    owa = bayes_point(s, "owa", naive_scale=1.0, smape_ref=18.383, mase_ref=2.395)[0]
    assert min(med, sm) - 1e-9 <= owa <= max(med, sm) + 1e-9


def test_shapes_and_per_column_naive_scale():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(50, 6, 2)) + 5.0  # (N, H, F)
    for metric in ("mae", "mse", "smape"):
        assert bayes_point(s, metric).shape == (6, 2)
    out = bayes_point(s, "owa", naive_scale=np.array([1.0, 100.0]), smape_ref=18.383, mase_ref=2.395)
    assert out.shape == (6, 2)
    assert np.isfinite(out).all()


def test_degenerate_samples_return_that_value():
    s = np.full((10, 3), 2.5)
    for metric in ("mae", "mse", "smape"):
        assert np.array_equal(bayes_point(s, metric), np.full(3, 2.5))


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        bayes_point(np.zeros((2, 1)), "nope")
