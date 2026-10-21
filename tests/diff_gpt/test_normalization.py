"""Per-window normalization: the statistics of each method, NaN and fallbacks."""

import numpy as np
import pytest
import torch

from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.model.gpt import GPT
from diff_gpt.normalization import (
    NORMALIZATIONS,
    WindowScale,
    denormalize,
    normalize,
    predict_samples_normalized,
    window_scale,
)
from diff_gpt.sampler.temperature import TemperatureSampler

# Two columns: [1, 3, 2, 6] has the mean 3, the mean |x| 3, the standard
# deviation √3.5 and the mean |step| 7/3; [-2, -4, -4, -6] has the mean -4,
# the mean |x| 4, the standard deviation √2 and the mean |step| 4/3.
CONTEXT = np.array([[1.0, -2.0], [3.0, -4.0], [2.0, -4.0], [6.0, -6.0]])
EXPECTED = {
    "mean_abs": WindowScale(loc=np.array([0.0, 0.0]), scale=np.array([3.0, 4.0])),
    "z": WindowScale(loc=np.array([3.0, -4.0]), scale=np.sqrt([3.5, 2.0])),
    "mean_abs_step": WindowScale(
        loc=np.array([3.0, -4.0]),
        scale=np.array([7 / 3, 4 / 3]),
    ),
}


@pytest.mark.parametrize("method", NORMALIZATIONS)
def test_window_scale_reads_each_method(method: str) -> None:
    """Check the location and the scale of every method on a known context."""
    window = window_scale(CONTEXT, method)
    np.testing.assert_allclose(window.loc, EXPECTED[method].loc)
    np.testing.assert_allclose(window.scale, EXPECTED[method].scale)


@pytest.mark.parametrize("method", NORMALIZATIONS)
def test_denormalize_inverts_normalize(method: str) -> None:
    """Check that values in the window's units come back in the series' units."""
    window = window_scale(CONTEXT, method)
    values = 10.0 * np.random.default_rng(0).normal(size=(5, 2))
    np.testing.assert_allclose(denormalize(normalize(values, window), window), values)


@pytest.mark.parametrize(
    ("method", "shift"),
    [("mean_abs", 0.0), ("z", 1e4), ("mean_abs_step", 1e4)],
)
def test_normalized_context_is_free_of_the_units(method: str, shift: float) -> None:
    """
    Check that the context a·x + b, a > 0, normalizes to the values of x.

    "mean_abs" keeps the location 0, so it frees the unit a only.
    """
    other = 250.0 * CONTEXT + shift
    np.testing.assert_allclose(
        normalize(other, window_scale(other, method)),
        normalize(CONTEXT, window_scale(CONTEXT, method)),
        atol=1e-9,
    )


def test_window_scale_reads_only_observed_values() -> None:
    """
    Check that a NaN counts for nothing, and that a step needs both its values.

    [1, NaN, 2, 6, 7] has the mean 4 of its four values and the mean 2.5 of
    its two steps, 2 → 6 and 6 → 7.
    """
    context = np.array([[1.0], [np.nan], [2.0], [6.0], [7.0]])
    window = window_scale(context, "mean_abs_step")
    np.testing.assert_allclose(window.loc, [4.0])
    np.testing.assert_allclose(window.scale, [2.5])


@pytest.mark.parametrize("method", NORMALIZATIONS)
def test_window_scale_falls_back_to_the_level_then_to_one(method: str) -> None:
    """
    Check the scale of the contexts that have none: it is never 0.

    A constant column takes its mean |x|, a column of zeros and one of NaN
    take 1, and a lone value, which has no step, takes its |x|. The column of
    NaN keeps the location 0.
    """
    nan = np.nan
    context = np.array(
        [[5.0, 0.0, nan, nan], [5.0, 0.0, nan, -4.0], [5.0, 0.0, nan, nan]],
    )
    window = window_scale(context, method)
    np.testing.assert_allclose(window.scale, [5.0, 1.0, 1.0, 4.0])
    assert window.loc[2] == 0.0


def test_window_scale_rejects_bad_contexts() -> None:
    """An unknown method, a context that is not (T, F) and an infinity raise."""
    with pytest.raises(ValueError, match="unknown normalization"):
        window_scale(CONTEXT, "max")
    with pytest.raises(ValueError, match=r"shape \(T, F\)"):
        window_scale(CONTEXT[:, 0], "z")
    with pytest.raises(ValueError, match="infinity"):
        window_scale(np.array([[1.0], [np.inf]]), "z")


def test_samples_come_back_in_the_units_of_each_context() -> None:
    """
    Check that the samples of 1024·x are 1024 times those of x, in input order.

    A power of 2 scales every statistic exactly, so the normalized contexts,
    their tokens and, under one seed, their samples are the same. A context
    with a gap, of another length, forecasts too.
    """
    torch.manual_seed(0)
    model = DiffGPT(
        model=GPT(vocab_size=64, n_embd=16, block_size=64, n_head=2, n_layer=1),
        order_of_derivative=1,
        domain_of_definition=np.array([4.0]),
    )
    walk = np.cumsum(np.random.default_rng(0).normal(size=(30, 1)), axis=0)
    gappy = walk[:20].copy()
    gappy[5:8] = np.nan
    sampler = TemperatureSampler(temperature=1.0)
    small, filled = predict_samples_normalized(
        model,
        [walk, gappy],
        max_new_points=4,
        num_samples=3,
        sampler=sampler,
    )
    (large,) = predict_samples_normalized(
        model,
        [1024.0 * walk],
        max_new_points=4,
        num_samples=3,
        sampler=sampler,
    )
    assert small.shape == (3, 4, 1)
    assert filled.shape == (3, 4, 1)
    assert np.isfinite(filled).all()
    assert not np.array_equal(small[0], small[1])
    np.testing.assert_array_equal(large, 1024.0 * small)
