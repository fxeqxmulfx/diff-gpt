"""The stand-in KernelSynth: TempoPFN's wrapper takes it; it samples the GP prior."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    ExpSineSquared,
    WhiteKernel,
)

from src import TEMPOPFN_ROOT
from src.synthetic_generation.kernel_synth import kernel_generator_wrapper, kernel_synth
from tempopfn_corpus import sources

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel

X = np.linspace(0, 1, 24)[:, None]
SAMPLES = 4000
TOLERANCE = 5  # standard errors


def test_tempopfn_wrapper_takes_the_stand_in() -> None:
    """TempoPFN's wrapper, from the clone, builds its generators from here."""
    wrapper = Path(kernel_generator_wrapper.__file__).resolve()
    assert wrapper.is_relative_to(TEMPOPFN_ROOT.resolve())
    assert Path(kernel_synth.__file__).resolve().is_relative_to(sources.STAND_INS)
    generator = kernel_generator_wrapper.KernelSynthGenerator
    assert generator is kernel_synth.KernelSynthGenerator


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(length_scale=1.0) + DotProduct(),
        ExpSineSquared(periodicity=0.3) + WhiteKernel(noise_level=0.1),
    ],
    ids=["rank-9", "full-rank"],
)
def test_samples_have_the_moments_of_the_prior(kernel: Kernel) -> None:
    """Over seeds, the samples have mean 0 and covariance K, to 5 standard errors."""
    cov = kernel(X)
    samples = np.stack(
        [kernel_synth.sample_gaussian(cov, seed) for seed in range(SAMPLES)]
    )
    variance = np.diag(cov)
    mean_error = np.abs(samples.mean(axis=0))
    assert np.all(mean_error <= TOLERANCE * np.sqrt(variance / SAMPLES))
    # Var(x_i x_j) = K_ii K_jj + K_ij^2 for a centered Gaussian.
    standard_error = np.sqrt((np.outer(variance, variance) + cov**2) / SAMPLES)
    cov_error = np.abs(samples.T @ samples / SAMPLES - cov)
    assert np.all(cov_error <= TOLERANCE * standard_error)


def test_a_rank_one_prior_gives_lines_through_the_origin() -> None:
    """DotProduct(sigma_0=0) is x x^T: each sample is z x, with no jitter added."""
    cov = DotProduct(sigma_0=0.0)(X)
    for seed in range(8):
        sample = kernel_synth.sample_gaussian(cov, seed)
        np.testing.assert_allclose(sample, sample[-1] * X[:, 0], rtol=0, atol=1e-12)


def test_series_follow_their_seed() -> None:
    """A seed gives one series of `length` steps, another seed another series."""
    generator = kernel_synth.KernelSynthGenerator(length=64)
    first, again, other = (generator.generate_time_series(seed) for seed in (7, 7, 8))
    assert first.shape == (64,)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
