"""
Stand-in for TempoPFN's `src.synthetic_generation.kernel_synth.kernel_synth`.

TempoPFN draws each KernelSynth series from its GP prior with scikit-learn's
GaussianProcessRegressor.sample_y, which samples with the legacy
RandomState.multivariate_normal: an SVD of the 2048 x 2048 covariance, 8.6 s
of the 9 a series takes on one thread. This is TempoPFN's
`KernelSynthGenerator` (Apache-2.0, a copy of the license in
src/diff_gpt/infrastructure/corpus/tempopfn/LICENSE-TempoPFN), changed in two
places: its import names the module of `src` by its name in
diff_gpt.infrastructure.corpus.tempopfn; `_sample_from_gp_prior` builds the
covariance from the lags of the grid (`lag_kernel_matrix`), factors it with
LAPACK's pivoted Cholesky, dpstrf, in 0.15 s, and draws from the same Gaussian
with the RandomState of the same seed. Most of the composite kernels are of low
numerical rank (a median of 295 of 2048 over 40 of them), where the
factorization stops, with no jitter. A seed gives another series than in
TempoPFN, from the same distribution: over 2304 seeds paired with the SVD, the
series differ in neither spread, lag-1 autocorrelation, increments nor range,
but on the 3% of kernels of rank 3 or less, constants and lines, where the SVD
adds round-off noise. dpstrf raises no LinAlgError, so `generate_time_series`
no longer retries with seed + 1.

On 79 roots of CauKer, whose bank this is too, `lag_kernel_matrix` took
0.015 s where sklearn's pairwise distances took 0.385, and agreed with them to
2.6e-13 of the largest entry. The CauKer stand-in draws its roots with it and
`sample_gaussian` as well.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import lapack, toeplitz
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Product,
    RationalQuadratic,
    Sum,
    WhiteKernel,
)
from sklearn.utils import check_random_state

from diff_gpt.infrastructure.corpus.tempopfn.synthetic_generation.abstract_classes import (  # noqa: E501 - the module's name alone is wider than the line.
    AbstractTimeSeriesGenerator,
)

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel


def lag_kernel_matrix(kernel: Kernel, x: np.ndarray) -> np.ndarray:
    """
    Return kernel(x[:, None]), for a sum or product of the bank's kernels.

    On a uniform grid `x`, every kernel of the bank but DotProduct depends on
    x_i - x_j alone, so its matrix is Toeplitz, and DotProduct is
    sigma_0^2 + x_i x_j. A sum or product of them is therefore
    sum_p diag(x^p) T_p diag(x^p) with each T_p Toeplitz, which this builds
    from the n lags of each T_p instead of sklearn's n^2 pairwise distances.
    The two agree up to round-off.
    """
    matrix = np.zeros((len(x), len(x)))
    for power, lags in _lag_terms(kernel, x - x[0]).items():
        term = toeplitz(lags)
        if power:
            scale = x**power
            term *= scale[:, None]
            term *= scale
        matrix += term
    return matrix


def _lag_terms(kernel: Kernel, lag: np.ndarray) -> dict[int, np.ndarray]:
    """Return {p: t_p}, kernel = sum_p diag(x^p) toeplitz(t_p) diag(x^p)."""
    if isinstance(kernel, Sum | Product):
        left, right = _lag_terms(kernel.k1, lag), _lag_terms(kernel.k2, lag)
        if isinstance(kernel, Sum):
            pairs = [*left.items(), *right.items()]
        else:
            pairs = [(p + q, t * s) for p, t in left.items() for q, s in right.items()]
        terms: dict[int, np.ndarray] = {}
        for power, lags in pairs:
            terms[power] = terms[power] + lags if power in terms else lags
        return terms
    # Leaves by their exact type: a subclass has a matrix of its own, as
    # Matern, a subclass of RBF.
    if type(kernel) is ExpSineSquared:
        sine = np.sin(np.pi * lag / kernel.periodicity) / kernel.length_scale
        return {0: np.exp(-2 * sine**2)}
    if type(kernel) is RBF:
        return {0: np.exp(-0.5 * (lag / kernel.length_scale) ** 2)}
    if type(kernel) is RationalQuadratic:
        base = 1 + lag**2 / (2 * kernel.alpha * kernel.length_scale**2)
        return {0: base**-kernel.alpha}
    if type(kernel) is DotProduct:
        return {0: np.full_like(lag, kernel.sigma_0**2), 1: np.ones_like(lag)}
    if type(kernel) is WhiteKernel:
        white = np.zeros_like(lag)
        white[0] = kernel.noise_level
        return {0: white}
    if type(kernel) is ConstantKernel:
        return {0: np.full_like(lag, kernel.constant_value)}
    msg = f"{kernel!r} is not a sum or product of the bank's kernels"
    raise TypeError(msg)


def sample_gaussian(
    cov: np.ndarray,
    rng: np.random.Generator | np.random.RandomState,
) -> np.ndarray:
    """
    Draw a sample from N(0, cov), for a positive semidefinite `cov`.

    The pivoted Cholesky factorization is P^T cov P = L L^T, with L of as many
    columns as the numerical rank of cov, and the sample is P L z, for z
    standard normal from `rng`.
    """
    factor, pivots, rank, _ = lapack.dpstrf(cov, lower=1)
    sample = np.empty(len(cov))
    sample[pivots - 1] = np.tril(factor[:, :rank]) @ rng.standard_normal(rank)
    return sample


class KernelSynthGenerator(AbstractTimeSeriesGenerator):
    """
    Generate independent synthetic univariate time series using kernel synthesis.

    Each series is sampled from a Gaussian process prior with a random composite kernel.
    """

    def __init__(
        self,
        length: int = 1024,
        max_kernels: int = 5,
        random_seed: int | None = None,
    ) -> None:
        """
        Set up the bank of base kernels for series of `length` steps.

        Parameters
        ----------
        length : int, optional
            Number of time steps per series (default: 1024).
        max_kernels : int, optional
            Maximum number of base kernels to combine (default: 5).
        random_seed : int, optional
            Seed for the random number generator.

        """
        self.length = length
        self.max_kernels = max_kernels
        self.rng = np.random.default_rng(random_seed)
        self.kernel_bank = [
            ExpSineSquared(periodicity=24 / length),  # H
            ExpSineSquared(periodicity=48 / length),  # 0.5H
            ExpSineSquared(periodicity=96 / length),  # 0.25H
            ExpSineSquared(periodicity=24 * 7 / length),  # H-week
            ExpSineSquared(periodicity=48 * 7 / length),  # 0.5H-week
            ExpSineSquared(periodicity=96 * 7 / length),  # 0.25H-week
            ExpSineSquared(periodicity=7 / length),  # day
            ExpSineSquared(periodicity=14 / length),  # 0.5-day
            ExpSineSquared(periodicity=30 / length),  # day
            ExpSineSquared(periodicity=60 / length),  # 0.5-day
            ExpSineSquared(periodicity=365 / length),  # year
            ExpSineSquared(periodicity=365 * 2 / length),  # 0.5-year
            ExpSineSquared(periodicity=4 / length),  # week
            ExpSineSquared(periodicity=26 / length),  # week
            ExpSineSquared(periodicity=52 / length),  # week
            ExpSineSquared(periodicity=4 / length),  # month
            ExpSineSquared(periodicity=6 / length),  # month
            ExpSineSquared(periodicity=12 / length),  # month
            ExpSineSquared(periodicity=4 / length),  # quarter
            ExpSineSquared(periodicity=4 * 10 / length),  # quarter
            ExpSineSquared(periodicity=10 / length),  # year
            DotProduct(sigma_0=0.0),
            DotProduct(sigma_0=1.0),
            DotProduct(sigma_0=10.0),
            RBF(length_scale=0.1),
            RBF(length_scale=1.0),
            RBF(length_scale=10.0),
            RationalQuadratic(alpha=0.1),
            RationalQuadratic(alpha=1.0),
            RationalQuadratic(alpha=10.0),
            WhiteKernel(noise_level=0.1),
            WhiteKernel(noise_level=1.0),
            ConstantKernel(),
        ]

    def _random_binary_map(self, a: Kernel, b: Kernel) -> Kernel:
        """Randomly combine two kernels with + or *."""
        ops = [lambda x, y: x + y, lambda x, y: x * y]
        return self.rng.choice(ops)(a, b)

    def _sample_from_gp_prior(
        self,
        kernel: Kernel,
        X: np.ndarray,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Draw a sample from the GP prior, N(0, kernel(X)), as sample_y does."""
        cov = lag_kernel_matrix(kernel, np.ravel(X))
        return sample_gaussian(cov, check_random_state(random_seed))

    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """
        Generate a single independent univariate time series.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducible generation.

        Returns
        -------
        np.ndarray
            Shape: [seq_len]

        """
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)

        X = np.linspace(0, 1, self.length)
        num_kernels = self.rng.integers(1, self.max_kernels + 1)
        selected = self.rng.choice(self.kernel_bank, num_kernels, replace=True)
        composite = functools.reduce(self._random_binary_map, selected)
        try:
            values = self._sample_from_gp_prior(composite, X, random_seed=random_seed)
        except np.linalg.LinAlgError:
            new_seed = (random_seed + 1) if random_seed is not None else None
            return self.generate_time_series(new_seed)

        return values
