"""The stand-in CauKer: TempoPFN's wrapper takes it; its roots are independent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from src.synthetic_generation.generator_params import CauKerGeneratorParams

from src import TEMPOPFN_ROOT
from src.synthetic_generation.cauker import cauker_generator, cauker_generator_wrapper
from tempopfn_corpus import sources

LENGTH = 2048
# Two roots of the SCM of this seed, a series of the pilot's holdout, are of
# kernel 4 I and zero mean: TempoPFN's one noise made them one series.
TWIN_SEED = 402653769
TWINS = (4, 5)
TOLERANCE = 5  # standard errors


def test_tempopfn_wrapper_takes_the_stand_in() -> None:
    """TempoPFN's wrapper, from the clone, builds its generators from here."""
    wrapper = Path(cauker_generator_wrapper.__file__).resolve()
    assert wrapper.is_relative_to(TEMPOPFN_ROOT.resolve())
    stand_in = Path(cauker_generator.__file__).resolve()
    assert stand_in.is_relative_to(sources.STAND_INS)
    assert cauker_generator_wrapper.CauKerGenerator is cauker_generator.CauKerGenerator


def test_roots_of_one_kernel_are_independent() -> None:
    """
    Two roots of kernel 4 I and zero mean are two independent white noises.

    Each has the standard deviation 2 and the two are uncorrelated, to five
    standard errors, where TempoPFN made them one series.
    """
    generator = cauker_generator.CauKerGenerator(CauKerGeneratorParams())
    nodes = generator.generate_scm_time_series(
        time_length=LENGTH, max_parents=3, seed=TWIN_SEED, num_nodes=6
    )
    first, second = (nodes[root] for root in TWINS)
    for series in (first, second):
        assert abs(series.std() - 2) <= TOLERANCE * 2 / np.sqrt(2 * LENGTH)
    assert abs(np.corrcoef(first, second)[0, 1]) <= TOLERANCE / np.sqrt(LENGTH)


def test_series_follow_their_seed() -> None:
    """A seed gives one series of (length, 6), another seed another series."""
    generator = cauker_generator.CauKerGenerator(CauKerGeneratorParams(length=256))
    first, again, other = (generator.generate_time_series(seed) for seed in (7, 7, 8))
    assert first.shape == (256, 6)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
