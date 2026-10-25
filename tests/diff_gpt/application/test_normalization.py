"""Samples drawn in the units of each context (predict_samples_normalized)."""

import numpy as np
import torch

from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.application.normalization import predict_samples_normalized
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.sampler.temperature import TemperatureSampler


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
