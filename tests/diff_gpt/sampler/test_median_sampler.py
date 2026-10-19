# Copyright 2025 Mikhail Borisov
"""Tests for the median sampler."""

import torch

from diff_gpt.sampler.median import MedianSampler
from diff_gpt.sampler.temperature import TemperatureSampler

# Expected median bins of the hand-built distributions in the tests below.
KNOWN_MEDIAN_BIN = 2
SKEWED_MEDIAN_BIN = 2


def test_median_of_known_distribution() -> None:
    """Check that a known 4-bin distribution gives median bin 2 with shape (1, 1)."""
    # probs: 0.1, 0.2, 0.3, 0.4 -> cdf 0.1, 0.3, 0.6, 1.0 -> median bin = 2
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    out = MedianSampler()(probs.log())
    assert out.shape == (1, 1)
    assert out.item() == KNOWN_MEDIAN_BIN


def test_median_differs_from_argmax_on_skewed_distribution() -> None:
    """Check that on a skewed distribution argmax gives bin 0 and the median bin 2."""
    # Mode at bin 0 but most of the mass sits on the right.
    # cdf 0.30, 0.45, 0.70: median bin 2
    probs = torch.tensor([[0.3, 0.15, 0.25, 0.2, 0.1]])
    logits = probs.log()
    assert TemperatureSampler(0.0)(logits).item() == 0
    assert MedianSampler()(logits).item() == SKEWED_MEDIAN_BIN


def test_median_equals_argmax_for_symmetric_peaked_distribution() -> None:
    """Check that median and argmax agree on a symmetric peaked distribution."""
    logits = torch.zeros(3, 64)
    logits[:, 20] = 10.0
    logits[:, 19] = logits[:, 21] = 5.0
    assert torch.equal(MedianSampler()(logits), TemperatureSampler(0.0)(logits))


def test_median_batch_shape_and_range() -> None:
    """Check that batched medians have shape (B, 1), dtype int64 and ids in [0, V)."""
    torch.manual_seed(0)
    B, V = 8, 256
    logits = torch.randn(B, V)
    out = MedianSampler()(logits)
    assert out.shape == (B, 1)
    assert out.dtype == torch.int64
    assert (out >= 0).all()
    assert (out < V).all()
