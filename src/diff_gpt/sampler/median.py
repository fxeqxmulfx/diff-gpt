# Copyright 2025 Mikhail Borisov
"""Median readout of the predictive distribution."""

import torch

from diff_gpt.sampler.sampler import Sampler

# Cumulative probability mass at the median.
_MEDIAN_MASS = 0.5


class MedianSampler(Sampler):
    """
    Deterministic point readout: the median bin of the predictive distribution.

    The vocabulary is ordinal (derivative bins), so the median is the
    MAE-optimal point estimate. Argmax (``TemperatureSampler(temperature=0.0)``)
    returns the mode, which summarises a skewed or bimodal distribution poorly.
    The median bin is fed back autoregressively exactly like the argmax token.
    """

    __slots__ = ()

    def __call__(
        self,
        logits: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return the median token id along the last axis of `logits`."""
        # Deterministic: `rng` is only part of the Sampler signature.
        del rng
        cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1)
        # Smallest index whose cumulative mass reaches 1/2.
        median = (cdf < _MEDIAN_MASS).sum(dim=-1, keepdim=True)
        return median.clamp_(max=logits.size(-1) - 1)
