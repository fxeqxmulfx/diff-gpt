# Copyright 2025 Mikhail Borisov
"""Token sampler interface."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Sampler:
    """Base class for token samplers."""

    def __call__(
        self,
        logits: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample from logits of shape `[..., n_tokens]`."""
        raise NotImplementedError
