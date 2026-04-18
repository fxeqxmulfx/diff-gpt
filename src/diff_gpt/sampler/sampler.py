import torch


class Sampler:
    """Base class for token samplers."""

    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample from logits of shape `[..., n_tokens]`."""
        raise NotImplementedError()
