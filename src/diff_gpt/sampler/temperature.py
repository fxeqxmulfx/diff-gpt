"""Temperature sampling of the next token."""

import torch

from diff_gpt.sampler.sampler import Sampler


class TemperatureSampler(Sampler):
    """Sampler with Temperature."""

    __slots__ = ("temperature",)

    def __init__(self, temperature: float = 1.0) -> None:
        """:param temperature: is the temperature to sample with"""
        self.temperature = temperature

    def __call__(
        self,
        logits: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one token id per row of `logits` ([..., n_tokens] -> [..., 1])."""
        temperature = self.temperature
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = torch.softmax(logits / temperature, dim=-1)
        # torch.multinomial takes 1-D or 2-D input: flatten the leading dims.
        flat = probs.reshape(-1, probs.size(-1))
        ids = torch.multinomial(flat, num_samples=1, generator=rng)
        return ids.reshape(*probs.shape[:-1], 1)
