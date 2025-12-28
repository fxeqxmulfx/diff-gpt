import torch

from diff_gpt.sampler.sampler import Sampler


class TemperatureSampler(Sampler):
    """Sampler with Temperature"""

    __slots__ = "temperature"

    def __init__(self, temperature: float = 1.0) -> None:
        """
        :param temperature: is the temperature to sample with
        """
        self.temperature = temperature

    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample from logits"""
        temperature = self.temperature
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        result = torch.multinomial(probs, num_samples=1, generator=rng)
        return result
