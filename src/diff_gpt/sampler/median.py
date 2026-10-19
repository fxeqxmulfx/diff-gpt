import torch

from diff_gpt.sampler.sampler import Sampler


class MedianSampler(Sampler):
    """Deterministic point readout: the median bin of the predictive distribution.

    The vocabulary is ordinal (derivative bins), so the median is the
    MAE-optimal point estimate. Argmax (``TemperatureSampler(temperature=0.0)``)
    returns the mode, which summarises a skewed or bimodal distribution poorly.
    The median bin is fed back autoregressively exactly like the argmax token.
    """

    __slots__ = ()

    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1)
        # Smallest index whose cumulative mass reaches 1/2.
        median = (cdf < 0.5).sum(dim=-1, keepdim=True)
        return median.clamp_(max=logits.size(-1) - 1)
