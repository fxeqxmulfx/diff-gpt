import torch
from torch import nn

from diff_gpt.sampler.sampler import Sampler


class NucleusSampler(Sampler):
    """Nucleus Sampler"""

    __slots__ = (
        "p",
        "sampler",
        "softmax",
    )

    def __init__(self, p: float, sampler: Sampler) -> None:
        """
        :param p: is the sum of probabilities of tokens to pick $p$
        :param sampler: is the sampler to use for the selected tokens
        """
        self.p = p
        self.sampler = sampler
        self.softmax = nn.Softmax(dim=-1)

    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample from logits with Nucleus Sampling"""
        probs = self.softmax(logits)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < self.p
        nucleus = torch.cat(
            [nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1
        )
        original_token_mask = torch.zeros_like(nucleus, dtype=torch.bool)
        original_token_mask.scatter_(-1, indices, nucleus)
        masked_logits = logits.clone()
        masked_logits[~original_token_mask] = float("-inf")
        return self.sampler(masked_logits, rng=rng)
