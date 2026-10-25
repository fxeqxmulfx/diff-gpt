"""One-step unit-grid readout with a known finite-field output support."""

from __future__ import annotations

import math

import torch

from diff_gpt.domain.sampler.sampler import Sampler

MATRIX_DIMENSIONS = 2


class CanonicalFieldSampler(Sampler):
    """
    Pool probabilities of equivalent original residues and emit a legal token.

    Only observed past rows, the difference order and the known support enter
    this readout. It does not evaluate a recurrence. The integer grid must
    have scale one; the sampler is bound to one batch and one forecast step.
    """

    def __init__(
        self,
        past: torch.Tensor,
        order: int,
        vocab_size: int,
        modulus: int,
        lower: int = 0,
    ) -> None:
        """Build the decoder's residue classes from observed rows only."""
        if order < 1 or past.ndim != MATRIX_DIMENSIONS or past.shape[1] < order:
            msg = "field readout needs enough observed rows for a positive order"
            raise ValueError(msg)
        self.base = torch.zeros((len(past), 1), dtype=torch.int64, device=past.device)
        for lag in range(1, order + 1):
            coefficient = (-1) ** lag * math.comb(order, lag)
            self.base -= coefficient * past[:, -lag, None]
        self.center = vocab_size // 2
        self.lower = lower
        self.modulus = modulus
        tokens = torch.arange(vocab_size, device=past.device)[None, :]
        self.residues = (tokens - self.center + self.base).remainder(modulus)

    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        """Choose a canonical value by its total model probability."""
        del rng
        if logits.shape != self.residues.shape:
            msg = "field readout is bound to one batch and one forecast step"
            raise ValueError(msg)
        return self.choose(logits.softmax(-1))

    def choose(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Return a derivative token from an explicit token distribution."""
        mass = probabilities.new_zeros((len(probabilities), self.modulus))
        mass.scatter_add_(1, self.residues, probabilities)
        if self.lower:
            mass[:, : self.lower] = -1
        value = mass.argmax(-1, keepdim=True)
        return value - self.base + self.center
