"""Retain the learned activation and change its declared scalar initialization."""

import torch

from research.transformer_stress_parity.models import ConfiguredNetwork


class InitializedNetwork(ConfiguredNetwork):
    """Keep all matrices and operations, initializing the learned even coefficient."""

    def __init__(self, settings: dict, depth: int, kind: str) -> None:
        """Use one common initial value in every branch and both teacher tasks."""
        super().__init__(settings, depth, kind)
        with torch.no_grad():
            for branch in self.branches:
                branch.even_gain.fill_(settings["even_gain_initial"])
