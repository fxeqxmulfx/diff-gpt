"""Reduce SSMax score strength without changing normalized projections or weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from research.early99.experiments.factories import (
    Configuration as ParentConfiguration,
)
from research.early99.experiments.factories import (
    Factories,
    peri_rms_scaled_ssmax_model,
)
from research.early99.experiments.scalable_softmax import ScalableSoftmaxAttention

if TYPE_CHECKING:
    import torch

    from diff_gpt.domain.model.gpt import GPT
    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case
    from research.current_grokking.train import Logits


class FixedScoreAttention(ScalableSoftmaxAttention):
    """Retain normalized SSMax/XSA and multiply its fixed per-head gain by m."""

    def __init__(self, attention: ScalableSoftmaxAttention, multiplier: float) -> None:
        """Borrow every original tensor without additional parameters or RNG."""
        if not math.isfinite(multiplier) or not 0 < multiplier <= 1:
            message = "Score multiplier must be finite and in (0, 1]"
            raise ValueError(message)
        if not attention.use_qk_norm:
            message = "This shared-model experiment requires Q/K RMS"
            raise ValueError(message)
        super().__init__(attention, normalize_qk=True)
        self.score_multiplier = multiplier

    def attention_gain(self) -> torch.Tensor:
        """Apply the declared scalar after Q/K RMS in dense and cached paths."""
        return super().attention_gain() * self.score_multiplier


def install_score_scale(model: GPT, multiplier: float) -> None:
    """Replace only the fixed attention gain of the already prepared parent."""
    for block in model.blocks:
        block.sa = FixedScoreAttention(block.sa, multiplier)


@dataclass(frozen=True)
class Configuration(ParentConfiguration):
    """Archive the extra scalar while retaining the parent's optimizer protocol."""

    name: str = "peri-rms-scaled-ssmax-gain-half-lr2"
    learning_rate_multiplier: float = 2.0
    model_variant: str = "peri-rms-embedding-scaled-ssmax"
    score_multiplier: float = 0.5

    def __post_init__(self) -> None:
        """Reject undeclared parents and invalid score settings before execution."""
        super().__post_init__()
        if self.model_variant != "peri-rms-embedding-scaled-ssmax":
            message = "The score study requires the Q/K-normalized SSMax parent"
            raise ValueError(message)
        if (
            not math.isfinite(self.score_multiplier)
            or not 0 < self.score_multiplier <= 1
        ):
            message = "Score multiplier must be finite and in (0, 1]"
            raise ValueError(message)

    def metadata(self, case: Case) -> dict:
        """Record effective SSMax s, exact gain placement and unchanged capacity."""
        result = super().metadata(case)
        result["model_factory"] = "peri_rms_scaled_ssmax_fixed_score"
        result["scalable_softmax"]["s"] = self.score_multiplier
        result["score_scale"] = {
            "multiplier": self.score_multiplier,
            "placement": "multiply fixed attention gain after Q/K RMS and rotary",
            "parent": "peri-rms-scaled-ssmax-lr2",
            "added_parameters": 0,
            "qk_rms_retained": True,
        }
        return result


def score_model(task: Task, case: Case, device: str, *, multiplier: float) -> Logits:
    """Reuse the exact parent initializer, then change only its score coefficient."""
    model = peri_rms_scaled_ssmax_model(task, case, device)
    install_score_scale(model.model, multiplier)
    return model


def configured_factories(configuration: Configuration) -> Factories:
    """Keep the verified native AdamW factory and the existing training stepper."""
    return Factories(
        model=partial(score_model, multiplier=configuration.score_multiplier)
    )
