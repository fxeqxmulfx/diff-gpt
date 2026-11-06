"""Use one inverse head-width score rule without changing learned tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from research.early99.experiments.factories import (
    Configuration as ParentConfiguration,
)
from research.early99.experiments.factories import (
    Factories,
    peri_rms_scaled_ssmax_model,
)
from research.early99.experiments.scalable_softmax import ScalableSoftmaxAttention

if TYPE_CHECKING:
    from diff_gpt.domain.model.gpt import GPT
    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case
    from research.current_grokking.train import Logits


class InverseWidthAttention(ScalableSoftmaxAttention):
    """Use log(N)/d with the existing RMS, rotary, centered XSA and cached paths."""

    def __init__(self, attention: ScalableSoftmaxAttention) -> None:
        """Keep all original tensors and add only a deterministic fixed buffer."""
        if not attention.use_qk_norm:
            message = "Inverse-width study requires Q/K RMS"
            raise ValueError(message)
        super().__init__(attention, normalize_qk=True)
        self.register_buffer(
            "inverse_width_gain",
            torch.full_like(self.log_alpha, 1 / self.head_dim),
            persistent=False,
        )

    def attention_gain(self) -> torch.Tensor:
        """Preserve the original log_alpha buffer while using the declared 1/d."""
        return self.inverse_width_gain.view(1, self.n_head, 1, 1)


def install_inverse_width(model: GPT) -> None:
    """Replace only the fixed gain after constructing the unchanged parent."""
    for block in model.blocks:
        block.sa = InverseWidthAttention(block.sa)


@dataclass(frozen=True)
class Configuration(ParentConfiguration):
    """Record the common width rule and retain the parent's optimizer protocol."""

    name: str = "peri-rms-scaled-ssmax-inverse-width-lr2"
    learning_rate_multiplier: float = 2.0
    model_variant: str = "peri-rms-embedding-scaled-ssmax"
    gain_rule: str = "inverse_head_width"

    def __post_init__(self) -> None:
        """Reject a different parent or gain rule before constructing any model."""
        super().__post_init__()
        if (
            self.model_variant != "peri-rms-embedding-scaled-ssmax"
            or self.gain_rule != "inverse_head_width"
        ):
            message = "Only Q/K-normalized SSMax with inverse_head_width is declared"
            raise ValueError(message)

    def metadata(self, case: Case) -> dict:
        """State the exact coefficient, retained Q/K RMS and unchanged capacity."""
        result = super().metadata(case)
        result["model_factory"] = "peri_rms_scaled_ssmax_inverse_width"
        result["dimension_gain"] = {
            "rule": self.gain_rule,
            "head_width": case.width // case.heads,
            "fixed_gain": 1 / (case.width // case.heads),
            "placement": "after Q/K RMS and rotary, with original log(visible keys)",
            "parent": "peri-rms-scaled-ssmax-gain-quarter-lr2",
            "added_parameters": 0,
            "qk_rms_retained": True,
            "original_log_alpha_buffer_preserved": True,
        }
        return result


def dimension_model(task: Task, case: Case, device: str) -> Logits:
    """Reuse the exact parent initializer without consuming any additional RNG."""
    model = peri_rms_scaled_ssmax_model(task, case, device)
    install_inverse_width(model.model)
    return model


def configured_factories(configuration: Configuration) -> Factories:
    """Keep native AdamW and the existing trainer and stepper unchanged."""
    if configuration.gain_rule != "inverse_head_width":
        message = "Undeclared gain rule"
        raise ValueError(message)
    return Factories(model=dimension_model)
