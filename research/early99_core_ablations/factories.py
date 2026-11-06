"""Add one declared mechanism to the original model and retain native AdamW."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from research.early99.experiments.factories import (
    Configuration as OriginalConfiguration,
)
from research.early99.experiments.factories import Factories, baseline_model
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)
from research.early99.experiments.scalable_softmax import install_scalable_softmax
from research.early99_cosine.factories import FactoryStepper

if TYPE_CHECKING:
    from diff_gpt.domain.model.gpt import GPT
    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case
    from research.current_grokking.train import Logits

__all__ = [
    "Configuration",
    "FactoryStepper",
    "configured_factories",
    "install_mechanism",
]

MECHANISMS = ("embedding_rms", "scaled_peri_rms", "ssmax")
RATE_MULTIPLIER = 2.0


@dataclass(frozen=True)
class Configuration(OriginalConfiguration):
    """The original factory variant is a base; mechanism records the sole addition."""

    name: str = "core-embedding-rms-lr2"
    learning_rate_multiplier: float = RATE_MULTIPLIER
    mechanism: str = "embedding_rms"

    def __post_init__(self) -> None:
        """Reject combined mechanisms or an unannounced optimizer/schedule change."""
        super().__post_init__()
        if (
            self.mechanism not in MECHANISMS
            or self.model_variant != "baseline"
            or self.learning_rate_multiplier != RATE_MULTIPLIER
            or self.betas != (0.9, 0.98)
            or self.schedule != "constant"
        ):
            message = (
                "Use one declared original-model mechanism with constant native LR2"
            )
            raise ValueError(message)

    def metadata(self, case: Case) -> dict:
        """Describe the actual individual addition instead of the old combined stack."""
        result = super().metadata(case)
        result.update(
            model_factory=f"original_core_plus_{self.mechanism}",
            embedding_rms=self.mechanism == "embedding_rms",
            branch_output_rms=self.mechanism == "scaled_peri_rms",
            qk_rms=True,
            residual_scale=(2 * case.layers) ** -0.5
            if self.mechanism == "scaled_peri_rms"
            else 1.0,
            core_ablation={
                "mechanism": self.mechanism,
                "base": "frozen original Q/K-normalized model",
                "qk_rms_retained": True,
                "gain": "original fixed exp(log_alpha)/head_width",
                "visible_key_multiplier": "log(N)"
                if self.mechanism == "ssmax"
                else "1",
                "fixed_rope": True,
                "added_parameters": 0,
            },
            execution_journal="durable original per-update accounting",
        )
        if self.mechanism == "ssmax":
            result["scalable_softmax"] = {
                "s": 1.0,
                "bias": 0.0,
                "visible_keys": "global causal position + 1",
                "placement": "after Q/K normalization and RoPE, before masking",
            }
        return result


def install_mechanism(model: GPT, mechanism: str) -> None:
    """Reuse the exact existing modules, with no new parameters or RNG draws."""
    if mechanism == "embedding_rms":
        install_embedding_rms(model)
    elif mechanism == "scaled_peri_rms":
        install_peri_rms(model)
        install_residual_scaling(model)
    elif mechanism == "ssmax":
        install_scalable_softmax(model, normalize_qk=True)
    else:
        message = f"Undeclared architecture mechanism: {mechanism}"
        raise ValueError(message)


def core_model(task: Task, case: Case, device: str, *, mechanism: str) -> Logits:
    """Construct the old model first, then apply only the declared addition."""
    model = baseline_model(task, case, device)
    install_mechanism(model.model, mechanism)
    return model


def configured_factories(configuration: Configuration) -> Factories:
    """Keep the original native optimizer and select only the model adapter."""
    return Factories(model=partial(core_model, mechanism=configuration.mechanism))
