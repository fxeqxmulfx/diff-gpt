"""Explicit model and optimizer construction, without changing the baseline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

import torch

from diff_gpt.domain.model.gpt import GPT
from research.current_grokking.train import Logits, Stepper
from research.early99.experiments.causal_centering import install_causal_centering
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)
from research.early99.protocol import exposure

if TYPE_CHECKING:
    from collections.abc import Callable

    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case


@dataclass(frozen=True)
class Configuration:
    """One task-independent candidate setting; data and scoring are immutable."""

    name: str = "baseline-factory"
    learning_rate_multiplier: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    model_variant: str = "baseline"
    schedule: str = "constant"
    decay_epochs: float = 200.0
    minimum_lr_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Reject unsafe output names and invalid rates before allocating a model."""
        if not self.name or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in self.name
        ):
            message = "Configuration name allows lowercase letters, digits and hyphens"
            raise ValueError(message)
        if (
            not math.isfinite(self.learning_rate_multiplier)
            or self.learning_rate_multiplier <= 0
        ):
            message = "Learning-rate multiplier must be finite and positive"
            raise ValueError(message)
        if not all(math.isfinite(beta) and 0 <= beta < 1 for beta in self.betas):
            message = "AdamW moment coefficients must be finite and in [0, 1)"
            raise ValueError(message)
        if self.model_variant not in {
            "baseline",
            "peri-rms",
            "peri-rms-embedding",
            "peri-rms-embedding-scaled",
            "peri-rms-embedding-scaled-csa",
        }:
            message = "Unknown model variant"
            raise ValueError(message)
        if self.schedule not in {"constant", "cosine"}:
            message = "Unknown learning-rate schedule"
            raise ValueError(message)
        if not math.isfinite(self.decay_epochs) or self.decay_epochs <= 0:
            message = "Cosine duration must be finite and positive"
            raise ValueError(message)
        if (
            not math.isfinite(self.minimum_lr_ratio)
            or not 0 < self.minimum_lr_ratio <= 1
        ):
            message = "Minimum learning-rate ratio must be finite and in (0, 1]"
            raise ValueError(message)

    @property
    def betas(self) -> tuple[float, float]:
        """Return the exact coefficients passed to native AdamW."""
        return self.beta1, self.beta2

    def training_case(self, case: Case) -> Case:
        """Override only the allowed optimizer field, leaving the task intact."""
        return replace(
            case, learning_rate=case.learning_rate * self.learning_rate_multiplier
        )

    def learning_rate_at(
        self, case: Case, completed: int, train_examples: int
    ) -> float:
        """Warm up by issued updates, then decay by actual post-warmup exposure."""
        peak = case.learning_rate * self.learning_rate_multiplier
        warmup = min(1.0, completed / case.warmup) if case.warmup else 1.0
        if self.schedule == "constant" or completed <= case.warmup:
            return peak * warmup
        epochs = exposure(completed, train_examples, case.batch)["epochs"]
        start = exposure(case.warmup, train_examples, case.batch)["epochs"]
        progress = min((epochs - start) / self.decay_epochs, 1.0)
        decay = (
            self.minimum_lr_ratio
            + (1 - self.minimum_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )
        return peak * decay

    def metadata(self, case: Case) -> dict:
        """Record all effective optimizer and schedule hyperparameters."""
        effective = self.training_case(case)
        return {
            "configuration": asdict(self),
            "model_factory": {
                "baseline": "frozen_public_gpt",
                "peri-rms": "peri_rms_public_gpt",
                "peri-rms-embedding": "peri_rms_embedding_public_gpt",
                "peri-rms-embedding-scaled": "peri_rms_embedding_scaled_public_gpt",
                "peri-rms-embedding-scaled-csa": (
                    "peri_rms_embedding_scaled_csa_public_gpt"
                ),
            }[self.model_variant],
            "residual_scale": (2 * case.layers) ** -0.5
            if self.model_variant
            in {"peri-rms-embedding-scaled", "peri-rms-embedding-scaled-csa"}
            else 1.0,
            **(
                {
                    "attention_centering": {
                        "kind": "causal-prefix-mean",
                        "coefficient": 1.0,
                        "placement": "after softmax value averaging, before XSA",
                    }
                }
                if self.model_variant == "peri-rms-embedding-scaled-csa"
                else {}
            ),
            "optimizer_factory": "native_adamw",
            "optimizer": "torch.optim.AdamW",
            "learning_rate": effective.learning_rate,
            "betas": list(self.betas),
            "epsilon": 1e-8,
            "weight_decay": effective.weight_decay,
            "warmup_updates": effective.warmup,
            "warmup_uses_completed_updates": True,
            "schedule": "linear warmup from zero, then constant"
            if self.schedule == "constant"
            else (
                "linear warmup from zero, then cosine decay "
                "by actual epochs to a fixed floor"
            ),
            "dtype": "torch.float32",
            "matmul_precision": "highest",
            "rms_backward": "unchanged FP64 calculation",
        }


def baseline_model(task: Task, case: Case, device: str) -> Logits:
    """Preserve public-GPT construction, initialization and parameter ordering."""
    return Logits(
        GPT(
            task.vocab,
            case.width,
            block_size=task.context,
            n_head=case.heads,
            n_layer=case.layers,
            use_checkpoint=False,
            pad_vocab_size_to=1,
        ).to(device)
    )


def peri_rms_model(task: Task, case: Case, device: str) -> Logits:
    """Add parameter-free branch output RMS to the matched baseline weights."""
    model = baseline_model(task, case, device)
    install_peri_rms(model.model)
    return model


def peri_rms_embedding_model(task: Task, case: Case, device: str) -> Logits:
    """Add embedding RMS to the branch-normalized model with the same weights."""
    model = peri_rms_model(task, case, device)
    install_embedding_rms(model.model)
    return model


def peri_rms_scaled_model(task: Task, case: Case, device: str) -> Logits:
    """Add fixed depth scaling after the embedding-normalized Peri-RMS branches."""
    model = peri_rms_embedding_model(task, case, device)
    install_residual_scaling(model.model)
    return model


def peri_rms_scaled_csa_model(task: Task, case: Case, device: str) -> Logits:
    """Add causal prefix centering to the unchanged scaled Peri-RMS parent."""
    model = peri_rms_scaled_model(task, case, device)
    install_causal_centering(model.model)
    return model


def native_adamw(
    model: Logits,
    rate: torch.Tensor,
    case: Case,
    device: str,
    *,
    betas: tuple[float, float] = (0.9, 0.98),
) -> torch.optim.AdamW:
    """Construct native AdamW with explicit moments and baseline execution settings."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=rate,
        betas=betas,
        weight_decay=case.weight_decay,
        eps=1e-8,
        capturable=device == "cuda",
        fused=device == "cuda",
        foreach=False if device != "cuda" else None,
    )


@dataclass(frozen=True)
class Factories:
    """Inject model/optimizer construction explicitly into candidate training."""

    model: Callable = baseline_model
    optimizer: Callable = native_adamw


def configured_factories(configuration: Configuration) -> Factories:
    """Select an explicit model factory from the archived configuration."""
    return Factories(
        model={
            "baseline": baseline_model,
            "peri-rms": peri_rms_model,
            "peri-rms-embedding": peri_rms_embedding_model,
            "peri-rms-embedding-scaled": peri_rms_scaled_model,
            "peri-rms-embedding-scaled-csa": peri_rms_scaled_csa_model,
        }[configuration.model_variant]
    )


class FactoryStepper(Stepper):
    """Reuse verified capture, evaluation and updates with an explicit optimizer."""

    def __init__(
        self,
        model: Logits,
        task: Task,
        case: Case,
        device: str,
        *,
        configuration: Configuration,
        factories: Factories,
        compiled: bool = True,
    ) -> None:
        """Set the effective schedule before the original update path is built."""
        effective = configuration.training_case(case)
        self.configuration, self.original_case = configuration, case
        self.train_examples = len(task.splits["train"])
        super().__init__(model, task, effective, device, compiled=compiled)
        # AdamW construction consumes no updates or RNG. The original Stepper
        # owns capture and evaluation; this factory owns the scientific optimizer.
        self.optimizer = factories.optimizer(
            model, self.rate, effective, device, betas=configuration.betas
        )

    def step(self, indices: torch.Tensor, completed: int) -> None:
        """Set the declared schedule before replaying one verified update."""
        if self.configuration.schedule == "constant":
            super().step(indices, completed)
            return
        rate = self.configuration.learning_rate_at(
            self.original_case, completed, self.train_examples
        )
        self.rate.fill_(rate)
        if self.device == "cuda":
            graph, index, _ = self.graphs[len(indices)]
            index.copy_(indices.to(self.device))
            graph.replay()
        else:
            self.model.train()
            self.update(indices)
