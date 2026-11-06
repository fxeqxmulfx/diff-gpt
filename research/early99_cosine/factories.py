"""Retain native optimizer/model construction and record only real issued updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from research.early99.experiments.factories import FactoryStepper as OriginalStepper
from research.early99_dimension_gain.factories import (
    Configuration as ParentConfiguration,
)
from research.early99_dimension_gain.factories import configured_factories
from research.early99_execution.journal import UpdateJournal, recorded_step

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case
    from research.current_grokking.train import Logits
    from research.early99.experiments.factories import Factories

__all__ = ["Configuration", "FactoryStepper", "configured_factories"]


@dataclass(frozen=True)
class Configuration(ParentConfiguration):
    """Reuse the exact existing cosine formula with the newly declared floor."""

    name: str = "peri-rms-inverse-width-adamw-cosine200-floor01-lr2"
    schedule: str = "cosine"
    decay_epochs: float = 200.0
    minimum_lr_ratio: float = 0.1


class FactoryStepper(OriginalStepper):
    """Retain original capture/update math and bracket genuine issued steps."""

    def __init__(
        self,
        model: Logits,
        task: Task,
        case: Case,
        device: str,
        *,
        configuration: Configuration,
        factories: Factories,
        journal_path: Path,
        compiled: bool = True,
    ) -> None:
        """Keep preparation untouched and defer journal writes until training."""
        super().__init__(
            model,
            task,
            case,
            device,
            configuration=configuration,
            factories=factories,
            compiled=compiled,
        )
        self.execution_journal = UpdateJournal(
            journal_path, self.train_examples, case.batch
        )

    def step(self, indices: torch.Tensor, completed: int) -> None:
        """Issue one unchanged native update between durable execution markers."""
        recorded_step(
            self.execution_journal, indices, completed, self.device, super().step
        )
