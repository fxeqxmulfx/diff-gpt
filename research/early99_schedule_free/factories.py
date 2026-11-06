"""Explicit optimizer mode/state handling with the original model and scorer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW
from research.current_grokking.train import Stepper
from research.early99_dimension_gain.factories import (
    Configuration as ParentConfiguration,
)

if TYPE_CHECKING:
    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case
    from research.current_grokking.train import Logits


@dataclass(frozen=True)
class Configuration(ParentConfiguration):
    """Keep model settings explicit; change optimizer and the declared peak rate."""

    name: str = "peri-rms-inverse-width-sf-adamw-lr2"
    optimizer_kind: str = "schedule_free_adamw"
    optimizer_internal_warmup: int = 0

    def __post_init__(self) -> None:
        """Reject double warmup and unsupported momentum/schedule variants."""
        super().__post_init__()
        if (
            self.optimizer_kind != "schedule_free_adamw"
            or self.optimizer_internal_warmup != 0
        ):
            message = "Only Schedule-Free AdamW with external warmup is declared"
            raise ValueError(message)
        if not 0 < self.beta1 < 1 or self.schedule != "constant":
            message = (
                "Schedule-Free requires positive beta1 and constant post-warmup rate"
            )
            raise ValueError(message)

    def metadata(self, case: Case) -> dict:
        """Distinguish averaging/query semantics from native AdamW momentum."""
        result = super().metadata(case)
        result.update(
            optimizer_factory="repo_schedule_free_adamw",
            optimizer="diff_gpt.domain.optimizer.schedule_free.ScheduleFreeAdamW",
            schedule_free={
                "training_point": "y=(1-beta1)*z+beta1*x",
                "evaluation_point": "x",
                "checkpoint_point": "x",
                "state_tensors": ["base=z", "second=v"],
                "scalar_state": ["step", "maximum", "total", "training"],
                "averaging_weight": "maximum_effective_rate_squared/running_total",
                "averaging_power": 2.0,
                "polynomial_averaging_exponent": 0.0,
                "decay_point": "y",
                "bias_correction": "second moment, with epsilon outside sqrt",
                "internal_warmup": 0,
                "mode_roundtrip": "in-place x/y conversion with retained rounding",
            },
            execution={
                "compiled_loss": "CUDA only, unchanged torch.compile options",
                "optimizer": "eager foreach, outside CUDA update graphs",
                "prepare_optimizer_updates": 0,
                "evaluation_cuda_graphs": True,
            },
        )
        return result


class ScheduleFreeStepper(Stepper):
    """Use original task loss/evaluation with explicit x/y optimizer transitions."""

    def __init__(
        self,
        model: Logits,
        task: Task,
        case: Case,
        device: str,
        *,
        configuration: Configuration,
        compiled: bool = True,
    ) -> None:
        """Preserve original loss compilation; initialize only the declared state."""
        self.configuration, self.original_case = configuration, case
        self.train_examples = len(task.splits["train"])
        effective = configuration.training_case(case)
        super().__init__(model, task, effective, device, compiled=compiled)
        # Parent AdamW construction initializes no moments and consumes no RNG.
        self.optimizer = ScheduleFreeAdamW(
            model.parameters(),
            lr=effective.learning_rate,
            betas=configuration.betas,
            weight_decay=effective.weight_decay,
            eps=1e-8,
            warmup=0,
        )

    def prepare(self, sizes: list[int]) -> None:
        """Do not issue optimizer updates or capture host-controlled averaging state."""
        if not sizes or any(size <= 0 for size in sizes):
            message = "Training batches must have positive sizes"
            raise ValueError(message)

    def step(self, indices: torch.Tensor, completed: int) -> None:
        """Recover y and issue exactly one step at the completed-update warmup rate."""
        self.model.train()
        self.optimizer.train()
        self.optimizer.param_groups[0]["lr"] = self.configuration.learning_rate_at(
            self.original_case, completed, self.train_examples
        )
        self.update(indices.to(self.device))

    @torch.no_grad()
    def evaluate(self, name: str, batch: int = 256) -> dict:
        """Score x with the original complete-split evaluator and leave x for saving."""
        self.optimizer.eval()
        return super().evaluate(name, batch)


def verify_optimizer_state(stepper: ScheduleFreeStepper, completed: int) -> None:
    """Check checkpoint point, full state, issued steps and averaging clock."""
    optimizer = stepper.optimizer
    group = optimizer.param_groups[0]
    if group["step"] != completed or group["training"] is not False:
        message = "Checkpoint must hold averaged x at the exact completed update count"
        raise AssertionError(message)
    maximum = total = 0.0
    for count in range(completed):
        rate = stepper.configuration.learning_rate_at(
            stepper.original_case, count, stepper.train_examples
        )
        maximum = max(maximum, rate)
        total += maximum**2
    if not (
        math.isclose(group["maximum"], maximum, rel_tol=1e-12, abs_tol=0)
        and math.isclose(group["total"], total, rel_tol=1e-12, abs_tol=0)
    ):
        message = "Schedule-Free averaging clock differs from issued rates"
        raise AssertionError(message)
    if set(optimizer.state) != set(group["params"]):
        message = "Missing Schedule-Free parameter state"
        raise AssertionError(message)
    for state in optimizer.state.values():
        if set(state) != {"base", "second"} or not all(
            bool(torch.isfinite(v).all()) for v in state.values()
        ):
            message = "Incomplete or nonfinite Schedule-Free state"
            raise AssertionError(message)
        if not bool((state["second"] >= 0).all()):
            message = "Negative second moment"
            raise AssertionError(message)
