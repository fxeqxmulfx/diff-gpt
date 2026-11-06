"""Explicit matrix/auxiliary rates with unchanged model, loss and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from research.current_grokking.train import Stepper
from research.early99_dimension_gain.factories import (
    Configuration as ParentConfiguration,
)
from research.early99_execution.journal import UpdateJournal, recorded_step
from research.early99_muon.optimizer import (
    COEFFICIENTS,
    EPSILON,
    MOMENTUM,
    NS_STEPS,
    MuonBundle,
)

if TYPE_CHECKING:
    from pathlib import Path

    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case
    from research.current_grokking.train import Logits

AUXILIARY_LR_MULTIPLIER = 2.0


@dataclass(frozen=True)
class Configuration(ParentConfiguration):
    """One fixed matrix partition, with separate matrix and auxiliary peak rates."""

    name: str = "peri-rms-inverse-width-muon-adamw-fp32-lr2"
    optimizer_kind: str = "muon_adamw"
    auxiliary_lr_multiplier: float = AUXILIARY_LR_MULTIPLIER
    muon_momentum: float = MOMENTUM
    muon_nesterov: bool = True
    muon_ns_steps: int = NS_STEPS
    muon_ns_coefficients: list[float] = field(
        default_factory=lambda: list(COEFFICIENTS)
    )
    muon_eps: float = EPSILON
    muon_shape_scaling: str = "match_rms_adamw"
    muon_compute_dtype: str = "float32"
    muon_normalization_dtype: str = "float64"

    def __post_init__(self) -> None:
        """Permit only the exact preregistered algorithm and constant schedule."""
        super().__post_init__()
        if (
            self.optimizer_kind not in {"muon_adamw", "muon"}
            or self.auxiliary_lr_multiplier != AUXILIARY_LR_MULTIPLIER
            or self.betas != (0.9, 0.98)
            or self.schedule != "constant"
            or self.muon_momentum != MOMENTUM
            or not self.muon_nesterov
            or self.muon_ns_steps != NS_STEPS
            or self.muon_ns_coefficients != list(COEFFICIENTS)
            or self.muon_eps != EPSILON
            or self.muon_shape_scaling != "match_rms_adamw"
            or self.muon_compute_dtype != "float32"
            or self.muon_normalization_dtype != "float64"
        ):
            message = "Configuration changes the declared Muon rule"
            raise ValueError(message)

    def auxiliary_rate_at(self, case: Case, completed: int) -> float:
        """Retain the parent AdamW rate for the hybrid embedding/readout."""
        warmup = min(1.0, completed / case.warmup) if case.warmup else 1.0
        return case.learning_rate * self.auxiliary_lr_multiplier * warmup

    def metadata(self, case: Case) -> dict:
        """Record the actual precision, partition, rates and update semantics."""
        result = super().metadata(case)
        result.update(
            optimizer_factory="fp32_matrix_muon_bundle",
            optimizer="research.early99_muon.optimizer.MuonBundle",
            betas=list(self.betas) if self.optimizer_kind == "muon_adamw" else None,
            epsilon=EPSILON,
            matrix_update={
                "momentum": "B=.95*B+.05*G; H=.05*G+.95*B(new)",
                "initial_normalization": "FP64 H/max(Frobenius(H),1e-7), then FP32",
                "iteration": "5 quintic steps, coefficients(3.4445,-4.775,2.0315)",
                "precision": "FP32 parameters/momentum/iterations; highest matmul",
                "direction_scaling": "0.2*sqrt(max(rows,columns))",
                "decay_rate": "unadjusted group rate, before direction update",
                "exact_polar_factor": False,
            },
            auxiliary_adamw=(
                {
                    "peak_rate": case.learning_rate * self.auxiliary_lr_multiplier,
                    "betas": list(self.betas),
                    "eps": 1e-8,
                    "weight_decay": case.weight_decay,
                    "parameters": "tied embedding/readout exactly once",
                }
                if self.optimizer_kind == "muon_adamw"
                else None
            ),
            execution={
                "compiled_loss": "unchanged CUDA torch.compile options",
                "optimizer": "eager; native fused CUDA auxiliary AdamW if hybrid",
                "prepare_optimizer_updates": 0,
                "evaluation_cuda_graphs": True,
                "durable_per_update_journal": True,
            },
        )
        return result


class MuonStepper(Stepper):
    """Issue genuine eager matrix updates with unchanged task loss and scorer."""

    def __init__(
        self,
        model: Logits,
        task: Task,
        case: Case,
        device: str,
        *,
        configuration: Configuration,
        journal_path: Path,
        compiled: bool = True,
    ) -> None:
        """Preserve original loss compilation and initialize only empty state."""
        self.configuration, self.original_case = configuration, case
        self.train_examples = len(task.splits["train"])
        effective = configuration.training_case(case)
        super().__init__(model, task, effective, device, compiled=compiled)
        # The parent's discarded AdamW constructor consumes no RNG or updates.
        self.optimizer = MuonBundle(
            model.model,
            configuration.optimizer_kind,
            matrix_lr=effective.learning_rate,
            auxiliary_lr=case.learning_rate * configuration.auxiliary_lr_multiplier,
            weight_decay=case.weight_decay,
            device=device,
        )
        self.execution_journal = UpdateJournal(
            journal_path, self.train_examples, case.batch
        )

    def prepare(self, sizes: list[int]) -> None:
        """Do not issue warmup updates or capture host-controlled optimizer state."""
        if not sizes or any(size <= 0 for size in sizes):
            message = "Training batches must have positive sizes"
            raise ValueError(message)

    def step(self, indices: torch.Tensor, completed: int) -> None:
        """Record exactly one update, including the first zero-rate step."""
        self.model.train()
        config, case = self.configuration, self.original_case
        self.optimizer.muon.param_groups[0]["lr"] = config.learning_rate_at(
            case, completed, self.train_examples
        )
        if self.optimizer.adamw is not None:
            self.optimizer.adamw.param_groups[0]["lr"] = config.auxiliary_rate_at(
                case, completed
            )
        recorded_step(
            self.execution_journal,
            indices,
            completed,
            self.device,
            lambda batch, _: self.update(batch.to(self.device)),
        )


def verify_optimizer_state(stepper: MuonStepper, completed: int) -> None:
    """Check every moment, shape, issued count, hyperparameter and last rate."""
    bundle = stepper.optimizer
    for name, optimizer in [("muon", bundle.muon), ("adamw", bundle.adamw)]:
        if optimizer is None:
            continue
        group = optimizer.param_groups[0]
        if completed and set(optimizer.state) != set(group["params"]):
            message = "Missing parameter state"
            raise AssertionError(message)
        for parameter, state in optimizer.state.items():
            keys = (
                {"step", "momentum_buffer"}
                if name == "muon"
                else {"step", "exp_avg", "exp_avg_sq"}
            )
            if set(state) != keys or int(state["step"]) != completed:
                message = "Incomplete optimizer state or wrong issued count"
                raise AssertionError(message)
            for key in keys - {"step"}:
                value = state[key]
                if value.shape != parameter.shape or not bool(
                    torch.isfinite(value).all()
                ):
                    message = "Nonfinite or wrong-shape optimizer moment"
                    raise AssertionError(message)
            if name == "adamw" and not bool((state["exp_avg_sq"] >= 0).all()):
                message = "Negative AdamW second moment"
                raise AssertionError(message)
        settings = (
            {
                "momentum": MOMENTUM,
                "nesterov": True,
                "ns_steps": NS_STEPS,
                "ns_coefficients": COEFFICIENTS,
                "eps": EPSILON,
                "adjust_lr_fn": "match_rms_adamw",
            }
            if name == "muon"
            else {"betas": (0.9, 0.98), "eps": 1e-8, "capturable": False}
        )
        settings["weight_decay"] = stepper.original_case.weight_decay
        if any(group[key] != value for key, value in settings.items()):
            message = "Optimizer hyperparameters changed"
            raise AssertionError(message)
        if completed:
            config, case = stepper.configuration, stepper.original_case
            expected = (
                config.learning_rate_at(case, completed - 1, stepper.train_examples)
                if name == "muon"
                else config.auxiliary_rate_at(case, completed - 1)
            )
            if not math.isclose(float(group["lr"]), expected, rel_tol=1e-12):
                message = "Saved rate differs from the last issued update"
                raise AssertionError(message)
