"""
Declared FP32 Muon with FP64 initial normalization and native auxiliary AdamW.

Uses the actual EMA rule and RMS-matching shape scaling in Torch's _muon.py.
The upstream iteration uses BF16; this study explicitly declares FP32 iterations,
widened initial normalization, and scaled-add momentum/Nesterov arithmetic.
The five-step quintic is an approximate direction, not an exact polar factor.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

MOMENTUM = 0.95
COEFFICIENTS = (3.4445, -4.775, 2.0315)
NS_STEPS = 5
EPSILON = 1e-7
MATRIX_DIMENSIONS = 2


@torch.no_grad()
def matrix_direction(momentum: torch.Tensor) -> torch.Tensor:
    """Normalize finite FP32 momentum safely, then apply the declared quintic."""
    if momentum.ndim != MATRIX_DIMENSIONS or momentum.dtype != torch.float32:
        message = "This declared Muon direction requires a real FP32 matrix"
        raise ValueError(message)
    if not bool(torch.isfinite(momentum).all()):
        message = "Nonfinite Muon momentum"
        raise FloatingPointError(message)
    wide = momentum.double()
    x = (wide / torch.linalg.vector_norm(wide).clamp_min(EPSILON)).float()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    a, b, c = COEFFICIENTS
    for _ in range(NS_STEPS):
        gram = x @ x.T
        polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, polynomial, x, beta=a)
    return x.T if transpose else x


class MatrixMuon(torch.optim.Optimizer):
    """Five-step matrix Muon, with explicit per-parameter issued-update counts."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        weight_decay: float,
    ) -> None:
        """Reject duplicates, unsupported tensors and invalid hyperparameters."""
        pairs = list(named_parameters)
        if not pairs or len({id(p) for _, p in pairs}) != len(pairs):
            message = "Muon requires nonempty, unique parameters"
            raise ValueError(message)
        if any(
            p.ndim != MATRIX_DIMENSIONS or p.dtype != torch.float32 for _, p in pairs
        ):
            message = "Pure matrix optimizer does not accept nonmatrix/non-FP32 tensors"
            raise ValueError(message)
        if not all(math.isfinite(v) and v >= 0 for v in [lr, weight_decay]):
            message = "Muon rate and decay must be finite and nonnegative"
            raise ValueError(message)
        super().__init__(
            [{"params": [p for _, p in pairs], "param_names": [n for n, _ in pairs]}],
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": MOMENTUM,
                "nesterov": True,
                "ns_steps": NS_STEPS,
                "ns_coefficients": COEFFICIENTS,
                "eps": EPSILON,
                "adjust_lr_fn": "match_rms_adamw",
            },
        )

    @torch.no_grad()
    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Issue one EMA/Nesterov, matrix direction, decay and parameter update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or not bool(torch.isfinite(gradient).all()):
                    message = "Muon requires finite dense gradients"
                    raise FloatingPointError(message)
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.mul_(MOMENTUM).add_(gradient, alpha=1 - MOMENTUM)
                nesterov = gradient.mul(1 - MOMENTUM).add(buffer, alpha=MOMENTUM)
                direction = matrix_direction(nesterov)
                shape_scale = 0.2 * math.sqrt(max(parameter.shape))
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction, alpha=-group["lr"] * shape_scale)
                state["step"] += 1
        return loss


def partition(model: torch.nn.Module, kind: str) -> tuple[list, list]:
    """Enumerate unique matrices and assign the tied weight exactly once."""
    if kind not in {"muon_adamw", "muon"}:
        message = "Unknown Muon partition"
        raise ValueError(message)
    pairs = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not pairs or any(p.ndim != MATRIX_DIMENSIONS for _, p in pairs):
        message = "This study permits only the inventoried matrix-only model"
        raise ValueError(message)
    embedding = model.token_embedding_table.weight
    if model.lm_head.weight is not embedding:
        message = "The embedding and readout must remain tied"
        raise ValueError(message)
    muon, adamw = [], []
    for name, parameter in pairs:
        target = adamw if kind == "muon_adamw" and parameter is embedding else muon
        target.append((name, parameter))
    identifiers = [id(p) for _, p in [*muon, *adamw]]
    if len(set(identifiers)) != len(pairs) or id(embedding) not in identifiers:
        message = "Partition must be disjoint, exhaustive and include the tied weight"
        raise AssertionError(message)
    return muon, adamw


class MuonBundle:
    """Save and restore the complete matrix and optional native auxiliary state."""

    def __init__(
        self,
        model: torch.nn.Module,
        kind: str,
        *,
        matrix_lr: float,
        auxiliary_lr: float,
        weight_decay: float,
        device: str,
    ) -> None:
        """Construct both groups without consuming RNG or issuing updates."""
        muon, auxiliary = partition(model, kind)
        self.partition = {
            "kind": kind,
            "muon": [{"name": n, "shape": list(p.shape)} for n, p in muon],
            "adamw": [{"name": n, "shape": list(p.shape)} for n, p in auxiliary],
        }
        self.muon = MatrixMuon(muon, lr=matrix_lr, weight_decay=weight_decay)
        self.adamw = (
            torch.optim.AdamW(
                [
                    {
                        "params": [p for _, p in auxiliary],
                        "param_names": [n for n, _ in auxiliary],
                    }
                ],
                lr=auxiliary_lr,
                betas=(0.9, 0.98),
                eps=1e-8,
                weight_decay=weight_decay,
                fused=device == "cuda",
                capturable=False,
                foreach=False if device != "cuda" else None,
            )
            if auxiliary
            else None
        )

    @property
    def param_groups(self) -> list[dict]:
        """Expose the live groups in matrix-then-auxiliary order."""
        return self.muon.param_groups + (
            self.adamw.param_groups if self.adamw is not None else []
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear every unique parameter's gradient once."""
        self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Update disjoint groups once using the same current model gradient."""
        self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def state_dict(self) -> dict:
        """Retain the explicit partition and all optimizer state for resumption."""
        return {
            "partition": self.partition,
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        """Reject a changed partition before loading either optimizer."""
        if state["partition"] != self.partition or (
            (state["adamw"] is None) != (self.adamw is None)
        ):
            message = "Cannot restore a different optimizer partition"
            raise ValueError(message)
        self.muon.load_state_dict(state["muon"])
        if self.adamw is not None:
            self.adamw.load_state_dict(state["adamw"])
