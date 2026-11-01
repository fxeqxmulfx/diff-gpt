"""
Apply the verified depth-query coordinate change to every transformer mixer.

Source: residual_conditioning.py at ec39481 and Lean QueryConditioning QC6--QC9.
One compiled kernel handles all query/base/moment/scale buffers. This is the
same separate-before-forward change, now for several pseudo-queries. The
ordinary GPT parameters and all shared optimizer clocks stay in their frame.
"""

import math
from typing import TYPE_CHECKING

import torch

from diff_gpt.application.residual_conditioning import recondition_state
from diff_gpt.domain.model.conditioned_attention_residual import (
    ConditionedQKAttentionResidual,
)
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from torch import Tensor, nn


@torch.no_grad()
def recondition_queries(
    queries: tuple[Tensor, ...],
    bases: tuple[Tensor, ...],
    seconds: tuple[Tensor, ...],
    scales: tuple[Tensor, ...],
    beta: float,
    maximum_rms: float,
) -> Tensor:
    """Fuse independent scalar coordinate changes across all transformer queries."""
    return torch.stack(
        [
            recondition_state(q, z, v, s, beta, maximum_rms)
            for q, z, v, s in zip(queries, bases, seconds, scales, strict=True)
        ]
    )


class TransformerQueryConditioner:
    """Bind every conditioned mixer to its complete Schedule-Free state."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: ScheduleFreeAdamW,
        *,
        maximum_rms: float = 2.0,
        compiled: bool = False,
    ) -> None:
        """Discover only explicit conditioned mixers; reject unrelated optimizers."""
        if not isinstance(optimizer, ScheduleFreeAdamW):
            msg = "TransformerQueryConditioner requires ScheduleFreeAdamW"
            raise TypeError(msg)
        if not 1 <= maximum_rms < math.inf:
            msg = "maximum_rms must be finite and at least one"
            raise ValueError(msg)
        self.mixers = tuple(
            m for m in model.modules() if isinstance(m, ConditionedQKAttentionResidual)
        )
        if not self.mixers:
            msg = "The model has no conditioned depth mixers"
            raise ValueError(msg)
        self.optimizer = optimizer
        self.maximum_rms = maximum_rms
        owned = {id(p) for p in optimizer.param_groups[0]["params"]}
        if any(id(m.query) not in owned for m in self.mixers):
            msg = "The optimizer must own every conditioned query"
            raise ValueError(msg)
        self.kernel = (
            torch.compile(
                recondition_queries, mode="max-autotune", fullgraph=True, dynamic=False
            )
            if compiled
            else recondition_queries
        )

    def __call__(self) -> Tensor:
        """Call after clearing gradients and restoring training mode, before forward."""
        group = self.optimizer.param_groups[0]
        if not group["training"] or any(m.query.grad is not None for m in self.mixers):
            msg = "Condition in training mode after zero_grad(set_to_none=True)"
            raise RuntimeError(msg)
        # Resolve state after every load_state_dict: saved-state restoration can
        # replace the tensors without replacing the model's Parameter objects.
        states = tuple(self.optimizer.state[m.query] for m in self.mixers)
        return self.kernel(
            tuple(m.query for m in self.mixers),
            tuple(s["base"] for s in states),
            tuple(s["second"] for s in states),
            tuple(m.query_scale for m in self.mixers),
            group["betas"][0],
            self.maximum_rms,
        )
