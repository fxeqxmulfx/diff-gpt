"""
Condition depth-query coordinates without discarding Schedule-Free history.

Source: ScheduleFreeAdamW at 4e58930 and the companion Lean
Transformer.AttentionResiduals.QueryCoordinates. For a positive factor c,
y and z divide by c, the gradient second moment multiplies by c**2, and
the query coordinate scale multiplies by c. The implied average x divides
by c too; beta, averaging weights, step and sampler are unchanged.

This is an explicit change to optimization geometry. Subsequent ordinary
Schedule-Free AdamW steps operate in the new coordinates; their physical
learning rate is therefore different from the unconditioned baseline.
It is not a claim of identical optimizer trajectories or convergence.
"""

import math
from typing import TYPE_CHECKING

import torch

from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from torch import Tensor

    from diff_gpt.domain.model.conditioned_attention_residual import (
        ConditionedQKAttentionResidual,
    )


def stable_rms(value: Tensor) -> Tensor:
    """Measure RMS without squaring a potentially huge original coordinate."""
    scale = value.abs().amax().clamp_min(torch.finfo(value.dtype).tiny)
    return scale * (value / scale).square().mean().sqrt()


@torch.no_grad()
def recondition_state(
    query: Tensor,
    base: Tensor,
    second: Tensor,
    coordinate_scale: Tensor,
    beta: float,
    maximum_rms: float,
) -> Tensor:
    """Apply one common linear coordinate change to the complete query state."""
    average = torch.lerp(query, base, 1 - 1 / beta)
    largest = torch.stack([stable_rms(v) for v in (query, base, average)]).amax()
    factor = torch.where(largest > maximum_rms, largest, torch.ones_like(largest))
    query.div_(factor)
    base.div_(factor)
    second.mul_(factor).mul_(factor)
    coordinate_scale.mul_(factor)
    return factor


class DepthQueryConditioner:
    """
    Keep query, base and implied average in bounded coordinates before a step.

    Call after zero_grad(set_to_none=True), before the next forward/backward.
    Forward has no detached normalization or substituted gradient. A chart
    change is a separate operation on model and optimizer state. In real
    arithmetic, each state has RMS <= maximum_rms after conditioning. Floating-
    point range and roundoff are checked separately in the declared tests.
    Zero and small queries are left alone; inflating them would hurt learning.
    """

    def __init__(
        self,
        model: ConditionedQKAttentionResidual,
        optimizer: ScheduleFreeAdamW,
        *,
        maximum_rms: float = 2.0,
        compiled: bool = False,
    ) -> None:
        """Bind the actual query state; optionally compile the small state kernel."""
        if not isinstance(optimizer, ScheduleFreeAdamW):
            msg = "DepthQueryConditioner requires ScheduleFreeAdamW"
            raise TypeError(msg)
        if not 1 <= maximum_rms < math.inf:
            msg = "maximum_rms must be finite and at least one"
            raise ValueError(msg)
        if not any(p is model.query for p in optimizer.param_groups[0]["params"]):
            msg = "The optimizer must own this model's query"
            raise ValueError(msg)
        self.model = model
        self.optimizer = optimizer
        self.maximum_rms = maximum_rms
        self.kernel = (
            torch.compile(
                recondition_state, mode="max-autotune", fullgraph=True, dynamic=False
            )
            if compiled
            else recondition_state
        )

    def __call__(self) -> Tensor:
        """Recondition training coordinates without resetting any optimizer clock."""
        group = self.optimizer.param_groups[0]
        if not group["training"]:
            msg = "Call optimizer.train() before conditioning the query"
            raise RuntimeError(msg)
        if self.model.query.grad is not None:
            msg = (
                "Condition after zero_grad(set_to_none=True), "
                "before computing gradients"
            )
            raise RuntimeError(msg)
        state = self.optimizer.state[self.model.query]
        return self.kernel(
            self.model.query,
            state["base"],
            state["second"],
            self.model.query_scale,
            group["betas"][0],
            self.maximum_rms,
        )
