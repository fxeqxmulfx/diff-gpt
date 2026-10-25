"""
Schedule-Free AdamW (Defazio et al., "The Road Less Scheduled").

Algorithm 1 of arXiv:2405.15682v4, Section 4.4, as facebookresearch's
schedulefree has computed it since version 1.3: the bias correction divides
the second moment instead of scaling the rate, the weights of the average x
follow the largest rate so far, and only z and v are stored while the
parameters hold y or x. The step is trainingStep of the Lean model
Transformer.ScheduleFree.AdamW with its default averaging powers, which
trainingRun_pack proves equal to stepping z and x explicitly; evaluation_pack
proves that eval() recovers x. lean_optim's ScheduleFreeAdamW and the Lean
Transformer.ScheduleFree.adamStep keep the paper's version.
"""

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """
    Schedule-Free AdamW over one group of parameters.

    Each parameter keeps a base iterate z, starting at the initial weights,
    and the second moment v of its gradients. Step t takes the gradient g at
    the query y = (1 - β1)·z + β1·x, where x is the weighted average of the
    z iterates, and updates

        rate = lr · min(1, t / warmup),  m = the largest rate over steps 1..t
        v ← β2·v + (1 - β2)·g²
        u = g / (sqrt(v / (1 - β2^t)) + eps) + weight_decay·y
        z ← z - rate·u
        x ← x + c·(z - x),  c = m² / (sum of m² over steps 1..t)

    x itself is not stored: with the z before the step, y moves to

        y ← y + c·(z - y) + rate·(β1·(1 - c) - 1)·u

    The bias correction divides v, as in torch.optim.AdamW, so eps adds to
    the square root of the corrected v. The paper folds sqrt(1 - β2^t) into
    the rate instead, which also weakens the decay and the weights of x over
    the first ~1/(1 - β2) steps; schedulefree 1.3 moved it to v for
    stability. The decay acts at y. The weights of x follow m, not the rate,
    so a falling lr does not shrink them, and at lr 0 z stays while x still
    averages. No learning-rate schedule is needed: x averages the z iterates.

    The parameters hold y in training mode. eval() turns them into
    x = y + (1 - 1/β1)·(z - y), for evaluation and saving, and train() turns
    them back into y = x + (1 - β1)·(z - x) before the next step; a repeated
    call changes nothing. Each switch rounds, so a round trip can move y in
    its last bits, and whatever the parameters hold in eval mode is x.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        warmup: int = 0,
    ) -> None:
        """
        Start z at the current weights, in training mode.

        `params` are the tensors themselves, not parameter groups; those
        that do not require grad are left out. The rate ramps up linearly
        over the first `warmup` steps (0: no warmup). β1 must be above 0,
        since eval() divides by it. ValueError on parameter groups, on no
        trainable tensor, or on a hyperparameter out of range.
        """
        params = list(params)
        if any(isinstance(p, dict) for p in params):
            msg = "ScheduleFreeAdamW takes one parameter group: pass the tensors"
            raise ValueError(msg)
        params = [p for p in params if p.requires_grad]
        if not params:
            msg = "ScheduleFreeAdamW needs a tensor that requires grad"
            raise ValueError(msg)
        beta1, beta2 = betas
        checks = {
            "lr must be finite and >= 0": 0 <= lr < math.inf,
            "weight_decay must be finite and >= 0": 0 <= weight_decay < math.inf,
            "eps must be finite and > 0": 0 < eps < math.inf,
            "betas must lie in (0, 1) and [0, 1)": 0 < beta1 < 1 and 0 <= beta2 < 1,
            "warmup must be an int >= 0": type(warmup) is int and warmup >= 0,
        }
        for msg, ok in checks.items():
            if not ok:
                raise ValueError(msg)
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
            "warmup": warmup,
            "training": True,  # the parameters hold y, not x
            "step": 0,
            "maximum": 0.0,  # m, the largest rate so far
            "total": 0.0,  # sum of m² over the steps so far
        }
        super().__init__(params, defaults)
        for p in params:
            self.state[p] = {
                "base": p.detach().clone(),  # z
                "second": torch.zeros_like(p),  # v
            }

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add the one parameter group; ValueError on a second one."""
        if self.param_groups:
            msg = "ScheduleFreeAdamW takes one parameter group"
            raise ValueError(msg)
        super().add_param_group(param_group)

    @torch.no_grad()
    def train(self) -> None:
        """Turn the parameters from x into y, for training steps."""
        self._switch(training=True)

    @torch.no_grad()
    def eval(self) -> None:
        """Turn the parameters from y into x, for evaluation and saving."""
        self._switch(training=False)

    def _switch(self, *, training: bool) -> None:
        group = self.param_groups[0]
        if group["training"] == training:
            return
        group["training"] = training
        beta1 = group["betas"][0]
        params = group["params"]
        base = [self.state[p]["base"] for p in params]
        # Both follow from y = (1 - β1)·z + β1·x.
        torch._foreach_lerp_(params, base, 1 - beta1 if training else 1 - 1 / beta1)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Update v, z and y from the gradients at y.

        `closure`, if given, recomputes the loss with gradients enabled; its
        loss is returned. RuntimeError in eval mode; ValueError when a
        parameter has no gradient or a sparse one.
        """
        group = self.param_groups[0]
        if not group["training"]:
            msg = "call train() before step(): the parameters hold x, not y"
            raise RuntimeError(msg)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = group["params"]
        grads = [p.grad for p in params]
        if any(g is None or g.is_sparse for g in grads):
            msg = "ScheduleFreeAdamW needs a dense gradient for every parameter"
            raise ValueError(msg)
        beta1, beta2 = group["betas"]
        group["step"] += 1
        t = group["step"]
        ramp = min(1.0, t / group["warmup"]) if group["warmup"] else 1.0
        rate = group["lr"] * ramp
        group["maximum"] = max(rate, group["maximum"])
        group["total"] += group["maximum"] ** 2
        weight = group["maximum"] ** 2 / group["total"] if group["total"] else 0.0
        second = [self.state[p]["second"] for p in params]
        base = [self.state[p]["base"] for p in params]
        torch._foreach_mul_(second, beta2)
        torch._foreach_addcmul_(second, grads, grads, value=1 - beta2)
        # torch exposes the foreach kernels only as private functions.
        denominator = torch._foreach_div(second, 1 - beta2**t)  # noqa: SLF001
        torch._foreach_sqrt_(denominator)
        torch._foreach_add_(denominator, group["eps"])
        update = torch._foreach_div(grads, denominator)  # noqa: SLF001
        # The decay acts at y, which the parameters hold.
        torch._foreach_add_(update, params, alpha=group["weight_decay"])
        # y follows x and z, from the z before the step.
        torch._foreach_lerp_(params, base, weight)
        torch._foreach_add_(params, update, alpha=rate * (beta1 * (1 - weight) - 1))
        torch._foreach_sub_(base, update, alpha=rate)
        return loss
