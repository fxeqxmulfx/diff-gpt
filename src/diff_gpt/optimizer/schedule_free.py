"""
Schedule-Free AdamW (Defazio et al., "The Road Less Scheduled").

Algorithm 1 of arXiv:2405.15682v4, Section 4.4, written from the paper with
explicit iterates. The update is that of lean_optim's ScheduleFreeAdamW,
which the Lean model Transformer.ScheduleFree.adamStep pins down.
"""

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """
    Schedule-Free AdamW over one group of parameters.

    Each parameter keeps a base iterate z and its weighted average x, both
    starting at the initial weights, and the raw second moment v of its
    gradients. Step t takes the gradient g at the query y = (1 - β1)·z + β1·x
    and updates

        rate = lr · sqrt(1 - β2^t) · min(1, t / warmup)
        v ← β2·v + (1 - β2)·g²
        z ← z - rate·g / (sqrt(v) + eps) - rate·weight_decay·y
        x ← x + c·(z - x),  c = rate² / (sum of rate² over steps 1..t)

    The rate folds in Adam's bias correction, so eps adds to the square root
    of the raw v, not of the bias-corrected one as in torch.optim.AdamW, and
    the decay acts at y. No learning-rate schedule is needed: x averages
    the z iterates.

    The parameters hold y in training mode. eval() writes x into them, for
    evaluation and saving; train() writes y back before the next step. Both
    rewrite the parameters from the iterates, whatever they held.
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
        Start z and x at the current weights, in training mode.

        `params` are the tensors themselves, not parameter groups; those
        that do not require grad are left out. The rate ramps up linearly
        over the first `warmup` steps (0: no warmup). ValueError on
        parameter groups, on no trainable tensor, or on a hyperparameter
        out of range.
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
            "betas must lie in [0, 1)": 0 <= beta1 < 1 and 0 <= beta2 < 1,
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
            "total": 0.0,  # sum of rate² over the steps so far
        }
        super().__init__(params, defaults)
        for p in params:
            self.state[p] = {
                "base": p.detach().clone(),  # z
                "evaluation": p.detach().clone(),  # x
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
        """Write y into the parameters, for training steps."""
        self._write(training=True)

    @torch.no_grad()
    def eval(self) -> None:
        """Write x into the parameters, for evaluation and saving."""
        self._write(training=False)

    def _write(self, *, training: bool) -> None:
        group = self.param_groups[0]
        group["training"] = training
        params = group["params"]
        evaluation = [self.state[p]["evaluation"] for p in params]
        if training:
            # y = lerp(z, x, β1), as torch.lerp(z, x, β1, out=y) computes it.
            torch._foreach_copy_(params, [self.state[p]["base"] for p in params])
            torch._foreach_lerp_(params, evaluation, group["betas"][0])
        else:
            torch._foreach_copy_(params, evaluation)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Update v, z and x from the gradients at y, then write the new y.

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
        beta2 = group["betas"][1]
        group["step"] += 1
        t = group["step"]
        ramp = min(1.0, t / group["warmup"]) if group["warmup"] else 1.0
        rate = group["lr"] * math.sqrt(1 - beta2**t) * ramp
        group["total"] += rate**2
        weight = rate**2 / group["total"] if group["total"] else 0.0
        second = [self.state[p]["second"] for p in params]
        base = [self.state[p]["base"] for p in params]
        torch._foreach_mul_(second, beta2)
        torch._foreach_addcmul_(second, grads, grads, value=1 - beta2)
        # torch exposes the foreach kernels only as private functions.
        denominator = torch._foreach_sqrt(second)  # noqa: SLF001
        torch._foreach_add_(denominator, group["eps"])
        torch._foreach_addcdiv_(base, grads, denominator, value=-rate)
        # The decay acts at y, which the parameters hold.
        torch._foreach_add_(base, params, alpha=-rate * group["weight_decay"])
        if weight:
            evaluation = [self.state[p]["evaluation"] for p in params]
            torch._foreach_lerp_(evaluation, base, weight)
        self._write(training=True)
        return loss
