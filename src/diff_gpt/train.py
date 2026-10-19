# Copyright 2025 Mikhail Borisov
"""
Training loop that drives a `BaseGPT` with batches from a `DataLoader`.

Uses schedule-free AdamW with periodic train/val evaluation, optional early
stopping and best-checkpoint restore. The same loop runs inside DDP workers
(`train_ddp`), where eval losses are averaged across ranks.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import torch
import torch.distributed as dist
from tqdm.autonotebook import tqdm

from diff_gpt.optimizer.adamw_schedulefree import AdamWScheduleFree

if TYPE_CHECKING:
    from diff_gpt.model.gpt import BaseGPT


class DataLoader(Protocol):
    """Minimal protocol every loader must satisfy for `train` to drive it."""

    block_size: int
    batch_size: int

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        """Return an (inputs, next-token targets) batch from `split`."""
        ...

    def has_val(self) -> bool:
        """Return True if the loader can serve the 'val' split."""
        ...


def _dist_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_rank() -> bool:
    return not _dist_active() or dist.get_rank() == 0


def _all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a scalar to the global mean across ranks; no-op outside DDP.

    All ranks must see the same eval-loss value so early-stop and save-best
    fire on the same iteration — otherwise ranks desync and DDP deadlocks.
    """
    if not _dist_active():
        return t
    world = dist.get_world_size()
    # NCCL needs CUDA tensors; gloo operates on CPU tensors.
    work = t.to("cuda") if dist.get_backend() == "nccl" else t.clone()
    dist.all_reduce(work, op=dist.ReduceOp.SUM)
    return (work / world).to(t.device)


def _agree_on_val(*, has_val: bool) -> bool:
    """
    Return `has_val` after checking that every rank has the same answer.

    A rank with a val split runs one more all-reduce per evaluation than a rank
    without, so ranks that disagree abort (gloo) or hang (NCCL) at that
    collective; raise on every rank instead.
    """
    share = float(_all_reduce_mean(torch.tensor(float(has_val))))
    if 0.0 < share < 1.0:
        msg = (
            "loader.has_val() differs across ranks; give every rank a val split or none"
        )
        raise ValueError(msg)
    return has_val


# Parameters with at least this many dimensions (weight matrices, embeddings)
# are weight-decayed; vectors such as LayerNorm gains are not.
_MIN_DECAY_NDIM = 2


def _param_groups(model: BaseGPT, weight_decay: float) -> list[dict]:
    """
    Split the trainable parameters into weight-decay and no-decay groups.

    Decay weight matrices and embeddings only, never LayerNorm gains (the
    nanoGPT convention).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {
            "params": [p for p in params if p.dim() >= _MIN_DECAY_NDIM],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in params if p.dim() < _MIN_DECAY_NDIM],
            "weight_decay": 0.0,
        },
    ]
    return [g for g in groups if g["params"]]


# no_grad rather than inference_mode: inference tensors created here would
# later break CUDA-graph recording of the training step (in-place update of
# an inference tensor outside inference mode).
@torch.no_grad()
def _estimate_loss(
    model: BaseGPT,
    loader: DataLoader,
    eval_iters: int,
    split: str,
) -> torch.Tensor:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = loader.get_batch(split)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    return _all_reduce_mean(losses.mean())


class _Evaluation:
    """Losses that `train` has seen so far and its best-val snapshot."""

    __slots__ = ("best_loss", "best_state", "best_step", "loss")

    def __init__(self) -> None:
        """Start with no evaluation: every loss is infinite."""
        self.loss = float("inf")  # reported loss of the latest evaluation
        self.best_loss = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_step = -1

    def run(
        self,
        model: BaseGPT,
        loader: DataLoader,
        eval_iters: int,
        step: int,
        *,
        has_val: bool,
        save_best: bool,
    ) -> str:
        """
        Score the current weights after `step` updates; return the status line.

        The model and the optimizer must be in eval mode. The reported loss is
        the val loss when `has_val`, else the train loss. With `save_best`, a
        new best val loss snapshots the weights.
        """
        train_loss = float(_estimate_loss(model, loader, eval_iters, "train"))
        status = f"step {step}: train loss {train_loss:.4f}"
        if not has_val:
            self.loss = train_loss
            return status
        self.loss = float(_estimate_loss(model, loader, eval_iters, "val"))
        status += f", val loss {self.loss:.4f}"
        if save_best and self.loss < self.best_loss:
            self.best_loss = self.loss
            # Detached copies, so that resumed training doesn't mutate the
            # snapshot.
            self.best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            self.best_step = step
            status += "  (best)"
        return status


def train(
    mut_model: BaseGPT,
    loader: DataLoader,
    learning_rate: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    max_iters: int = 5_000,
    eval_interval: int = 5_000,
    eval_iters: int = 200,
    *,
    use_tqdm: bool = True,
    use_early_stopping: bool = True,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    save_best: bool = True,
    optimizer: AdamWScheduleFree | None = None,
) -> tuple[float, int]:
    """
    Train `mut_model` against batches produced by `loader`.

    Expects `loader.get_batch('train')` (and optionally `loader.get_batch('val')`
    when `loader.has_val()` is True) to return `(x, y)` tensors on the model's
    device. Validation is only run when `loader.has_val()` — SQLite-style
    stream loaders that don't carry a val split will just report train loss.

    `grad_accum_steps` enables gradient accumulation for memory-constrained
    GPUs; `grad_clip_norm` clips the global gradient norm before each step.

    Evaluation runs every `eval_interval` steps, before that step's update,
    and once more after the last update, so the final weights are scored
    too (`max_iters=0` scores the initial weights). With `use_early_stopping`
    and a val split, training stops at the first in-loop evaluation whose
    val loss is above the previous one.

    `save_best` (default True, only meaningful when `has_val`): keep an
    in-memory snapshot of model weights whenever a new best val_loss is
    seen and restore it before returning. Lets training run past the val
    optimum without losing the best model (useful when val is noisy).
    Returns the best val_loss instead of the final one when active. Either
    way the returned loss is that of the returned weights.

    `optimizer`: a schedule-free AdamW over the parameters of `mut_model`;
    by default one is built from `learning_rate`, `betas` and `weight_decay`,
    which are ignored when `optimizer` is given. It is left in eval mode.
    After a best-snapshot restore only the weights come from the snapshot:
    the optimizer state (z, exp_avg_sq, step count) is that of the last
    step, so training resumed with it starts from y = β1·x_best +
    (1 - β1)·z_last.

    In DDP every rank must agree on `loader.has_val()`; ValueError otherwise.
    """
    if grad_accum_steps < 1:
        msg = "grad_accum_steps must be >= 1"
        raise ValueError(msg)
    if optimizer is None:
        optimizer = AdamWScheduleFree(
            _param_groups(mut_model, weight_decay),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
    has_val = _agree_on_val(has_val=loader.has_val())
    start = datetime.now(tz=UTC)
    is_main = _is_main_rank()
    pbar = tqdm(range(max_iters)) if (use_tqdm and is_main) else range(max_iters)

    def report(msg: str) -> None:
        if isinstance(pbar, tqdm):
            pbar.set_description(msg)
        elif is_main:
            print(msg)

    evaluation = _Evaluation()
    previous_loss = float("inf")
    stopped_early = False
    mut_model.train()
    optimizer.train()
    for step in pbar:
        if step % eval_interval == 0:
            mut_model.eval()
            optimizer.eval()
            report(
                evaluation.run(
                    mut_model,
                    loader,
                    eval_iters,
                    step,
                    has_val=has_val,
                    save_best=save_best,
                ),
            )
            # Stop while still in eval mode: the weights handed back are then
            # exactly the scored ones (the train/eval switch rounds).
            if has_val and use_early_stopping and previous_loss < evaluation.loss:
                stopped_early = True
                break
            previous_loss = evaluation.loss
            mut_model.train()
            optimizer.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            xb, yb = loader.get_batch("train")
            _, loss = mut_model(xb, yb)
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(mut_model.parameters(), grad_clip_norm)
        optimizer.step()
    # Schedule-free: parameters hold the fast "y" sequence while training; move
    # them to the averaged "x" point before handing the model back. Without
    # this the returned weights are the y point whenever no best-snapshot is
    # restored (no val split, or save_best=False).
    optimizer.eval()
    mut_model.eval()
    if not stopped_early:
        # Score the last update too; the in-loop evaluations precede it.
        report(
            evaluation.run(
                mut_model,
                loader,
                eval_iters,
                max_iters,
                has_val=has_val,
                save_best=save_best,
            ),
        )
    train_time_s = int((datetime.now(tz=UTC) - start).total_seconds())
    if save_best and evaluation.best_state is not None:
        mut_model.load_state_dict(evaluation.best_state)
        msg = (
            f"restored best checkpoint from step {evaluation.best_step} "
            f"(val_loss={evaluation.best_loss:.4f})"
        )
        if isinstance(pbar, tqdm):
            pbar.write(msg)
        elif is_main:
            print(msg)
        return evaluation.best_loss, train_time_s
    return evaluation.loss, train_time_s
