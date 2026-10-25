"""
Training loop that drives a `BaseGPT` with batches from a `DataLoader`.

Uses schedule-free AdamW with periodic train/val evaluation, which an
`on_eval` callback can record, and opt-in early stopping and best-checkpoint
restore. The same loop runs inside DDP workers (`train_ddp`), where eval
losses are averaged across ranks.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
import torch.distributed as dist
from tqdm.autonotebook import tqdm

from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from collections.abc import Callable

    from diff_gpt.domain.model.gpt import BaseGPT


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


class Evaluation(NamedTuple):
    """
    The losses of the weights after `step` updates, as `train` measured them.

    Each loss is the mean over `eval_iters` random batches of its split;
    `val_loss` is None when the loader has no val split.
    """

    step: int
    train_loss: float
    val_loss: float | None


class _Evaluations:
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
    ) -> tuple[Evaluation, str]:
        """
        Score the current weights after `step` updates.

        Return the evaluation and its status line. The model and the optimizer
        must be in eval mode. The reported loss is the val loss when
        `has_val`, else the train loss. With `save_best`, a new best val loss
        snapshots the weights.
        """
        train_loss = float(_estimate_loss(model, loader, eval_iters, "train"))
        status = f"step {step}: train loss {train_loss:.4f}"
        if not has_val:
            self.loss = train_loss
            return Evaluation(step, train_loss, None), status
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
        return Evaluation(step, train_loss, self.loss), status


def _check_start(
    start_step: int,
    max_iters: int,
    optimizer: ScheduleFreeAdamW | None,
    *,
    resumable: bool,
) -> None:
    """Raise ValueError unless `train` can go on from `start_step` updates."""
    if not resumable:
        msg = (
            "start_step > 0 cannot restore the state of use_early_stopping or "
            "save_best: train on without them"
        )
        raise ValueError(msg)
    if not 0 < start_step < max_iters:
        msg = f"start_step must lie in (0, max_iters = {max_iters}), got {start_step}"
        raise ValueError(msg)
    if optimizer is None:
        msg = "start_step > 0 needs the optimizer of the saved state"
        raise ValueError(msg)
    made = optimizer.param_groups[0].get("step")
    if made != start_step:
        msg = f"the optimizer has made {made} steps, not start_step = {start_step}"
        raise ValueError(msg)


def train(
    mut_model: BaseGPT,
    loader: DataLoader,
    learning_rate: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.03,
    max_iters: int = 5_000,
    eval_interval: int = 5_000,
    eval_iters: int = 200,
    *,
    warmup: int = 1_000,
    use_tqdm: bool = True,
    use_early_stopping: bool = False,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    save_best: bool = False,
    optimizer: ScheduleFreeAdamW | None = None,
    on_eval: Callable[[Evaluation], None] | None = None,
    autocast: torch.dtype | None = None,
    start_step: int = 0,
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
    too (`max_iters=0` scores the initial weights). By default training runs
    all `max_iters` steps and hands back the final weights and the loss of
    the last evaluation.

    `on_eval`: called with every evaluation, on rank 0 only in DDP. While
    it runs, the model is in eval mode and holds the weights just scored,
    so it can record the curve, save a checkpoint or score forecasts to
    choose one by. It must leave the weights and the mode as they are, and
    draw no batches from `loader`: that would change the training batches.

    `use_early_stopping` and `save_best` select by the val loss, which can
    rise for a long stretch before a model generalizes, so either one hides
    late generalization. Both need a val split. With `use_early_stopping`,
    training stops at the first in-loop evaluation whose val loss is above
    the previous one. With `save_best`, an in-memory snapshot of the weights
    is taken at each new best val loss and restored before returning, and
    the best val loss is returned. Either way the returned loss is that of
    the returned weights.

    `optimizer`: a ScheduleFreeAdamW over the parameters of `mut_model`;
    by default one is built from `learning_rate`, `betas`, `weight_decay`
    and `warmup` (steps of linear lr warmup), which are ignored when
    `optimizer` is given. The weight decay applies to every parameter: the
    GPT has only matrices and embeddings. The optimizer is left in eval
    mode, with its average x in the parameters. A best-snapshot restore
    overwrites them while the optimizer keeps the z and v of the last step,
    so its train() would then start y from the snapshot and that z: train
    on from the returned weights with a new optimizer.

    `autocast`: torch.bfloat16 runs the training forward passes under
    `torch.autocast` (mixed precision). The parameters, their gradients and
    the optimizer state stay float32: the master weights the updates go to.
    The evaluations run without it, so their losses compare across
    precisions. float16 would need loss scaling: ValueError, as for any
    other dtype. The parameters themselves must be float32 or float64: those
    of a model cast to float16 or bfloat16 raise ValueError, since every
    update would round to that precision.

    `start_step`: the updates a saved run has already made, to go on from
    its state. The loop then makes the updates after them, up to
    `max_iters`, and skips the evaluation after `start_step` updates, which
    the saved run made before saving. The caller restores that state: the
    weights at x, `optimizer` (eval mode, `start_step` steps), the loader
    and the RNGs as they were after that evaluation; then the updates and
    evaluations that follow are those of an unbroken run. Early stopping and
    the best snapshot keep in-memory state that a saved run does not carry:
    ValueError with either, without `optimizer`, with a step count other
    than `start_step`, or unless 0 < `start_step` < `max_iters`. 0 (the
    default) trains from the start.

    In DDP every rank must agree on `loader.has_val()`; ValueError otherwise.
    """
    if grad_accum_steps < 1:
        msg = "grad_accum_steps must be >= 1"
        raise ValueError(msg)
    if start_step:
        resumable = not (use_early_stopping or save_best)
        _check_start(start_step, max_iters, optimizer, resumable=resumable)
    if autocast is not None and autocast != torch.bfloat16:
        msg = f"autocast must be None or torch.bfloat16, got {autocast}"
        raise ValueError(msg)
    narrow = {p.dtype for p in mut_model.parameters()} - {torch.float32, torch.float64}
    if narrow:
        msg = (
            "parameters must be float32 or float64, the master weights the updates "
            f"go to; got {sorted(map(str, narrow))}: for mixed precision train a "
            "float32 model with autocast=torch.bfloat16"
        )
        raise ValueError(msg)
    if optimizer is None:
        optimizer = ScheduleFreeAdamW(
            mut_model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            warmup=warmup,
        )
    has_val = _agree_on_val(has_val=loader.has_val())
    start = datetime.now(tz=UTC)
    is_main = _is_main_rank()
    steps = range(start_step, max_iters)
    pbar = tqdm(steps) if (use_tqdm and is_main) else steps

    def report(msg: str) -> None:
        if isinstance(pbar, tqdm):
            pbar.set_description(msg)
        elif is_main:
            print(msg)

    evaluations = _Evaluations()

    def evaluate(step: int) -> None:
        # The model and the optimizer are in eval mode, holding the x point.
        scores, status = evaluations.run(
            mut_model,
            loader,
            eval_iters,
            step,
            has_val=has_val,
            save_best=save_best,
        )
        report(status)
        if on_eval is not None and is_main:
            on_eval(scores)

    previous_loss = float("inf")
    stopped_early = False
    mut_model.train()
    optimizer.train()
    for step in pbar:
        # A resumed run made the evaluation of its first step before saving.
        if step % eval_interval == 0 and not (start_step and step == start_step):
            mut_model.eval()
            optimizer.eval()
            evaluate(step)
            # Stop while still in eval mode, holding the scored weights.
            if has_val and use_early_stopping and previous_loss < evaluations.loss:
                stopped_early = True
                break
            previous_loss = evaluations.loss
            mut_model.train()
            optimizer.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            xb, yb = loader.get_batch("train")
            with torch.autocast(
                mut_model.device_type,
                dtype=autocast,
                enabled=autocast is not None,
            ):
                _, loss = mut_model(xb, yb)
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(mut_model.parameters(), grad_clip_norm)
        optimizer.step()
    # Schedule-free: the parameters hold the query y while training; move
    # them to the average x before handing the model back. Without this the
    # returned weights are y whenever no best-snapshot is restored (no val
    # split, or save_best=False).
    optimizer.eval()
    mut_model.eval()
    if not stopped_early:
        # Score the last update too; the in-loop evaluations precede it.
        evaluate(max_iters)
    train_time_s = int((datetime.now(tz=UTC) - start).total_seconds())
    if save_best and evaluations.best_state is not None:
        mut_model.load_state_dict(evaluations.best_state)
        msg = (
            f"restored best checkpoint from step {evaluations.best_step} "
            f"(val_loss={evaluations.best_loss:.4f})"
        )
        if isinstance(pbar, tqdm):
            pbar.write(msg)
        elif is_main:
            print(msg)
        return evaluations.best_loss, train_time_s
    return evaluations.loss, train_time_s
