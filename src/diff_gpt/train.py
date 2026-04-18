from datetime import datetime
from typing import Protocol

import torch
from tqdm.autonotebook import tqdm

from diff_gpt.model.gpt import BaseGPT
from diff_gpt.optimizer.adamw_schedulefree import AdamWScheduleFree


class DataLoader(Protocol):
    """
    Minimal protocol every loader must satisfy for `train` to drive it.
    """

    block_size: int
    batch_size: int

    def get_batch(
        self, split: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def has_val(self) -> bool: ...


@torch.inference_mode()
def _estimate_loss(
    model: BaseGPT, loader: DataLoader, eval_iters: int, split: str
) -> torch.Tensor:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = loader.get_batch(split)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean()


def train(
    mut_model: BaseGPT,
    loader: DataLoader,
    learning_rate: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    max_iters: int = 5_000,
    eval_interval: int = 5_000,
    eval_iters: int = 200,
    use_tqdm: bool = True,
    use_early_stopping: bool = True,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    save_best: bool = True,
) -> tuple[float, int]:
    """
    Train `mut_model` against batches produced by `loader`.

    Expects `loader.get_batch('train')` (and optionally `loader.get_batch('val')`
    when `loader.has_val()` is True) to return `(x, y)` tensors on the model's
    device. Validation is only run when `loader.has_val()` — SQLite-style
    stream loaders that don't carry a val split will just report train loss.

    `grad_accum_steps` enables gradient accumulation for memory-constrained
    GPUs; `grad_clip_norm` clips the global gradient norm before each step.

    `save_best` (default True, only meaningful when `has_val`): keep an
    in-memory snapshot of model weights whenever a new best val_loss is
    seen and restore it before returning. Lets training run past the val
    optimum without losing the best model (useful when val is noisy).
    Returns the best val_loss instead of the final one when active.
    """
    assert grad_accum_steps >= 1, "grad_accum_steps must be >= 1"
    optimizer = AdamWScheduleFree(
        mut_model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
    has_val = loader.has_val()
    start = datetime.now()
    pbar = tqdm(range(max_iters)) if use_tqdm else range(max_iters)
    last_reported_loss: torch.Tensor = torch.tensor(float("inf"))
    last_val_loss = torch.inf
    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_step = -1
    mut_model.train()
    optimizer.train()
    for iter in pbar:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            mut_model.eval()
            optimizer.eval()
            train_loss = _estimate_loss(mut_model, loader, eval_iters, "train")
            if has_val:
                val_loss = _estimate_loss(mut_model, loader, eval_iters, "val")
                last_reported_loss = val_loss
                current_val = float(val_loss)
                improved = ""
                if save_best and current_val < best_val_loss:
                    best_val_loss = current_val
                    # Clone to CPU-agnostic detached copies so resumed
                    # training doesn't mutate the snapshot.
                    best_state_dict = {
                        k: v.detach().clone() for k, v in mut_model.state_dict().items()
                    }
                    best_step = iter
                    improved = "  (best)"
                status = (
                    f"step {iter}: train loss {train_loss:.4f}, "
                    f"val loss {val_loss:.4f}{improved}"
                )
            else:
                last_reported_loss = train_loss
                status = f"step {iter}: train loss {train_loss:.4f}"
            mut_model.train()
            optimizer.train()
            if isinstance(pbar, tqdm):
                pbar.set_description(status)
            else:
                print(status)
            if has_val and use_early_stopping and last_val_loss < float(
                last_reported_loss
            ):
                break
            last_val_loss = float(last_reported_loss)
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
    train_time_s = (datetime.now() - start).seconds
    mut_model.eval()
    if save_best and best_state_dict is not None:
        mut_model.load_state_dict(best_state_dict)
        msg = f"restored best checkpoint from step {best_step} (val_loss={best_val_loss:.4f})"
        if isinstance(pbar, tqdm):
            pbar.write(msg)
        else:
            print(msg)
        return best_val_loss, train_time_s
    return float(last_reported_loss), train_time_s
