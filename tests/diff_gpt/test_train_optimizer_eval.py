"""Tests for the weights and the loss that train() hands back."""

import pytest
import torch

from diff_gpt.data_loader import TokenSequenceDataLoader
from diff_gpt.model.gpt import GPT
from diff_gpt.optimizer.adamw_schedulefree import AdamWScheduleFree
from diff_gpt.train import train


def test_train_returns_weights_at_schedule_free_x_point() -> None:
    """
    Check that train() without a val split hands back the weights at the x point.

    Without a val split nothing restores a snapshot, so the weights handed
    back must be the averaged x point (optimizer in eval mode), not the y point.
    """
    torch.manual_seed(0)
    tokens = torch.randint(0, 64, (2000,)).tolist()
    model = GPT(
        vocab_size=64,
        n_embd=16,
        block_size=8,
        n_head=2,
        n_layer=1,
        use_checkpoint=False,
    )
    loader = TokenSequenceDataLoader(
        tokens,
        block_size=8,
        batch_size=4,
        device="cpu",
        train_part=1.0,
    )
    opt = AdamWScheduleFree(model.parameters())
    train(
        model,
        loader,
        max_iters=6,
        eval_interval=3,
        eval_iters=1,
        use_tqdm=False,
        use_early_stopping=False,
        save_best=True,
        optimizer=opt,
    )
    assert opt.param_groups[0]["train_mode"] is False
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.eval()  # already at x: a second eval() must be a no-op
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p)


# Train and val parts of `_loader`: 91 train tokens, and a val part of exactly
# block_size + 1 tokens, so that every val batch is the same single window.
BLOCK_SIZE = 8
N_TRAIN_TOKENS = 91
# Step budget of the early-stopping test; the val loss rises long before it.
EARLY_STOP_MAX_ITERS = 50


def _gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(
        vocab_size=64,
        n_embd=16,
        block_size=BLOCK_SIZE,
        n_head=2,
        n_layer=1,
        use_checkpoint=False,
    )


def _loader(val_tokens: list[int] | None = None) -> TokenSequenceDataLoader:
    """
    Return a loader over a 0, 1, 2, 3 cycle with a one-window val part.

    `val_tokens` replaces the val part (block_size + 1 tokens).
    """
    tokens = [i % 4 for i in range(N_TRAIN_TOKENS + BLOCK_SIZE + 1)]
    if val_tokens is not None:
        tokens[N_TRAIN_TOKENS:] = val_tokens
    return TokenSequenceDataLoader(
        tokens,
        block_size=BLOCK_SIZE,
        batch_size=4,
        device="cpu",
        train_part=N_TRAIN_TOKENS / len(tokens),
        rng=torch.Generator().manual_seed(0),
    )


class _ScoredWeightsLoader:
    """
    Pass `loader`'s batches through, copying the model's weights at each val batch.

    train() scores the weights it holds when it asks for a val batch, so
    `scored` ends up with the weights of the last evaluation.
    """

    def __init__(self, loader: TokenSequenceDataLoader, model: GPT) -> None:
        self.loader = loader
        self.model = model
        self.block_size = loader.block_size
        self.batch_size = loader.batch_size
        self.scored: dict[str, torch.Tensor] = {}

    def has_val(self) -> bool:
        return self.loader.has_val()

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        if split == "val":
            self.scored = {k: v.clone() for k, v in self.model.state_dict().items()}
        return self.loader.get_batch(split)


@torch.no_grad()
def _val_loss(model: GPT, loader: TokenSequenceDataLoader) -> float:
    x, y = loader.get_batch("val")
    _, loss = model(x, y)
    assert loss is not None
    return float(loss)


@pytest.mark.parametrize("save_best", [False, True])
def test_train_returns_the_val_loss_of_the_weights_it_hands_back(
    *,
    save_best: bool,
) -> None:
    """
    Check that the returned loss is the val loss of the returned weights.

    The last evaluation ran before the last update, so the returned loss was
    a step stale: 1.714, while the returned weights score 1.656.
    """
    model = _gpt()
    loader = _loader()
    loss, _ = train(
        model,
        loader,
        max_iters=20,
        eval_interval=10,
        eval_iters=1,
        use_tqdm=False,
        use_early_stopping=False,
        save_best=save_best,
    )
    assert loss == pytest.approx(_val_loss(model, loader), rel=1e-6)


def test_train_one_step_keeps_the_update_when_it_lowers_val_loss() -> None:
    """
    Check that max_iters=1 scores the update and keeps it.

    With save_best, the one evaluation ran before the update, so the snapshot
    restored at the end was always the untrained model.
    """
    model = _gpt()
    loader = _loader()
    model.eval()
    initial_loss = _val_loss(model, loader)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    loss, _ = train(
        model,
        loader,
        max_iters=1,
        eval_interval=1,
        eval_iters=1,
        use_tqdm=False,
        use_early_stopping=False,
        save_best=True,
    )
    assert loss < initial_loss
    assert loss == pytest.approx(_val_loss(model, loader), rel=1e-6)
    assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())


def test_train_zero_steps_scores_the_initial_weights() -> None:
    """Check that max_iters=0 returns the val loss of the untouched model, not inf."""
    model = _gpt()
    loader = _loader()
    model.eval()
    initial_loss = _val_loss(model, loader)
    loss, _ = train(
        model,
        loader,
        max_iters=0,
        eval_iters=1,
        use_tqdm=False,
    )
    assert loss == pytest.approx(initial_loss, rel=1e-6)


def test_early_stopping_hands_back_exactly_the_scored_weights() -> None:
    """
    Check that early stopping returns the weights it scored, bit for bit.

    Val runs the cycle backwards: once the model learns the train cycle's
    transitions, the val loss rises and training stops. Training used to
    switch back to train mode before it stopped, and the switch to eval mode
    after the loop moved a few dozen weights off the scored ones by rounding.
    """
    model = _gpt()
    reversed_val = [3 - i % 4 for i in range(BLOCK_SIZE + 1)]
    loader = _ScoredWeightsLoader(_loader(val_tokens=reversed_val), model)
    opt = AdamWScheduleFree(model.parameters(), lr=1e-2, betas=(0.9, 0.95))
    loss, _ = train(
        model,
        loader,
        max_iters=EARLY_STOP_MAX_ITERS,
        eval_interval=5,
        eval_iters=1,
        use_tqdm=False,
        use_early_stopping=True,
        save_best=False,
        optimizer=opt,
    )
    assert opt.param_groups[0]["k"] < EARLY_STOP_MAX_ITERS
    for k, v in model.state_dict().items():
        assert torch.equal(loader.scored[k], v), k
    assert loss == _val_loss(model, loader.loader)
