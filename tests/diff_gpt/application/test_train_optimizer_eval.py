"""Tests for the weights and the loss that train() hands back."""

import io
from typing import TYPE_CHECKING, Any

import pytest
import torch

from diff_gpt.application.data_loader import TokenSequenceDataLoader
from diff_gpt.application.train import train
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from diff_gpt.application.train import Evaluation


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
    opt = ScheduleFreeAdamW(model.parameters())
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
    assert opt.param_groups[0]["training"] is False
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
    opt = ScheduleFreeAdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95))
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
    assert opt.param_groups[0]["step"] < EARLY_STOP_MAX_ITERS
    for k, v in model.state_dict().items():
        assert torch.equal(loader.scored[k], v), k
    assert loss == _val_loss(model, loader.loader)


def test_train_by_default_runs_every_step_and_keeps_the_final_weights() -> None:
    """
    Check that train() by default neither stops early nor restores the best weights.

    On the reversed val cycle the val loss rises once the model learns the
    train cycle's transitions. train() used to stop there and hand back the
    weights with the lowest val loss, hiding whatever training did later.
    """
    model = _gpt()
    reversed_val = [3 - i % 4 for i in range(BLOCK_SIZE + 1)]
    loader = _ScoredWeightsLoader(_loader(val_tokens=reversed_val), model)
    opt = ScheduleFreeAdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95))
    seen: list[Evaluation] = []
    loss, _ = train(
        model,
        loader,
        max_iters=EARLY_STOP_MAX_ITERS,
        eval_interval=5,
        eval_iters=1,
        use_tqdm=False,
        optimizer=opt,
        on_eval=seen.append,
    )
    assert opt.param_groups[0]["step"] == EARLY_STOP_MAX_ITERS
    for k, v in model.state_dict().items():
        assert torch.equal(loader.scored[k], v), k
    assert loss == seen[-1].val_loss == _val_loss(model, loader.loader)
    val_losses = [e.val_loss for e in seen if e.val_loss is not None]
    assert len(val_losses) == len(seen)
    assert min(val_losses) < loss


def test_on_eval_sees_every_evaluation_with_the_scored_weights() -> None:
    """
    Check that on_eval gets each evaluation while the model holds the scored weights.

    The model is in eval mode with the schedule-free average x in its
    parameters, so the val window scored inside the callback reproduces the
    val loss bit for bit.
    """
    model = _gpt()
    # The same one-window val part; scoring it leaves the rng of `loader` alone.
    probe = _loader()
    opt = ScheduleFreeAdamW(model.parameters())
    seen: list[tuple[Evaluation, float]] = []

    def on_eval(evaluation: Evaluation) -> None:
        assert not model.training
        assert opt.param_groups[0]["training"] is False
        seen.append((evaluation, _val_loss(model, probe)))

    loss, _ = train(
        model,
        _loader(),
        max_iters=20,
        eval_interval=5,
        eval_iters=1,
        use_tqdm=False,
        optimizer=opt,
        on_eval=on_eval,
    )
    assert [evaluation.step for evaluation, _ in seen] == [0, 5, 10, 15, 20]
    for evaluation, probed in seen:
        assert evaluation.val_loss == probed
    assert loss == seen[-1][0].val_loss


def test_on_eval_reports_no_val_loss_without_a_val_split() -> None:
    """
    Check that without a val split val_loss is None.

    train() then returns the train loss of the last evaluation.
    """
    model = _gpt()
    loader = TokenSequenceDataLoader(
        [i % 4 for i in range(N_TRAIN_TOKENS)],
        block_size=BLOCK_SIZE,
        batch_size=4,
        device="cpu",
        train_part=1.0,
    )
    seen: list[Evaluation] = []
    loss, _ = train(
        model,
        loader,
        max_iters=2,
        eval_interval=1,
        eval_iters=1,
        use_tqdm=False,
        on_eval=seen.append,
    )
    assert [e.val_loss for e in seen] == [None, None, None]
    assert loss == seen[-1].train_loss


def test_bfloat16_autocast_trains_float32_master_weights() -> None:
    """
    Check that autocast=torch.bfloat16 trains in mixed precision on float32 weights.

    The parameters, their gradients and the optimizer's z and v stay float32,
    the bfloat16 forward passes change the updates, and the evaluations run
    without autocast: the returned val loss is the float32 loss of the
    returned weights.
    """
    model = _gpt()
    optimizer = ScheduleFreeAdamW(model.parameters(), lr=1e-2)
    settings = {"max_iters": 20, "eval_interval": 10, "eval_iters": 1}
    loss, _ = train(
        model,
        _loader(),
        use_tqdm=False,
        optimizer=optimizer,
        autocast=torch.bfloat16,
        **settings,
    )
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    grads = {p.grad.dtype for p in model.parameters() if p.grad is not None}
    assert grads == {torch.float32}
    states = {t.dtype for state in optimizer.state.values() for t in state.values()}
    assert states == {torch.float32}
    assert loss == pytest.approx(_val_loss(model, _loader()), rel=1e-6)
    full = _gpt()
    train(
        full,
        _loader(),
        use_tqdm=False,
        optimizer=ScheduleFreeAdamW(full.parameters(), lr=1e-2),
        **settings,
    )
    assert not torch.equal(model.lm_head.weight, full.lm_head.weight)


def test_autocast_takes_bfloat16_only() -> None:
    """float16 autocast would need loss scaling, so train() refuses it."""
    with pytest.raises(ValueError, match="autocast must be None"):
        train(_gpt(), _loader(), max_iters=1, use_tqdm=False, autocast=torch.float16)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_train_refuses_half_precision_master_weights(dtype: torch.dtype) -> None:
    """A model cast to float16 or bfloat16 would round every update to it."""
    with pytest.raises(ValueError, match="parameters must be float32 or float64"):
        train(_gpt().to(dtype), _loader(), max_iters=1, use_tqdm=False)


def test_train_takes_float64_master_weights() -> None:
    """A float64 model trains, and its weights stay float64."""
    model = _gpt().double()
    train(model, _loader(), max_iters=2, eval_interval=1, eval_iters=1, use_tqdm=False)
    assert {p.dtype for p in model.parameters()} == {torch.float64}


# A warmup of 8 steps spans the break after 6 updates of 12.
RESUME_WARMUP = 8
BREAK = 6
RESUME_MAX_ITERS = 12


def _saved(
    model: GPT,
    opt: ScheduleFreeAdamW,
    loader: TokenSequenceDataLoader,
) -> dict[str, Any]:
    """Return the weights, the optimizer and the loader's rng, through a file."""
    buffer = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "loader": loader.rng.get_state(),
        },
        buffer,
    )
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def test_train_goes_on_from_a_saved_state_as_an_unbroken_run() -> None:
    """
    Check that 6 updates, saved and restored, then 6 more give the run of 12.

    The restored run skips the evaluation after 6 updates, which the first
    run made before saving, and makes the same evaluations after 9 and 12
    updates; it ends with the same weights and optimizer state, bit for bit.
    """
    seen: list[Evaluation] = []
    straight = _gpt()
    straight_opt = ScheduleFreeAdamW(straight.parameters(), warmup=RESUME_WARMUP)
    train(
        straight,
        _loader(),
        max_iters=RESUME_MAX_ITERS,
        eval_interval=3,
        eval_iters=1,
        use_tqdm=False,
        optimizer=straight_opt,
        on_eval=seen.append,
    )
    first = _gpt()
    first_opt = ScheduleFreeAdamW(first.parameters(), warmup=RESUME_WARMUP)
    first_loader = _loader()
    train(
        first,
        first_loader,
        max_iters=BREAK,
        eval_interval=3,
        eval_iters=1,
        use_tqdm=False,
        optimizer=first_opt,
    )
    saved = _saved(first, first_opt, first_loader)
    model = _gpt()
    opt = ScheduleFreeAdamW(model.parameters())
    loader = _loader()
    model.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    loader.rng.set_state(saved["loader"])
    resumed: list[Evaluation] = []
    train(
        model,
        loader,
        max_iters=RESUME_MAX_ITERS,
        eval_interval=3,
        eval_iters=1,
        use_tqdm=False,
        optimizer=opt,
        on_eval=resumed.append,
        start_step=BREAK,
    )
    assert [e.step for e in seen] == [0, 3, 6, 9, 12]
    assert resumed == [e for e in seen if e.step > BREAK]
    for name, value in straight.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name
    ends = [o.state_dict() for o in (straight_opt, opt)]
    assert ends[0]["param_groups"] == ends[1]["param_groups"]
    assert ends[0]["param_groups"][0]["step"] == RESUME_MAX_ITERS
    for k, state in ends[0]["state"].items():
        for key, value in state.items():
            assert torch.equal(ends[1]["state"][k][key], value), (k, key)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"save_best": True}, "use_early_stopping or save_best"),
        ({"use_early_stopping": True}, "use_early_stopping or save_best"),
        ({"optimizer": None}, "needs the optimizer"),
        ({"start_step": 6}, "must lie in"),
        ({"start_step": 2}, "has made 3 steps, not start_step = 2"),
    ],
)
def test_train_refuses_a_start_it_cannot_go_on_from(
    change: dict[str, Any],
    match: str,
) -> None:
    """
    A start that train() cannot go on from raises before any update.

    Early stopping and the best snapshot keep state that a saved run does
    not carry; a start needs the saved optimizer, with as many steps, and
    must leave updates to make.
    """
    model = _gpt()
    opt = ScheduleFreeAdamW(model.parameters())
    train(
        model,
        _loader(),
        max_iters=3,
        eval_interval=3,
        eval_iters=1,
        use_tqdm=False,
        optimizer=opt,
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    settings: dict[str, Any] = {"optimizer": opt, "start_step": 3} | change
    with pytest.raises(ValueError, match=match):
        train(
            model,
            _loader(),
            max_iters=6,
            eval_interval=3,
            eval_iters=1,
            use_tqdm=False,
            **settings,
        )
    for k, v in model.state_dict().items():
        assert torch.equal(before[k], v), k
