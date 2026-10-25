"""
Modular division mod 97 against a compute budget for generalization.

A GPT learns q = a * b^(-1) mod 97 from a random half of the 9,312 equations
(a in 0..96, b in 1..96) and answers the other half: the task of
`benchmarks/grokking/div97.py`, but trained by the library's own `train()`.
The input is [a, b] and only q is supervised; batches of 512 are drawn with
replacement. The benchmark is the training compute until the test accuracy of
the schedule-free average weights first reaches 99%. The final accuracy only
shows that the model stays there: it saturates at 100%.

The model is that of test_text's last ledger entry at width 128, with one
layer and four heads: 212,864 parameters. ScheduleFreeAdamW runs at lr 1e-2,
betas 0.9/0.98, weight decay 1 and ten warmup updates, as in the grokking
benchmark.

Historical ledger (seeds 11, 17 and 23 on CUDA in FP32 with a budget of 1,000
updates: the updates up to the first evaluation with test accuracy >= 99%,
and their mean training FLOPs, GPT.estimate_flops() * trained tokens):
    # INIT:                updates 400 / 430 / 360  flops_to_99=5.20e11
    #   TF32 matmuls: 390 / 440 / 340; seeds 0-8: 150 to 550 updates
    #   weight decay 0.3 instead of 1: 1000 / not reached / 870 updates

The test below runs all three seeds, about 6 s each on an RTX 3050 Laptop
GPU: single seeds differ far more than a tolerance on one seed could absorb.
Its tolerance covers the numerics, which TF32 shows to move the mean by 2%,
yet catches a change of the training stack that matters.
"""

from typing import TYPE_CHECKING

import pytest
import torch

from diff_gpt.application.train import train
from diff_gpt.domain.model.gpt import GPT

if TYPE_CHECKING:
    from diff_gpt.application.train import Evaluation

P = 97
N_TRAIN = P * (P - 1) // 2  # half of the 9,312 equations
VOCAB_SIZE = 128  # the residues 0..96, padded to a multiple of 64
BLOCK_SIZE = 2  # the input [a, b]
BATCH_SIZE = 512
IGNORE_INDEX = -100  # F.cross_entropy's default
SEEDS = (11, 17, 23)
MAX_ITERS = 1_000
EVAL_INTERVAL = 10
TARGET_ACCURACY = 0.99

# Current benchmark (see the ledger): the mean over SEEDS on CUDA of the
# training FLOPs to TARGET_ACCURACY, and the compute of the whole run.
FLOPS_TO_TARGET_BASELINE = 5.20e11
FLOPS_TO_TARGET_TOLERANCE = 0.10  # relative
TRAIN_FLOPS_BASELINE = 1.31e12


def _equations(rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rows (a, b, q) of every equation, split at random into two halves."""
    rows = torch.tensor(
        [(a, b, a * pow(b, -1, P) % P) for a in range(P) for b in range(1, P)],
    )
    order = torch.randperm(len(rows), generator=rng)
    return rows[order[:N_TRAIN]], rows[order[N_TRAIN:]]


class _DivisionLoader:
    """
    Batches of division equations: the input [a, b] and q as its only target.

    One generator seeded with `seed` splits the equations and then draws the
    batches. The val split is the test half.
    """

    block_size = BLOCK_SIZE
    batch_size = BATCH_SIZE

    def __init__(self, seed: int, device: str) -> None:
        self.rng = torch.Generator().manual_seed(seed)
        train_rows, test_rows = _equations(self.rng)
        self.rows = {"train": train_rows.to(device), "val": test_rows.to(device)}

    def has_val(self) -> bool:
        return True

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        rows = self.rows[split]
        picks = torch.randint(len(rows), (self.batch_size,), generator=self.rng)
        batch = rows[picks.to(rows.device)]
        targets = torch.full_like(batch[:, :BLOCK_SIZE], IGNORE_INDEX)
        targets[:, -1] = batch[:, -1]
        return batch[:, :BLOCK_SIZE], targets


def _gpt() -> GPT:
    return GPT(
        vocab_size=VOCAB_SIZE,
        n_embd=128,
        block_size=BLOCK_SIZE,
        n_head=4,
        n_layer=1,
        use_checkpoint=False,
    )


@torch.no_grad()
def _accuracy(model: GPT, rows: torch.Tensor) -> float:
    """Return the share of `rows` whose q is the argmax of the logits after [a, b]."""
    logits, _ = model(rows[:, :BLOCK_SIZE])
    return float((logits[:, -1].argmax(dim=-1) == rows[:, -1]).float().mean())


def _test_accuracy_curve(seed: int) -> list[tuple[int, float]]:
    """Train on CUDA with `seed`; return (updates, test accuracy) of each evaluation."""
    torch.manual_seed(seed)
    model = _gpt().to(device="cuda")
    loader = _DivisionLoader(seed, device="cuda")
    curve: list[tuple[int, float]] = []

    def on_eval(evaluation: Evaluation) -> None:
        curve.append((evaluation.step, _accuracy(model, loader.rows["val"])))

    train(
        mut_model=model,
        loader=loader,
        learning_rate=1e-2,
        betas=(0.9, 0.98),
        weight_decay=1.0,
        max_iters=MAX_ITERS,
        eval_interval=EVAL_INTERVAL,
        eval_iters=1,
        warmup=10,
        use_tqdm=False,
        on_eval=on_eval,
    )
    return curve


def test_split_holds_every_division_once() -> None:
    """Check that the two halves hold each of the 9,312 equations once, q = a / b."""
    train_rows, test_rows = _equations(torch.Generator().manual_seed(0))
    rows = torch.cat([train_rows, test_rows])
    assert len(train_rows) == len(test_rows) == N_TRAIN
    assert len({(int(a), int(b)) for a, b, _ in rows}) == P * (P - 1)
    assert bool(((rows[:, 2] * rows[:, 1] - rows[:, 0]) % P == 0).all())


def test_batches_supervise_only_the_answer() -> None:
    """Check that a train batch holds training equations with q as the only target."""
    loader = _DivisionLoader(seed=0, device="cpu")
    x, y = loader.get_batch("train")
    assert x.shape == y.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert bool((y[:, 0] == IGNORE_INDEX).all())
    assert bool(((y[:, 1] * x[:, 1] - x[:, 0]) % P == 0).all())
    train_pairs = {(int(a), int(b)) for a, b, _ in loader.rows["train"]}
    assert all((int(a), int(b)) in train_pairs for a, b in x)


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the baseline is measured on CUDA",
)
def test_div97_flops_to_generalize() -> None:
    """
    Check the training FLOPs to 99% test accuracy against the current benchmark.

    End-to-end regression guard for the model, the optimizer and `train()`
    on an algorithmic task, next to `test_text`. A drift in either direction
    beyond the tolerance means that the training stack changed; update the
    baselines and the ledger above together when it changes deliberately.
    Matmuls run in full FP32, as measured, whatever an earlier test set.
    """
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        curves = [_test_accuracy_curve(seed) for seed in SEEDS]
    finally:
        torch.set_float32_matmul_precision(precision)

    # Every update trains on batch_size * block_size tokens.
    flops_per_update = _gpt().estimate_flops() * BATCH_SIZE * BLOCK_SIZE
    assert flops_per_update * MAX_ITERS == pytest.approx(TRAIN_FLOPS_BASELINE, rel=0.01)
    crossings = []
    for seed, curve in zip(SEEDS, curves, strict=True):
        reached = [step for step, accuracy in curve if accuracy >= TARGET_ACCURACY]
        assert reached, f"seed {seed} never reached {TARGET_ACCURACY:.0%}"
        assert curve[-1][1] >= TARGET_ACCURACY, (
            f"seed {seed} ended at test accuracy {curve[-1][1]:.2%}"
        )
        crossings.append(reached[0])
    flops = flops_per_update * sum(crossings) / len(crossings)
    assert abs(flops / FLOPS_TO_TARGET_BASELINE - 1) < FLOPS_TO_TARGET_TOLERANCE, (
        f"FLOPs to {TARGET_ACCURACY:.0%} {flops:.3g} (updates {crossings}) "
        f"drifted from the {FLOPS_TO_TARGET_BASELINE:.3g} baseline"
    )
