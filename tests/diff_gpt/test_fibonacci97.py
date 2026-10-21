"""
Fibonacci mod 97 across rows at derivative order 1 against a compute budget.

One column follows x[t+2] = x[t] + x[t+1] mod 97. Its 9,408 states (a, b)
other than (0, 0) fall into 48 cycles of 196, and a random half of the
cycles trains while the other half tests, so no test window, nor any shift
of it, is ever trained on: the task of `benchmarks/grokking/recurrence.py`
at k=1 with context 2, but trained by the library's own `train()`. `encode`
turns a window of four rows into its first row and three difference tokens,
anchored at that row as `DiffGPT.predict` anchors a context. The input is
the first two tokens and only the third is supervised: the model sees steps,
never values, and has to recover the level from the recurrence. The forecast
is the fourth row, decoded from the argmax token and rounded.

Division across rows, x[t+2] = x[t] / x[t+1], has no such test. At k >= 1
it generalizes only by copying the period with context 48, it stays near
chance with context 4 (README), and so does div97 behind a zero prefix
(`div97.py --order 1`): 1% test accuracy after 3,000 updates on the model
of test_div97.

The grid has vocabulary 512 and domain 96, scale 2.66: every row decodes
exactly, and a third of the tokens differ between the tokenizers, which
coincide on the integer grid (vocabulary 256, domain 127). The benchmark is
the training compute until the decoded test accuracy of the schedule-free
average weights first reaches 90%, for each tokenizer. Tokens would be unfair
to LGT: its targets carry the absolute rounding phase, and a token one level
off still rounds to the true value four times in five.

The model has width 128, one layer and one head: 262,016 parameters.
ScheduleFreeAdamW runs at lr 3e-2 (the fast Fibonacci setting of
grokking_plan.md), betas 0.9/0.98, weight decay 1 and ten warmup updates.

Historical ledger (seeds 11, 17 and 23 on CUDA in FP32 with a budget of 5,000
updates: the updates up to the first evaluation with decoded test accuracy
>= 90%, and their mean training FLOPs, GPT.estimate_flops() * trained tokens):
    # INIT:  MASH  updates 1300 / 1750 / 1350  flops_to_90=2.37e12
    #        LGT   updates 3300 / 3450 / 3250  flops_to_90=5.38e12
    #   final accuracy: MASH 93.8% to 94.4%, LGT 92.6% to 93.0%
    #   TF32 matmuls: MASH 1300 / 1750 / 2000, LGT 2800 / 2500 / 3050
    #   seeds 0-5: MASH 1100 to 1500 updates, LGT 2300 to 4250 (seed 1 ends
    #   at 89.97%)
    #   lr 1e-2: MASH 2600 / 3150 / 2950, LGT not reached (75% / 11% / 9%
    #   at the end)

The test runs all three seeds for each tokenizer, about 22 s per seed on an
RTX 3050 Laptop GPU: single seeds differ far more than a tolerance on one
seed could absorb. A rerun in FP32 repeats the ledger exactly. The tolerance
covers the numerics, which TF32 shows to move the means by 15% (MASH) and 17%
(LGT), yet catches a change of the training stack or of a tokenizer that
matters: LGT needs 2.3 times the compute of MASH.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from diff_gpt.encoder_decoder import decode, encode, level_grid
from diff_gpt.model.gpt import GPT
from diff_gpt.train import train

if TYPE_CHECKING:
    from diff_gpt.train import Evaluation

P = 97
N_STATES = P * P - 1  # every (a, b) but (0, 0), which stays at zero
ORDER = 1
VOCAB_SIZE = 512
DOMAIN = 96.0  # the steps of residues lie in [-96, 96]
BLOCK_SIZE = 2  # the input: the two observed steps
BATCH_SIZE = 512
IGNORE_INDEX = -100  # F.cross_entropy's default
TOKENIZERS = ("mash", "lgt")
SEEDS = (11, 17, 23)
MAX_ITERS = 5_000
EVAL_INTERVAL = 50
TARGET_ACCURACY = 0.9

# Every window gets this grid: the forecast of the step token q is the
# decoded last observed row plus (q - zero) / scale.
GRID = level_grid(VOCAB_SIZE, np.array([DOMAIN]), ORDER)
SCALE, ZERO = float(GRID.scale[0]), int(GRID.zero[0])

# Current benchmark (see the ledger): the mean over SEEDS on CUDA of the
# training FLOPs to TARGET_ACCURACY, and the compute of the whole run.
FLOPS_TO_TARGET_BASELINE = {"mash": 2.37e12, "lgt": 5.38e12}
FLOPS_TO_TARGET_TOLERANCE = 0.20  # relative
TRAIN_FLOPS_BASELINE = 8.06e12


def _cycles() -> list[list[tuple[int, int]]]:
    """Return the cycles of the state map (a, b) -> (b, a + b mod 97) but (0, 0)."""
    remaining = {(a, b) for a in range(P) for b in range(P)} - {(0, 0)}
    cycles = []
    while remaining:
        state = min(remaining)
        cycle = []
        while state in remaining:
            remaining.remove(state)
            cycle.append(state)
            state = (state[1], (state[0] + state[1]) % P)
        cycles.append(cycle)
    return cycles


def _windows(states: list[tuple[int, int]]) -> np.ndarray:
    """Return the rows a, b, a + b, a + 2b (mod 97) of each state (a, b)."""
    a, b = np.array(states, dtype=np.int64).T
    return np.stack((a, b, (a + b) % P, (a + 2 * b) % P), axis=1)


def _split(rng: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return the windows of a random half of the cycles, then of the other half."""
    cycles = _cycles()
    order = torch.randperm(len(cycles), generator=rng).tolist()
    half = len(order) // 2
    train_states = [state for index in order[:half] for state in cycles[index]]
    test_states = [state for index in order[half:] for state in cycles[index]]
    return _windows(train_states), _windows(test_states)


def _encode(windows: np.ndarray, tokenizer: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode each window as a series of its own; return its tokens and its base.

    The tokens are (n, 3): two observed steps and the forecast step. The
    base is the decoded last row for the step token ZERO, so the forecast of
    a step token q is base + (q - ZERO) / SCALE.
    """
    start, grid, tokens = encode(
        windows.T.astype(np.float64),
        VOCAB_SIZE,
        np.full(len(windows), DOMAIN),
        ORDER,
        tokenizer=tokenizer,
    )
    zero_step = tokens.copy()
    zero_step[-1] = ZERO
    base = decode(start, grid, zero_step, VOCAB_SIZE, ORDER, tokenizer=tokenizer)
    return tokens.T.copy(), base[-1]


class _RowsLoader:
    """
    Batches of windows: two observed steps, and the forecast step as their target.

    One generator seeded with `seed` splits the cycles and then draws the
    batches. The val split is the test half.
    """

    block_size = BLOCK_SIZE
    batch_size = BATCH_SIZE

    def __init__(self, seed: int, tokenizer: str, device: str) -> None:
        self.rng = torch.Generator().manual_seed(seed)
        self.tokens, self.base, self.values = {}, {}, {}
        for split, windows in zip(("train", "val"), _split(self.rng), strict=True):
            tokens, base = _encode(windows, tokenizer)
            self.tokens[split] = torch.as_tensor(tokens, device=device)
            self.base[split] = torch.as_tensor(base, device=device)
            self.values[split] = torch.as_tensor(windows[:, -1], device=device)

    def has_val(self) -> bool:
        return True

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokens[split]
        picks = torch.randint(len(tokens), (self.batch_size,), generator=self.rng)
        batch = tokens[picks.to(tokens.device)]
        targets = torch.full_like(batch[:, :BLOCK_SIZE], IGNORE_INDEX)
        targets[:, -1] = batch[:, -1]
        return batch[:, :BLOCK_SIZE], targets


def _gpt() -> GPT:
    return GPT(
        vocab_size=VOCAB_SIZE,
        n_embd=128,
        block_size=BLOCK_SIZE,
        n_head=1,
        n_layer=1,
        use_checkpoint=False,
    )


@torch.no_grad()
def _forecast_accuracy(model: GPT, loader: _RowsLoader, split: str = "val") -> float:
    """Return the share of `split` whose forecast from the argmax token is right."""
    logits, _ = model(loader.tokens[split][:, :BLOCK_SIZE])
    steps = logits[:, -1].argmax(dim=-1) - ZERO
    forecasts = torch.round(loader.base[split] + steps / SCALE)
    return float((forecasts == loader.values[split]).double().mean())


def _test_accuracy_curve(seed: int, tokenizer: str) -> list[tuple[int, float]]:
    """Train on CUDA with `seed`; return (updates, test accuracy) of each evaluation."""
    torch.manual_seed(seed)
    model = _gpt().to(device="cuda")
    loader = _RowsLoader(seed, tokenizer, device="cuda")
    curve: list[tuple[int, float]] = []

    def on_eval(evaluation: Evaluation) -> None:
        curve.append((evaluation.step, _forecast_accuracy(model, loader)))

    train(
        mut_model=model,
        loader=loader,
        learning_rate=3e-2,
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


def test_split_holds_whole_cycles() -> None:
    """Check that each half holds whole cycles: every shift of its windows."""
    train_windows, test_windows = _split(torch.Generator().manual_seed(0))
    train_states = {(int(a), int(b)) for a, b in train_windows[:, :2]}
    test_states = {(int(a), int(b)) for a, b in test_windows[:, :2]}
    assert len(train_states) == len(test_states) == N_STATES // 2
    assert len(train_states | test_states) == N_STATES
    assert {(b, (a + b) % P) for a, b in train_states} == train_states
    windows = np.concatenate([train_windows, test_windows])
    assert bool(((windows[:, :-2] + windows[:, 1:-1]) % P == windows[:, 2:]).all())


def test_batches_supervise_only_the_forecast_step() -> None:
    """Check that a train batch holds training windows, their last step the target."""
    loader = _RowsLoader(seed=0, tokenizer="mash", device="cpu")
    x, y = loader.get_batch("train")
    assert x.shape == y.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert bool((y[:, 0] == IGNORE_INDEX).all())
    windows = {tuple(row) for row in loader.tokens["train"].tolist()}
    assert all(
        (*row, target) in windows
        for row, target in zip(x.tolist(), y[:, -1].tolist(), strict=True)
    )


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_forecast_is_the_decoded_step(tokenizer: str) -> None:
    """
    Check the forecast against `decode` and the inputs against the future.

    The true step token decodes to the true last row, any step token q to
    base + (q - ZERO) / SCALE, and a changed last row changes its target only.
    """
    windows = _windows(_cycles()[0])
    tokens, base = _encode(windows, tokenizer)
    start, grid, _ = encode(
        windows.T.astype(np.float64),
        VOCAB_SIZE,
        np.full(len(windows), DOMAIN),
        ORDER,
        tokenizer=tokenizer,
    )
    for offset in (-1, 0, 1):
        stepped = tokens.T.copy()
        stepped[-1] += offset
        decoded = decode(start, grid, stepped, VOCAB_SIZE, ORDER, tokenizer=tokenizer)
        np.testing.assert_allclose(
            decoded[-1], base + (stepped[-1] - ZERO) / SCALE, rtol=0, atol=1e-12
        )
        if offset == 0:
            np.testing.assert_array_equal(np.rint(decoded), windows.T)
    changed = windows.copy()
    changed[:, -1] = (changed[:, -1] + 1) % P
    changed_tokens, _ = _encode(changed, tokenizer)
    np.testing.assert_array_equal(changed_tokens[:, :-1], tokens[:, :-1])
    assert bool((changed_tokens[:, -1] != tokens[:, -1]).all())


def test_tokenizers_differ_on_this_grid() -> None:
    """Check that MASH and LGT give different tokens for about a third of the steps."""
    windows = _windows(_cycles()[0]).T.astype(np.float64)
    domains = np.full(windows.shape[1], DOMAIN)
    mash = encode(windows, VOCAB_SIZE, domains, ORDER, tokenizer="mash")[2]
    lgt = encode(windows, VOCAB_SIZE, domains, ORDER, tokenizer="lgt")[2]
    assert float(np.mean(mash != lgt)) == pytest.approx(1 / 3, abs=0.1)


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the baselines are measured on CUDA",
)
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_fibonacci97_flops_to_generalize(tokenizer: str) -> None:
    """
    Check the training FLOPs to 90% decoded test accuracy against the benchmark.

    End-to-end regression guard for the codec at k=1, the model, the
    optimizer and `train()`, next to `test_div97` at k=0. A drift in either
    direction beyond the tolerance means that the training stack or the
    tokenizer changed; update the baselines and the ledger above together
    when it changes deliberately. Matmuls run in full FP32, as measured,
    whatever an earlier test set.
    """
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        curves = [_test_accuracy_curve(seed, tokenizer) for seed in SEEDS]
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
    baseline = FLOPS_TO_TARGET_BASELINE[tokenizer]
    assert abs(flops / baseline - 1) < FLOPS_TO_TARGET_TOLERANCE, (
        f"FLOPs to {TARGET_ACCURACY:.0%} {flops:.3g} (updates {crossings}) "
        f"drifted from the {baseline:.3g} baseline"
    )
