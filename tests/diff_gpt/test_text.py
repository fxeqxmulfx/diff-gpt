"""
Char-level GPT training on tinyshakespeare against loss and compute budgets.

Verifies that the full training loop actually optimizes, at a pinned training
compute.

Historical ledger (average over 3 runs unless noted, benchmark config,
max_iters=5000; lr=1e-2, weight decay 0.1 and no warmup until entries change
them). Runs with a fixed seed are deterministic on CUDA, so newer entries
average seeds 0-2 there and report training FLOPs (GPT.estimate_flops() *
trained tokens) instead of wall-clock seconds:
    # INIT:                            val_loss=1.82  train_time=188
    # ADD torch.compile:                val_loss=1.83  train_time=49.67
    # UPD eval_interval = 5_000:        val_loss=1.82  train_time=40.67
    # ADD flash-attention:              val_loss=1.80  train_time=26.67
    # UPD bias=False in lm_head:        val_loss=1.82  train_time=28.67
    # ADD RMSNorm:                      val_loss=1.82  train_time=28.0
    # ADD RoPE:                         val_loss=1.76  train_time=28.67
    # ADD SwiGLU:                       val_loss=1.74  train_time=29.67
    # ADD AdamWScheduleFree:            val_loss=1.68  train_time=28.0
    # ADD KQ norm and Gated Attention:  val_loss=1.67  train_time=43.0
    # CLR project:                      val_loss=1.68  train_time=37.33
    # ADD AMSGrad default (1 run):      val_loss=1.73  train_time=~110 (CPU)
    # ADD Block AttnRes, 1 layer/block:  val_loss=1.72  train_time=~155 (CPU)
    # UPD plain GPT (1 run):            val_loss=1.99  train_time=64 (CPU)
    #   same model at lr=3e-3: val_loss=1.83; at lr=1e-3: val_loss=1.96
    #   same model on CUDA:             val_loss=1.90  train_flops=3.45e12
    #   (lr=3e-3: 1.84; lr=1e-3: 1.97)
    # UPD no biases:                    val_loss=1.88  train_flops=3.41e12
    #   same model at lr=3e-3: val_loss=1.84; at lr=1e-3: val_loss=1.98
    # UPD RMSNorm without parameters:   val_loss=2.02  train_flops=3.40e12
    #   seeds 0-2: 2.22, 1.89, 1.94 (seed 0 is unstable at lr=1e-2)
    #   same model at lr=3e-3: val_loss=1.89; at lr=1e-3: val_loss=2.01
    # UPD benchmark lr 1e-2 -> 3e-3:    val_loss=1.89  train_flops=3.40e12
    #   seeds 0-2: 1.92, 1.86, 1.88
    # ADD RoPE (rotate_half):           val_loss=1.79  train_flops=3.40e12
    #   seeds 0-2: 1.77, 1.81, 1.80
    #   same model at lr=1e-2: val_loss=1.82; at lr=1e-3: val_loss=1.88
    #   interleaved pairs instead: val_loss=1.78 (seeds 1.78, 1.78, 1.78)
    # UPD optimizer ScheduleFreeAdamW:  val_loss=1.76  train_flops=3.40e12
    #   seeds 0-2: 1.77, 1.76, 1.76 (lean_optim's update, no AMSGrad)
    # UPD lr, weight decay and warmup:  val_loss=1.66  train_flops=3.40e12
    #   lr 3e-3 -> 1e-2, weight decay 0.1 -> 0.03, warmup 0 -> 1000 steps
    #   seeds 0-2: 1.65, 1.66, 1.66
    #   best of a seed-0 grid over lr 1e-3..2e-2, wd 0..1, warmup 0..2000
    # ADD XSA and SwiGLU:               val_loss=1.64  train_flops=3.40e12
    #   seeds 0-2: 1.64, 1.65, 1.64; means to 3 decimals: 1.660 before,
    #   1.655 with XSA alone, 1.645 with SwiGLU alone, 1.640 with both
    #   SwiGLU hidden 171 = 2/3 of 256: params 204_800 -> 205_056
    # UPD Peri-RMSNorm and Q/K RMSNorm: val_loss=1.63  train_flops=3.40e12
    #   seeds 0-2: 1.631, 1.631, 1.629; mean 1.630 (1.640 before)
    #   seed 0 of the previous model, re-measured here: 1.637
    # UPD schedulefree >= 1.3 update:   val_loss=1.63  train_flops=3.40e12
    #   bias correction on v, sqrt(v / (1 - β2^t)) + eps, not on the rate;
    #   averaging weights from the largest rate so far; z and v stored, the
    #   parameters hold y or x; Meta's AdamWScheduleFree: same losses bitwise
    #   seeds 0-2: 1.635, 1.632, 1.627; mean 1.631 (1.630 before)
    #   the same update with x stored rounds differently: mean 1.634

The test below is one run of the benchmark config (seed 0). Its loss tolerance
covers the spread between seeds (about 0.01) and hardware, yet is tight enough
to catch an optimization regression.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from diff_gpt.data_loader import TokenSequenceDataLoader
from diff_gpt.model.gpt import GPT
from diff_gpt.train import train

if TYPE_CHECKING:
    from collections.abc import Iterable

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"
PAD_TO = 64

# Current benchmark (see the ledger): mean val_loss over seeds 0-2 on CUDA and
# the training compute of the benchmark config.
VAL_LOSS_BASELINE = 1.63
VAL_LOSS_TOLERANCE = 0.05
TRAIN_FLOPS_BASELINE = 3.40e12


def _ceil_to(n: int, pad: int) -> int:
    return ((n + pad - 1) // pad) * pad


class CharEncoder:
    """
    Character-level encoder/decoder.

    Pads vocab_size up to a multiple of `pad_to` so it matches the GPT
    alignment invariant; tokens are only produced in the first
    `real_vocab_size` slots.
    """

    def __init__(self, text: str, pad_to: int = PAD_TO) -> None:
        """Build the vocabulary from the characters of `text`."""
        chars = "".join(sorted(set(text)))
        self.real_vocab_size = len(chars)
        self.vocab_size = _ceil_to(self.real_vocab_size, pad_to)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    def encode(self, text: str) -> tuple[int, ...]:
        """Map each character of `text` to its token id."""
        stoi = self.stoi
        return tuple(stoi[ch] for ch in text)

    def decode(self, tokens: Iterable[int]) -> str:
        """Map token ids back to text."""
        itos = self.itos
        # Model may sample padded ids; ignore them.
        return "".join(itos[t] for t in tokens if t in itos)


def test_char_encoder_pads_vocab_to_multiple_of_64() -> None:
    """Check that the vocabulary is padded to a multiple of 64 and round-trips."""
    text = "abcde"
    enc = CharEncoder(text, pad_to=PAD_TO)
    assert enc.real_vocab_size == len(text)
    assert enc.vocab_size == PAD_TO
    assert enc.vocab_size % PAD_TO == 0
    assert enc.decode(enc.encode(text)) == text


def test_char_encoder_decode_drops_padded_ids() -> None:
    """Check that decode skips the padded ids."""
    enc = CharEncoder("ab", pad_to=PAD_TO)
    # Real ids are 0, 1; ids >= 2 are padding and must be dropped on decode.
    assert enc.decode([0, 1, 5, 63, 1]) == "abb"


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the baseline is measured on CUDA",
)
def test_text_training_flops_and_loss() -> None:
    """
    Check val_loss and training FLOPs against the current benchmark.

    End-to-end regression guard: the GPT trained at lr=1e-2 with weight
    decay 0.03 and 1_000 warmup steps reaches val_loss ≈ 1.63 (mean over
    seeds 0-2 on CUDA) on the full tinyshakespeare run (5_000 iters) for
    3.40e12 training FLOPs. These settings are the best of a grid over lr,
    weight decay and warmup, run before XSA and SwiGLU (see the ledger).
    A drift in either direction beyond the tolerance means something in the
    training stack changed — intentional or not. Update the baselines and
    the ledger above together when the architecture changes deliberately.
    """
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    text = DATA_PATH.read_text(encoding="utf-8")
    encoder = CharEncoder(text, pad_to=PAD_TO)
    assert encoder.vocab_size % PAD_TO == 0
    encoded = encoder.encode(text)

    model = GPT(
        vocab_size=encoder.vocab_size,
        n_embd=64,
        block_size=32,
        n_head=4,
        n_layer=4,
    ).to(device="cuda")
    loader = TokenSequenceDataLoader(
        tokens=encoded,
        block_size=model.block_size,
        batch_size=16,
        device=model.device_type,
        train_part=0.9,
    )
    max_iters = 5_000
    val_loss, _ = train(
        mut_model=model,
        loader=loader,
        learning_rate=1e-2,
        betas=(0.9, 0.95),
        weight_decay=0.03,
        max_iters=max_iters,
        eval_interval=5_000,
        eval_iters=200,
        warmup=1_000,
        use_tqdm=False,
    )

    # Every iteration trains on batch_size * block_size tokens.
    tokens = max_iters * loader.batch_size * loader.block_size
    train_flops = model.estimate_flops() * tokens
    assert train_flops == pytest.approx(TRAIN_FLOPS_BASELINE, rel=0.01)
    assert abs(val_loss - VAL_LOSS_BASELINE) < VAL_LOSS_TOLERANCE, (
        f"val_loss {val_loss:.3f} drifted from {VAL_LOSS_BASELINE} baseline"
    )
