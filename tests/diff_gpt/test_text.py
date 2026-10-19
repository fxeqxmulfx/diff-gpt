# Copyright 2025 Mikhail Borisov
"""
Char-level GPT training on tinyshakespeare against loss and compute budgets.

Verifies that the full training loop actually optimizes, at a pinned training
compute.

Historical ledger (average over 3 runs unless noted, benchmark config,
max_iters=5000). Runs with a fixed seed are deterministic on CUDA, so newer
entries average seeds 0-2 there and report training FLOPs
(GPT.estimate_flops() * trained tokens) instead of wall-clock seconds:
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

The test below is one run of the benchmark config (seed 0). Its loss tolerance
covers the spread between seeds (about 0.02) and hardware, yet is tight enough
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
VAL_LOSS_BASELINE = 1.88
VAL_LOSS_TOLERANCE = 0.05
TRAIN_FLOPS_BASELINE = 3.41e12


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
@pytest.mark.skipif(not DATA_PATH.exists(), reason=f"{DATA_PATH} not available")
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the baseline is measured on CUDA",
)
def test_text_training_flops_and_loss() -> None:
    """
    Check val_loss and training FLOPs against the current benchmark.

    End-to-end regression guard: the GPT at the default lr=1e-2 reaches
    val_loss ≈ 1.88 (mean over seeds 0-2 on CUDA) on the full tinyshakespeare
    run (5_000 iters) for 3.41e12 training FLOPs. The earlier model (QK-norm,
    RoPE, SwiGLU, ...) reached 1.73 at this lr; the plain GPT does about as
    well at lr=3e-3 (1.84).
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
        weight_decay=0.1,
        max_iters=max_iters,
        eval_interval=5_000,
        eval_iters=200,
        use_tqdm=False,
    )

    # Every iteration trains on batch_size * block_size tokens.
    tokens = max_iters * loader.batch_size * loader.block_size
    train_flops = model.estimate_flops() * tokens
    assert train_flops == pytest.approx(TRAIN_FLOPS_BASELINE, rel=0.01)
    assert abs(val_loss - VAL_LOSS_BASELINE) < VAL_LOSS_TOLERANCE, (
        f"val_loss {val_loss:.3f} drifted from {VAL_LOSS_BASELINE} baseline"
    )
