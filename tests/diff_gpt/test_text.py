"""
Char-level GPT training on tinyshakespeare. Verifies that the full training
loop actually optimizes and stays within speed and loss budgets.

Historical ledger (average over 3 runs unless noted, benchmark config,
max_iters=5000):
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

The test below uses a reduced scale for CI friendliness; the assertions are
loose enough to accommodate machine variance but tight enough to catch a
regression in either training speed or optimization quality.
"""

from pathlib import Path

import pytest
import torch

from diff_gpt.model.gpt import GPT
from diff_gpt.train import train


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"


def _ceil_to(n: int, pad: int) -> int:
    return ((n + pad - 1) // pad) * pad


class CharEncoder:
    """
    Character-level encoder/decoder. Pads vocab_size up to a multiple of
    `pad_to` so it matches the GPT alignment invariant; tokens are only
    produced in the first `real_vocab_size` slots.
    """

    def __init__(self, text: str, pad_to: int = 64) -> None:
        chars = "".join(sorted(set(text)))
        self.real_vocab_size = len(chars)
        self.vocab_size = _ceil_to(self.real_vocab_size, pad_to)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> tuple[int, ...]:
        stoi = self.stoi
        return tuple(stoi[ch] for ch in text)

    def decode(self, tokens) -> str:
        itos = self.itos
        # Model may sample padded ids; ignore them.
        return "".join(itos[t] for t in tokens if t in itos)


def test_char_encoder_pads_vocab_to_multiple_of_64():
    enc = CharEncoder("abcde", pad_to=64)
    assert enc.real_vocab_size == 5
    assert enc.vocab_size == 64
    assert enc.vocab_size % 64 == 0
    assert enc.decode(enc.encode("abcde")) == "abcde"


def test_char_encoder_decode_drops_padded_ids():
    enc = CharEncoder("ab", pad_to=64)
    # Real ids are 0, 1; ids >= 2 are padding and must be dropped on decode.
    assert enc.decode([0, 1, 5, 63, 1]) == "abb"


@pytest.mark.skipif(not DATA_PATH.exists(), reason=f"{DATA_PATH} not available")
def test_text_training_speed_and_loss():
    """
    End-to-end regression guard against the current benchmark:
      val_loss ≈ 1.73 on the full tinyshakespeare run (5_000 iters) with
      AMSGrad on by default (was 1.68 with plain Adam — AMSGrad pays a
      small accuracy cost for its non-increasing effective-LR guarantee).
    A drift in either direction beyond the tolerance means something in the
    training stack changed — intentional or not. Update the tolerance and
    the ledger above together when the architecture changes deliberately.
    """
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    encoder = CharEncoder(text, pad_to=64)
    assert encoder.vocab_size % 64 == 0
    encoded = encoder.encode(text)

    model = GPT(
        vocab_size=encoder.vocab_size,
        n_embd=64,
        block_size=32,
        n_head=4,
        n_layer=4,
    )
    val_loss, train_time_s = train(
        mut_model=model,
        encoded_data=encoded,
        learning_rate=1e-2,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        max_iters=5_000,
        eval_interval=5_000,
        eval_iters=200,
        batch_size=16,
        train_part=0.9,
        use_tqdm=False,
    )

    # Current baseline is 1.73 with AMSGrad on. Allow ±0.05 for single-run
    # / hardware variance.
    assert abs(val_loss - 1.73) < 0.05, (
        f"val_loss {val_loss:.3f} drifted from 1.73 baseline"
    )
    # Historical wall-clock was ~37s on the reference GPU; leave generous
    # headroom for CPU / slower hardware without making the test useless.
    assert train_time_s < 300, f"training too slow: {train_time_s}s"
