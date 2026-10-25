"""Tests for rotary position embeddings (RoPE)."""

import torch
from torch.testing import assert_close

from diff_gpt.domain.model.rope import apply_rope, rope_tables

HEAD_DIM = 8
N_POS = 16


def test_rope_tables_start_at_identity_and_turn_pair_0_a_radian_per_step() -> None:
    """Position 0 does not rotate; pair 0 turns by one radian per position."""
    cos, sin = rope_tables(HEAD_DIM, N_POS)

    assert cos.shape == sin.shape == (N_POS, HEAD_DIM // 2)
    assert torch.equal(cos[0], torch.ones(HEAD_DIM // 2))
    assert torch.equal(sin[0], torch.zeros(HEAD_DIM // 2))
    pos = torch.arange(N_POS, dtype=torch.float32)
    assert_close(cos[:, 0], pos.cos())
    assert_close(sin[:, 0], pos.sin())


def test_apply_rope_pairs_channel_i_with_i_plus_half() -> None:
    """Half-split layout: channel 0 turns into channel D/2, nowhere else."""
    cos, sin = rope_tables(HEAD_DIM, N_POS)
    x = torch.zeros(N_POS, HEAD_DIM)
    x[:, 0] = 1.0

    expected = torch.zeros(N_POS, HEAD_DIM)
    expected[:, 0] = cos[:, 0]
    expected[:, HEAD_DIM // 2] = sin[:, 0]
    assert_close(apply_rope(x, cos, sin), expected)


def test_apply_rope_keeps_norms() -> None:
    """A rotation keeps the norm of every query and key."""
    torch.manual_seed(0)
    cos, sin = rope_tables(HEAD_DIM, N_POS)
    x = torch.randn(2, 3, N_POS, HEAD_DIM)  # (B, H, T, D)

    assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_scores_depend_only_on_the_offset() -> None:
    """
    Check that q·k after RoPE depends on the positions only through their offset.

    Shifting the query and the key by the same number of positions keeps the
    score; moving one of them changes it.
    """
    torch.manual_seed(0)
    cos, sin = rope_tables(HEAD_DIM, N_POS)
    q = torch.randn(HEAD_DIM)
    k = torch.randn(HEAD_DIM)

    def score(m: int, n: int) -> torch.Tensor:
        return apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n])

    m, n, shift = 5, 2, 7
    assert_close(score(m + shift, n + shift), score(m, n))
    assert not torch.allclose(score(m + 1, n), score(m, n))
