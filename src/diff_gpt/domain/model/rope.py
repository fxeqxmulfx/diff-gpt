"""Rotary position embeddings (RoPE) of the attention queries and keys."""

import torch


def rope_tables(
    head_dim: int,
    n_pos: int,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the cos and sin of the RoPE angles, each of shape (n_pos, head_dim / 2).

    At position p, channel pair i turns by p · base^(-2i / head_dim): pair 0
    by a radian per position, the last pairs by almost nothing.
    """
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    angles = torch.outer(torch.arange(n_pos, dtype=torch.float32), base**-exponents)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotate every channel pair of `x` (..., T, D) by the angle of its position.

    Pair i is channels i and i + D/2, the half-split layout of GPT-NeoX and
    LLaMA. `cos` and `sin` are the `rope_tables` rows of the T positions. The
    rotation keeps norms, and the dot product of a rotated query and key
    depends on their positions only through the offset between them.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
