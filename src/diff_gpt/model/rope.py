import torch


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding to `x` using precomputed (cos, sin)
    pairs in `freqs_cis`.

    `x`: (..., D), D even. The last axis is split into D/2 pairs; each
    pair (a, b) is rotated as a 2D vector by the corresponding (cos, sin)
    angle.
    `freqs_cis`: (..., D/2, 2) with the last dim packing (cos, sin).
    Broadcasts against `x`'s leading dims.

    Real-valued throughout — identical semantics to the conventional
    complex64 form (a+bi)·(cos+i·sin), but with no complex tensors. Plays
    well with backends that don't support complex broadcast (gloo, some
    compile passes, tracing tools).
    """
    x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (..., D/2, 2)
    x_re, x_im = x_shaped.unbind(-1)  # each (..., D/2)
    cos = freqs_cis[..., 0]
    sin = freqs_cis[..., 1]
    out_re = x_re * cos - x_im * sin
    out_im = x_re * sin + x_im * cos
    out = torch.stack([out_re, out_im], dim=-1).flatten(-2)  # (..., D)
    return out.type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    """
    Precompute the (cos, sin) table for rotary embeddings.

    Returns a real tensor of shape (end, dim/2, 2): at row m, col j the
    pair is (cos(m·θ_j), sin(m·θ_j)) with θ_j = theta^(-2j/dim) — same
    angles as the canonical complex polar form, just stored as real
    pairs.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim)
    )
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (end, dim/2)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # (end, dim/2, 2)
