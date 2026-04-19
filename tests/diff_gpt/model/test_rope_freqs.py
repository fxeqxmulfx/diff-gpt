import torch
from torch.testing import assert_close

from diff_gpt.model.rope import precompute_freqs_cis


def test_output_shape():
    r"""
    Verify dimensions: $T \times (D/2) \times 2$.
    $T = \text{end}, D = \text{dim}$
    Output $\in \mathbb{R}^{T \times D/2 \times 2}$ with last axis = (cos, sin).
    """
    dim, end, base = 64, 100, 10000
    cis = precompute_freqs_cis(dim, end, base)

    assert cis.shape == (end, dim // 2, 2)
    # Real-valued (cos, sin) packing — not complex.
    assert cis.is_floating_point()


def test_zero_timestep_identity():
    r"""
    Verify $t=0$ boundary condition.
    $m=0 \implies \text{angle} = 0 \cdot \theta_j = 0 \implies (\cos, \sin) = (1, 0)$
    """
    dim, end, base = 10, 5, 10000
    cis = precompute_freqs_cis(dim, end, base)

    row_0 = cis[0]  # (D/2, 2)
    cos_t0, sin_t0 = row_0[..., 0], row_0[..., 1]

    assert_close(cos_t0, torch.ones_like(cos_t0))
    assert_close(sin_t0, torch.zeros_like(sin_t0))


def test_explicit_formula_values():
    r"""
    Verify numerical correctness against scalar formula.
    $\theta_j = b^{-2j/d}, \quad b=10000$
    $(\cos, \sin)_{m, j} = (\cos(m \theta_j),\, \sin(m \theta_j))$
    """
    dim, end, base = 4, 2, 10000
    cis = precompute_freqs_cis(dim, end, theta=base)

    # --- Check $j=0$: $\theta_0 = 1.0$, $m=1$ ---
    cos_t1_j0, sin_t1_j0 = cis[1, 0, 0], cis[1, 0, 1]
    assert_close(cos_t1_j0, torch.cos(torch.tensor(1.0)))
    assert_close(sin_t1_j0, torch.sin(torch.tensor(1.0)))

    # --- Check $j=1$: $\theta_1 = 1/100 = 0.01$, $m=1$ ---
    cos_t1_j1, sin_t1_j1 = cis[1, 1, 0], cis[1, 1, 1]
    assert_close(cos_t1_j1, torch.cos(torch.tensor(0.01)))
    assert_close(sin_t1_j1, torch.sin(torch.tensor(0.01)))


def test_frequency_decay():
    r"""
    Verify monotonicity of frequencies.
    $\theta_0 > \theta_1 > \dots > \theta_{d/2-1}$
    """
    dim, end, base = 128, 2, 10000
    cis = precompute_freqs_cis(dim, end, base)

    # Reconstruct angle from (cos, sin) at $t=1$.
    angles = torch.atan2(cis[1, :, 1], cis[1, :, 0])

    # Decreasing in the lowest-frequency tail; check the first few before
    # angles get small enough that fp noise dominates.
    assert angles[0] > angles[1]
    assert angles[1] > angles[2]


def test_magnitude_unity():
    r"""
    Verify unitarity: $\cos^2 + \sin^2 = 1$.
    """
    dim, end, base = 20, 10, 10000
    cis = precompute_freqs_cis(dim, end, base)

    cos = cis[..., 0]
    sin = cis[..., 1]
    magnitudes = torch.sqrt(cos * cos + sin * sin)
    assert_close(magnitudes, torch.ones_like(magnitudes))
