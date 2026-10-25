"""
MASH tokenizer tests (tokenizer="mash"): the level grid anchored at the prefix.

MASH keeps README Theorem 1 as LGT does: every token lies in [1, V-1], or in
[0, V-1] at k = 0, and the error stays within Δ/2 = D/(V - 2^k), or D/V at
k = 0, which `_bound` gives. The level-grid encoder (the package default) is
covered by test_lgt.py.
"""

import warnings
from functools import partial

import numpy as np
import pytest

from diff_gpt.domain.encoder_decoder import decode as _decode
from diff_gpt.domain.encoder_decoder import encode as _encode
from diff_gpt.domain.encoder_decoder import get_domain_of_definition

encode = partial(_encode, tokenizer="mash")
decode = partial(_decode, tokenizer="mash")


def _bound(D: float, V: int, k: int) -> float:
    """Return Δ/2 = 1/(2s) = D / (V - 2^k), or D/V at k = 0, README Theorem 1."""
    return D / (V - (2**k if k else 0))


def _sin_cos(idx: np.ndarray) -> np.ndarray:
    return np.stack((np.sin(idx), np.cos(idx)), axis=1)


def _round_trip(
    inp: np.ndarray,
    vocab_size: int,
    order: int | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode `inp` on its own domain, where no token clips, and decode it."""
    dod = get_domain_of_definition(inp, order)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        start, grid, tokens = encode(inp, vocab_size, dod, order)
    return dod, tokens, decode(start, grid, tokens, vocab_size, order)


def _assert_theorem_1(
    inp: np.ndarray,
    vocab_size: int,
    order: int | np.ndarray,
) -> np.ndarray:
    """
    Check README Theorem 1 on `inp` and return the per-column errors.

    Every token lies in [1, V-1], or [0, V-1] at k = 0, the start rows come
    back as given, and each column stays within its Δ/2. Columns with D = 0
    at k >= 1 must come back exactly: the anchor passes through their prefix
    values.
    """
    dod, tokens, dec = _round_trip(inp, vocab_size, order)
    orders = np.broadcast_to(order, dod.shape)
    assert (tokens.min(axis=0, initial=vocab_size) >= (orders > 0)).all()
    assert tokens.max(initial=0) <= vocab_size - 1
    max_k = inp.shape[0] - tokens.shape[0]
    assert np.array_equal(dec[:max_k], inp[:max_k])
    err = np.abs(dec - inp).max(axis=0)
    for c in range(inp.shape[1]):
        bound = _bound(dod[c], vocab_size, int(orders[c]))
        assert err[c] <= bound * (1 + 1e-12), f"column {c}: {err[c]} > Δ/2 = {bound}"
    return err


@pytest.mark.slow
@pytest.mark.parametrize(
    ("order", "vocab_size"),
    [(0, 256), (1, 256), (2, 4096), (3, 65536)],
)
def test_diff_encoder_decoder_sin_cos(order: int, vocab_size: int) -> None:
    """Check Theorem 1 on 10^6 samples of sin/cos."""
    _assert_theorem_1(
        _sin_cos(np.arange(1_000_000, dtype=np.float64)), vocab_size, order
    )


def test_diff_encoder_decoder_finely_sampled_period() -> None:
    """
    Check Theorem 1 at order 3 on one sin/cos period sampled every 10^-3.

    Δ³x is about 10^-9, so the grid step 1/s is about 10^-11. Sampled every
    10^-6, Δ³x would fall below the float64 noise of the values, and the
    grid below what float64 resolves (see test_encoder_contracts.py).
    """
    _assert_theorem_1(_sin_cos(np.arange(0, 2 * np.pi, 1e-3, dtype=np.float64)), 256, 3)


@pytest.mark.slow
def test_diff_encoder_sin_cos_different_lenght() -> None:
    """
    Check that encoding a prefix keeps the start, scale and leading tokens.

    The anchor depends only on the first k rows, so the tokens of the first
    16 rows do not depend on the rows after them.
    """
    vocab_size = 256
    inp = _sin_cos(np.arange(1_000_000, dtype=np.float64))
    dod = get_domain_of_definition(inp, 1)
    start, grid, tokens = encode(inp, vocab_size, dod, 1)
    start_2, grid_2, tokens_2 = encode(inp[:16], vocab_size, dod, 1)
    assert np.array_equal(start, start_2)
    assert np.array_equal(grid.scale, grid_2.scale)
    assert np.array_equal(grid.zero, grid_2.zero)
    assert np.array_equal(tokens[: len(tokens_2)], tokens_2)


@pytest.mark.slow
def test_diff_encoder_decoder_lin() -> None:
    """Check that order 1 restores linear series exactly with 4 tokens: D = 1, s = 1."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack((idx, np.flip(idx)), axis=1)
    _, _, dec = _round_trip(inp, 4, 1)
    assert np.array_equal(dec, inp)


@pytest.mark.slow
@pytest.mark.parametrize("value", [1.0, 0.01])
def test_diff_encoder_decoder_const(value: float) -> None:
    """
    Check that order 1 restores a constant series exactly with 4 tokens.

    D = 0 gives the scale 2^20, and the anchor holds the fraction of 2^20·x,
    which LGT rounds away: it decodes 0.01 with an error of 2.3·10^-7.
    """
    inp = np.full((1_000_000, 1), value)
    _, _, dec = _round_trip(inp, 4, 1)
    assert np.array_equal(dec, inp)


@pytest.mark.slow
def test_diff_encoder_decoder_mix() -> None:
    """Check the order-1 bound on mixed sin/cos, linear and constant columns."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
            idx,
            np.flip(idx),
            np.ones_like(idx),
            np.full_like(idx, 0.01),
        ),
        axis=1,
    )
    err = _assert_theorem_1(inp, 256, 1)
    assert not err[2:].any()  # the ramps and constants come back exactly


@pytest.mark.slow
def test_diff_encoder_decoder_square() -> None:
    """
    Check that order 2 restores the square series exactly with 8 tokens.

    D = Δ²x = 2 and V = 8 give the scale 1. V = 4 = 2^k would leave no room
    for the rounding noise (test_encoder_contracts.py).
    """
    inp = np.square(np.arange(1_000_000, dtype=np.float64))[:, None]
    _, _, dec = _round_trip(inp, 8, 2)
    assert np.array_equal(dec, inp)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5, 6, 7])
def test_mash_bound_holds_at_every_order(order: int) -> None:
    """
    Check Theorem 1 on full-magnitude sin/cos for k up to 7.

    The decoder sums the levels in int64, exactly, so the bound holds at any
    k; the Σ-Δ decoder before version 2 summed floats and lost it from k = 5.
    """
    _assert_theorem_1(_sin_cos(np.arange(5_000, dtype=np.float64)), 4096, order)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
def test_mash_bound_holds_for_long_random_walks(order: int) -> None:
    """Check Theorem 1 on random walks of 10^5 steps."""
    walk = np.cumsum(np.random.default_rng(order).normal(size=(100_000, 1)), axis=0)
    _assert_theorem_1(walk, 256, order)


def test_k2_bound_is_uniform_in_T() -> None:
    """
    Check that the k=2 error stays within Δ/2 as T grows.

    The old 1st-order carry-based encoder drifted by O(T·Δ) at order 2.
    """
    for T in (2_000, 10_000, 50_000):
        _assert_theorem_1(_sin_cos(np.arange(T, dtype=np.float64) * 0.01), 256, 2)


def test_k2_float64_beats_k1_recon_when_signal_is_smooth() -> None:
    """
    Check that k=2 reconstructs a smooth signal better than k=1.

    For a smooth signal (small 2nd derivative relative to 1st), encoding the
    2nd derivative gives tighter bin width → smaller reconstruction error than
    k=1 at the same vocab size.
    """
    inp = np.sin(np.arange(5_000, dtype=np.float64) * 0.001)[:, None]  # very slow
    err_k1 = _assert_theorem_1(inp, 256, 1)[0]
    err_k2 = _assert_theorem_1(inp, 256, 2)[0]
    assert err_k2 < err_k1, (
        f"k=2 err ({err_k2}) should beat k=1 err ({err_k1}) on smooth signal"
    )


def test_per_column_order_of_derivative_round_trip() -> None:
    """
    Check the round trip with a derivative order per column.

    Mixed per-column k: column A with k=0 (raw), column B with k=1, column C
    with k=2. Each column is tokenized with its own derivative order and
    scale, and stays within its own Δ/2 = D_c / (V - 2^k_c), D_c / V at 0.
    """
    idx = np.arange(5_000, dtype=np.float64)
    inp = np.stack(
        (
            # k=0 target: roughly constant series with small drift.
            0.1 * idx,
            # k=1 target: sinusoid.
            np.sin(idx * 0.1),
            # k=2 target: quadratic — 2nd diff is constant.
            0.001 * idx**2,
        ),
        axis=1,
    )
    _assert_theorem_1(inp, 256, np.array([0, 1, 2], dtype=np.int64))


def test_per_column_order_scalar_fallback_equivalence() -> None:
    """
    Check that a uniform per-column order matches the scalar order.

    Passing order as a length-F array of identical values must produce the
    same tokens as passing the scalar.
    """
    V = 256
    idx = np.arange(2_000, dtype=np.float64)
    inp = np.stack((np.sin(idx * 0.1), np.cos(idx * 0.1)), axis=1)
    k_array = np.array([1, 1], dtype=np.int64)
    _, _, enc_s = encode(inp, V, get_domain_of_definition(inp, 1), 1)
    _, _, enc_a = encode(inp, V, get_domain_of_definition(inp, k_array), k_array)
    assert np.array_equal(enc_s, enc_a), (
        "uniform-order array path must match scalar path"
    )
