"""Level-grid tokenization (tokenizer="lgt")."""

import warnings

import numpy as np
import pytest

from diff_gpt.encoder_decoder import (
    TokenClippingWarning,
    decode,
    encode,
    get_domain_of_definition,
)

# A random walk with 0.01 steps plus one +50.0 jump: the max-based domain must
# exceed the floor (it sees the jump), the 0.999-quantile domain must stay
# under the ceiling (it ignores the jump).
JUMP_DOMAIN_FLOOR = 40
QUANTILE_DOMAIN_CEILING = 0.2


def _walk(T: int, F: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(T, F)), axis=0)


def _bound(D: float, V: int, k: int) -> float:
    """Return Δ/2 = 1/(2s) = D / (V - max(2, 2^k)), README Theorem 1."""
    return D / (V - max(2, 2**k))


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_lgt_bound_holds_exactly_in_float64_for_long_series(k: int) -> None:
    """
    Check that the exact integer decode keeps the Δ/2 bound on long series.

    The bound holds at T=100k up to k=5: the decoder sums the levels in
    int64, where a float64 cumulative sum was off by orders of magnitude.
    """
    V = 256
    x = _walk(100_000, 1, seed=k)
    dod = get_domain_of_definition(x, k)
    st, sc, tok = encode(x, V, dod, k, tokenizer="lgt")
    assert tok.min() >= 1
    assert tok.max() <= V - 1
    dec = decode(st, sc, tok, V, k, tokenizer="lgt")
    err = np.abs(dec - x).max()
    assert err <= _bound(dod[0], V, k) * (1 + 1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_lgt_window_encoding_equals_slice_of_full_encoding(k: int) -> None:
    """Check that encoding a window gives the matching slice of the full encoding."""
    V = 256
    x = _walk(5_000, 2, seed=1)
    dod = get_domain_of_definition(x, k)
    _, _, full = encode(x, V, dod, k, tokenizer="lgt")
    for a in range(100, 4_500, 700):
        st, _sc, win = encode(
            x[a : a + 200],
            V,
            dod,
            k,
            tokenizer="lgt",
        )
        assert np.array_equal(win, full[a : a + 200 - k])
        assert np.array_equal(st, x[a : a + k])


def test_lgt_scalar_and_per_column_orders_agree_including_k0() -> None:
    """Check that scalar and per-column orders give equal tokens, k = 0 included."""
    V = 64
    x = _walk(3_000, 3, seed=2)
    for k in (0, 1, 2):
        dod = get_domain_of_definition(x, k)
        _, _, t1 = encode(x, V, dod, k, tokenizer="lgt")
        _, _, t2 = encode(
            x,
            V,
            dod,
            np.array([k, k, k]),
            tokenizer="lgt",
        )
        assert np.array_equal(t1, t2)


def test_lgt_mixed_orders_round_trip_per_column_bound() -> None:
    """Check that mixed per-column orders round-trip within each column's bound."""
    V = 256
    x = _walk(3_000, 4, seed=3)
    order = np.array([0, 1, 2, 3])
    dod = get_domain_of_definition(x, order)
    st, sc, tok = encode(x, V, dod, order, tokenizer="lgt")
    assert st.shape == (3, 4)
    dec = decode(st, sc, tok, V, order, tokenizer="lgt")
    assert dec.shape == x.shape
    for c in range(4):
        err = np.abs(dec[:, c] - x[:, c]).max()
        assert err <= _bound(dod[c], V, int(order[c])) * (1 + 1e-12)


def test_lgt_edge_cases_and_constant_column() -> None:
    """Check the T == k edge case and the exact round trip of a constant column."""
    V = 64
    x = _walk(5, 2, seed=5)
    dod = get_domain_of_definition(_walk(200, 2, seed=6), 2)
    st, sc, tok = encode(x[:2], V, dod, 2, tokenizer="lgt")  # T == k
    assert tok.shape == (0, 2)
    dec = decode(st, sc, tok, V, 2, tokenizer="lgt")
    assert np.abs(dec - x[:2]).max() <= 1 / (2 * sc.min())
    xc = np.stack([np.ones(300), _walk(300, 1, seed=7)[:, 0]], axis=1)
    dod = get_domain_of_definition(xc, 1)
    st, sc, tok = encode(xc, V, dod, 1, tokenizer="lgt")
    assert np.unique(tok[:, 0]).tolist() == [V // 2]
    dec = decode(st, sc, tok, V, 1, tokenizer="lgt")
    assert np.abs(dec[:, 0] - 1.0).max() == 0.0


def test_unknown_tokenizer_rejected() -> None:
    """Check that encode rejects an unknown tokenizer name."""
    x = _walk(10, 1)
    dod = get_domain_of_definition(x, 1)
    with pytest.raises(ValueError, match="unknown tokenizer"):
        encode(x, 64, dod, 1, tokenizer="nope")


def test_lgt_requires_headroom_for_k_ge_2_and_is_bounded_with_it() -> None:
    """
    Check that LGT with k=2 needs V > 2^k and decodes x=t² exactly with it.

    The noise term Δ²ζ spans ±2 levels, so V=4 leaves it no room and must be
    rejected. With V=8 and D = Δ²x = 2 the scale is 1, so the levels are the
    integers t² themselves.
    """
    t = np.arange(20_000, dtype=np.float64)
    x = np.square(t)[:, None]
    dod = get_domain_of_definition(x, 2)
    with pytest.raises(ValueError, match=r"vocab_size > max\(2, 2\^k\)"):
        encode(x, 4, dod, 2, tokenizer="lgt")
    V = 8
    st, sc, tok = encode(x, V, dod, 2, tokenizer="lgt")
    assert tok.min() >= 1
    assert tok.max() <= V - 1
    dec = decode(st, sc, tok, V, 2, tokenizer="lgt")
    assert np.array_equal(dec, x)


def test_lgt_mixed_columns_bound() -> None:
    """Check the LGT error bound on sine, ramp and constant columns."""
    T = 3_000
    idx = np.arange(T, dtype=np.float64)
    x = np.stack(
        [np.sin(idx), idx, np.flip(idx), np.ones(T), np.ones(T) * 0.01],
        axis=1,
    )
    V = 256
    dod = get_domain_of_definition(x, 1)
    st, sc, tok = encode(x, V, dod, 1, tokenizer="lgt")
    assert tok.min() >= 1
    assert tok.max() <= V - 1
    dec = decode(st, sc, tok, V, 1, tokenizer="lgt")
    err = np.abs(dec - x).max(axis=0)
    for c in range(x.shape[1]):
        # Constant columns (D = 0) sit on the fine level grid: error <= 2^-21.
        bound = _bound(dod[c], V, 1) if dod[c] > 0 else 2.0**-21
        assert err[c] <= bound * (1 + 1e-12)


def test_quantile_domain_is_robust_to_one_extreme_step() -> None:
    """One pathological value must not set the grid for everything else."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, size=(5000, 1)), axis=0)
    x[2500, 0] += 50.0  # single huge jump
    d_max = get_domain_of_definition(x, 1)[0]
    d_q = get_domain_of_definition(x, 1, quantile=0.999)[0]
    assert d_max > JUMP_DOMAIN_FLOOR
    assert d_q < QUANTILE_DOMAIN_CEILING
    # quantile=1.0 must reproduce the max exactly
    assert get_domain_of_definition(x, 1, quantile=1.0)[0] == d_max


def test_quantile_domain_per_column() -> None:
    """Check that the quantile domain is taken per column."""
    rng = np.random.default_rng(1)
    x = np.stack([rng.normal(size=3000), rng.normal(size=3000) * 10], axis=1)
    d = get_domain_of_definition(x, np.array([1, 1]), quantile=0.99)
    assert d.shape == (2,)
    assert d[1] > 5 * d[0]


def test_lgt_k1_keeps_the_bound_when_every_level_is_a_rounding_tie() -> None:
    """
    Check the k=1 Δ/2 bound on an integer walk, where levels sit on ties.

    With D = 2 and V = 256 the scale is 63.5, so every odd value lands on
    n + 1/2. Rounding half to even let a step of exactly D reach the level
    step V/2, token V: 261 of 4999 steps clipped and the decode error
    reached 523·Δ/2.
    """
    V = 256
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.integers(-2, 3, size=(5000, 1)), axis=0).astype(np.float64)
    dod = np.array([2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", TokenClippingWarning)
        st, sc, tok = encode(x, V, dod, 1, tokenizer="lgt")
    dec = decode(st, sc, tok, V, 1, tokenizer="lgt")
    err = np.abs(dec - x).max()
    assert err <= _bound(2.0, V, 1) * (1 + 1e-12)
