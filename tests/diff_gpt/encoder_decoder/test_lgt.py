"""Level-grid tokenization (tokenizer="lgt")."""
import numpy as np
import pytest

from diff_gpt.encoder_decoder import (
    decode,
    encode,
    get_domain_of_definition,
    np_to_decimal,
)


def _walk(T, F, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(T, F)), axis=0)


def _bound(D, V, k):
    mash = k >= 2 and V > 2 ** k + 2
    return D / (V - 2 - (2 ** k if mash else 0))


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_lgt_bound_holds_exactly_in_float64_for_long_series(k):
    """Exact integer decode: the Δ/2 bound holds at T=100k even for k=5, where
    the Σ-Δ float64 decoder is off by orders of magnitude."""
    V = 256
    x = _walk(100_000, 1, seed=k)
    dod = get_domain_of_definition(x, k, False)
    st, sc, tok = encode(x, V, dod, k, False, tokenizer="lgt")
    assert tok.min() >= 1 and tok.max() <= V - 1
    dec = decode(st, sc, tok, V, k, False, tokenizer="lgt")
    err = np.abs(dec - x).max()
    assert err <= _bound(dod[0], V, k) * (1 + 1e-12)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_lgt_window_encoding_equals_slice_of_full_encoding(k):
    V = 256
    x = _walk(5_000, 2, seed=1)
    dod = get_domain_of_definition(x, k, False)
    _, _, full = encode(x, V, dod, k, False, tokenizer="lgt")
    for a in range(100, 4_500, 700):
        st, sc, win = encode(x[a : a + 200], V, dod, k, False, tokenizer="lgt")
        assert np.array_equal(win, full[a : a + 200 - k])
        assert np.array_equal(st, x[a : a + k])


def test_lgt_scalar_and_per_column_orders_agree_including_k0():
    V = 64
    x = _walk(3_000, 3, seed=2)
    for k in (0, 1, 2):
        dod = get_domain_of_definition(x, k, False)
        _, _, t1 = encode(x, V, dod, k, False, tokenizer="lgt")
        _, _, t2 = encode(x, V, dod, np.array([k, k, k]), False, tokenizer="lgt")
        assert np.array_equal(t1, t2)


def test_lgt_mixed_orders_round_trip_per_column_bound():
    V = 256
    x = _walk(3_000, 4, seed=3)
    order = np.array([0, 1, 2, 3])
    dod = get_domain_of_definition(x, order, False)
    st, sc, tok = encode(x, V, dod, order, False, tokenizer="lgt")
    assert st.shape == (3, 4)
    dec = decode(st, sc, tok, V, order, False, tokenizer="lgt")
    assert dec.shape == x.shape
    for c in range(4):
        err = np.abs(dec[:, c] - x[:, c]).max()
        assert err <= _bound(dod[c], V, int(order[c])) * (1 + 1e-12)


def test_lgt_decimal_matches_float64_tokens_and_bound():
    V = 256
    x = _walk(800, 2, seed=4)
    for k in (1, 2, 3):
        dod = get_domain_of_definition(x, k, False)
        _, _, tf = encode(x, V, dod, k, False, tokenizer="lgt")
        xd, dd = np_to_decimal(x), np_to_decimal(dod)
        st, sc, td = encode(xd, V, dd, k, True, tokenizer="lgt")
        assert np.array_equal(tf, td)
        dec = decode(st, sc, td, V, k, True, tokenizer="lgt")
        err = float(np.max(np.abs(dec - xd)))
        assert err <= _bound(dod[0], V, k) * (1 + 1e-12)


def test_lgt_edge_cases_and_constant_column():
    V = 64
    x = _walk(5, 2, seed=5)
    dod = get_domain_of_definition(_walk(200, 2, seed=6), 2, False)
    st, sc, tok = encode(x[:2], V, dod, 2, False, tokenizer="lgt")  # T == k
    assert tok.shape == (0, 2)
    dec = decode(st, sc, tok, V, 2, False, tokenizer="lgt")
    assert np.abs(dec - x[:2]).max() <= 1 / (2 * sc.min())
    xc = np.stack([np.ones(300), _walk(300, 1, seed=7)[:, 0]], axis=1)
    dod = get_domain_of_definition(xc, 1, False)
    st, sc, tok = encode(xc, V, dod, 1, False, tokenizer="lgt")
    assert np.unique(tok[:, 0]).tolist() == [V // 2]
    dec = decode(st, sc, tok, V, 1, False, tokenizer="lgt")
    assert np.abs(dec[:, 0] - 1.0).max() == 0.0


def test_unknown_tokenizer_rejected():
    x = _walk(10, 1)
    dod = get_domain_of_definition(x, 1, False)
    with pytest.raises(ValueError):
        encode(x, 64, dod, 1, False, tokenizer="nope")


def test_lgt_requires_headroom_for_k_ge_2_and_is_bounded_with_it():
    """k=2 needs V > 2^k + 2 under LGT: the noise term Δ²ζ spans ±2 bins and
    there is no carry-scheme fallback. V=4 must be rejected; V=8 on x=t²
    (Δ²x ≡ 2) stays within Δ/2 = D/(V-2-4). (The Σ-Δ encoder is exact on this
    signal because it quantises the derivative, which lies on its grid; the
    two anchors differ, see Theorem A in the notes.)"""
    t = np.arange(20_000, dtype=np.float64)
    x = np.square(t)[:, None]
    dod = get_domain_of_definition(x, 2, False)
    with pytest.raises(AssertionError, match="vocab_size > 2\\^k \\+ 2"):
        encode(x, 4, dod, 2, False, tokenizer="lgt")
    V = 8
    st, sc, tok = encode(x, V, dod, 2, False, tokenizer="lgt")
    assert tok.min() >= 0 and tok.max() <= V - 1
    dec = decode(st, sc, tok, V, 2, False, tokenizer="lgt")
    assert np.abs(dec - x).max() <= dod[0] / (V - 2 - 4) + 1e-9


def test_lgt_mixed_columns_bound_in_decimal():
    T = 3_000
    idx = np.arange(T, dtype=np.float64)
    x = np.stack([np.sin(idx), idx, np.flip(idx), np.ones(T), np.ones(T) * 0.01], axis=1)
    V = 256
    xd = np_to_decimal(x)
    dod = get_domain_of_definition(xd, 1, True)
    st, sc, tok = encode(xd, V, dod, 1, True, tokenizer="lgt")
    assert tok.min() >= 0 and tok.max() <= V - 1
    dec = decode(st, sc, tok, V, 1, True, tokenizer="lgt")
    err = np.abs(np.asarray(dec - xd, dtype=np.float64)).max(axis=0)
    D = np.asarray(dod, dtype=np.float64)
    for c in range(x.shape[1]):
        # Constant columns (D = 0) sit on the fine level grid: error <= 2^-21.
        bound = D[c] / (V - 2) if D[c] > 0 else 2.0 ** -21
        assert err[c] <= bound + 1e-12



def test_quantile_domain_is_robust_to_one_extreme_step():
    """One pathological value must not set the grid for everything else."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, size=(5000, 1)), axis=0)
    x[2500, 0] += 50.0  # single huge jump
    d_max = get_domain_of_definition(x, 1, False)[0]
    d_q = get_domain_of_definition(x, 1, False, quantile=0.999)[0]
    assert d_max > 40
    assert d_q < 0.2
    # quantile=1.0 must reproduce the max exactly
    assert get_domain_of_definition(x, 1, False, quantile=1.0)[0] == d_max


def test_quantile_domain_per_column_and_decimal_guard():
    rng = np.random.default_rng(1)
    x = np.stack([rng.normal(size=3000), rng.normal(size=3000) * 10], axis=1)
    d = get_domain_of_definition(x, np.array([1, 1]), False, quantile=0.99)
    assert d.shape == (2,)
    assert d[1] > 5 * d[0]
    with pytest.raises(AssertionError, match="float64-only"):
        get_domain_of_definition(np_to_decimal(x), 1, True, quantile=0.99)
