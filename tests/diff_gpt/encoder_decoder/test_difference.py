"""
Difference operators c(B) = Π (1 - B^S), such as the seasonal 1 - B^24, and the order 0.

README Theorem 1 holds for every such c: the tokens lie in [1, V-1] and the
error within Δ/2 = 1/(2s) = D/(V - ‖c‖₁), where ‖c‖₁ = Σ|c_j| is 2^m for m
factors unless their terms cancel. The order 0 takes the levels floor(s·x)
on a grid shifted by half a step: its tokens take all of [0, V-1] and its
error stays within a hair of D/V.
"""

import warnings
from fractions import Fraction

import numpy as np
import pytest

from diff_gpt.encoder_decoder import (
    Difference,
    column_lags,
    decode,
    differentiate,
    encode,
    get_domain_of_definition,
    integrate,
    prefix_rows,
)

TOKENIZERS = ["lgt", "mash"]
# Hourly data with a daily period.
DAY = 24
# The decode rounds x̂ = (R + F)/s, which may add a few ulps to Δ/2.
DECODE_SLACK = 1 + 1e-12
# On `_hourly` the steps x_t - x_{t-1} reach over twice x_t - x_{t-24}.
SEASONAL_GAIN = 2


def _hourly(T: int, seed: int = 0) -> np.ndarray:
    """Return two hourly series: a daily cycle, a slow trend and noise."""
    t = np.arange(T, dtype=np.float64)
    phase = 2 * np.pi * t / DAY
    cycle = np.stack([10 * np.sin(phase), 5 * np.cos(phase + 1)], axis=1)
    noise = np.random.default_rng(seed).normal(scale=0.1, size=(T, 2))
    return cycle + 0.01 * t[:, None] + noise


def _coefficients(lags: tuple[int, ...]) -> np.ndarray:
    """Return c_0, ..., c_p of Π (1 - B^S), multiplying out the factors."""
    c = np.array([1], dtype=np.int64)
    for lag in lags:
        factor = np.zeros(lag + 1, dtype=np.int64)
        factor[0], factor[lag] = 1, -1
        c = np.convolve(c, factor)
    return c


def _assert_theorem_1(
    x: np.ndarray,
    vocab_size: int,
    order: Difference,
    tokenizer: str,
) -> np.ndarray:
    """
    Check README Theorem 1 for `order` on `x` and return the per-column errors.

    The series is encoded on its own domain, where no token clips: every
    token lies in [1, V-1], the start rows come back as given and every
    column stays within D/(V - ‖c‖₁).
    """
    dod = get_domain_of_definition(x, order)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        start, scale, tokens = encode(x, vocab_size, dod, order, tokenizer=tokenizer)
    assert tokens.min() >= 1
    assert tokens.max() <= vocab_size - 1
    dec = decode(start, scale, tokens, vocab_size, order, tokenizer=tokenizer)
    p = start.shape[0]
    assert p == sum(order.lags)
    assert np.array_equal(dec[:p], x[:p])
    err = np.abs(dec - x).max(axis=0)
    norm = int(np.abs(_coefficients(order.lags)).sum())
    assert (err <= dod / (vocab_size - norm) * DECODE_SLACK).all()
    return err


def test_difference_sorts_its_lags_and_compares_by_them() -> None:
    """Check that the factors commute: the lags are kept sorted."""
    assert Difference(24, 1) == Difference(1, 24)
    assert hash(Difference(24, 1)) == hash(Difference(1, 24))
    assert repr(Difference(24, 1)) == "Difference(1, 24)"
    assert Difference(1, 1) != Difference(2)
    assert Difference(1, 24).lags == (1, 24)


@pytest.mark.parametrize("lags", [(0,), (1, -24), (1.5,)])
def test_difference_takes_integer_lags_of_at_least_1(lags: tuple) -> None:
    """Check that a lag below 1 or not an integer raises ValueError."""
    with pytest.raises(ValueError, match="integer lags >= 1"):
        Difference(*lags)


def test_orders_name_the_lags_of_every_column() -> None:
    """
    Check the forms of `order_of_derivative`.

    The order k is k lags of 1, Difference() the order 0, and a sequence may
    mix ints and operators, one per column.
    """
    assert column_lags(2, 3) == [(1, 1)] * 3
    assert column_lags(Difference(1, 1), 3) == [(1, 1)] * 3
    assert column_lags(Difference(), 2) == column_lags(0, 2) == [(), ()]
    assert column_lags([Difference(24), 1, 0], 3) == [(24,), (1,), ()]
    assert column_lags(np.array([2, 0]), 2) == [(1, 1), ()]
    with pytest.raises(ValueError, match="order_of_derivative shape"):
        column_lags([Difference(24), 1], 3)
    with pytest.raises(ValueError, match="non-negative"):
        column_lags([1, -1], 2)
    assert prefix_rows(0) == prefix_rows(Difference()) == 0
    assert prefix_rows(2) == prefix_rows(Difference(1, 1)) == len((1, 1))
    assert prefix_rows([Difference(1, 24), 2]) == DAY + 1
    assert prefix_rows([Difference(1, 24), 2], 2) == DAY + 1


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize(
    "order",
    [Difference(DAY), Difference(1, DAY), Difference(1, 1, DAY)],
)
def test_seasonal_operators_keep_theorem_1(tokenizer: str, order: Difference) -> None:
    """Check Theorem 1 on hourly series under the seasonal operators of SARIMA."""
    _assert_theorem_1(_hourly(5_000), 256, order, tokenizer)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_the_seasonal_difference_beats_the_first_on_daily_cycles(
    tokenizer: str,
) -> None:
    """
    Check that 1 - B^24 encodes a daily cycle finer than 1 - B at the same V.

    The cycle dominates the steps x_t - x_{t-1}, but cancels in
    x_t - x_{t-24}, so the seasonal domain, and with it the grid step and
    the error, is two to four times smaller.
    """
    x = _hourly(5_000)
    first = _assert_theorem_1(x, 256, Difference(1), tokenizer)
    seasonal = _assert_theorem_1(x, 256, Difference(DAY), tokenizer)
    ratio = get_domain_of_definition(x, 1) / get_domain_of_definition(
        x, Difference(DAY)
    )
    assert (ratio > SEASONAL_GAIN).all()
    assert (seasonal < first).all()


@pytest.mark.parametrize("order", [Difference(1, DAY), Difference(2, 3)])
def test_lgt_window_encoding_equals_slice_of_full_encoding(order: Difference) -> None:
    """Check that an LGT token depends on the p + 1 values it differences alone."""
    x = _hourly(3_000, seed=1)
    p = sum(order.lags)
    dod = get_domain_of_definition(x, order)
    _, _, full = encode(x, 256, dod, order, tokenizer="lgt")
    for a in range(100, 2_700, 400):
        start, _, window = encode(x[a : a + 200], 256, dod, order, tokenizer="lgt")
        assert np.array_equal(window, full[a : a + 200 - p])
        assert np.array_equal(start, x[a : a + p])


# (1 - B)(1 - B^2)(1 - B^3) = 1 - B - B^2 + B^4 + B^5 - B^6: the B^3 terms
# cancel, so ‖c‖₁ = 6, below 2^3.
CANCELLING = Difference(1, 2, 3)
CANCELLING_NORM = 6


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_cancelling_factors_need_only_their_norm(tokenizer: str) -> None:
    """
    Check that the scale leaves ‖c‖₁ levels for the noise, not 2^m.

    V = 8 = 2^3 leaves room for the noise of (1 - B)(1 - B^2)(1 - B^3),
    whose |c(B)ζ| stays below ‖c‖₁/2 = 3: the tokens fill [1, 7] and the
    error stays within D/2. V = 6 = ‖c‖₁ leaves none.
    """
    assert int(np.abs(_coefficients(CANCELLING.lags)).sum()) == CANCELLING_NORM
    x = np.cumsum(np.random.default_rng(2).normal(size=(5_000, 1)), axis=0)
    dod = get_domain_of_definition(x, CANCELLING)
    with pytest.raises(ValueError, match="vocab_size > ‖c‖₁ = 6"):
        encode(x, CANCELLING_NORM, dod, CANCELLING, tokenizer=tokenizer)
    _assert_theorem_1(x, CANCELLING_NORM + 2, CANCELLING, tokenizer)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_seasonal_decode_raises_where_the_tokens_leave_int64(tokenizer: str) -> None:
    """
    Check that the strided sums of the decoder catch an int64 overflow.

    Under (1 - B)^2 (1 - B^24) a run of the largest step grows the levels
    like 2^19·n^3/144, past 2^63 within 2·10^5 rows at V = 2^20.
    """
    vocab_size = 2**20
    order = Difference(1, 1, DAY)
    x = _hourly(100)
    start, scale, _ = encode(
        x,
        vocab_size,
        get_domain_of_definition(x, order),
        order,
        tokenizer=tokenizer,
    )
    tokens = np.full((200_000, 2), vocab_size - 1, dtype=np.int64)
    with pytest.raises(ValueError, match="outside int64"):
        decode(start, scale, tokens, vocab_size, order, tokenizer=tokenizer)
    short = decode(start, scale, tokens[:1_000], vocab_size, order, tokenizer=tokenizer)
    assert np.isfinite(short).all()


def test_differentiate_and_integrate_take_operators() -> None:
    """Check the seasonal differences and their exact inverse, per column."""
    t = np.arange(500, dtype=np.float64)
    x = np.stack([t**2, (t % DAY) * 7.0 + t, t**3], axis=1)
    order = [Difference(DAY), Difference(1, DAY), 2]
    start, diff = differentiate(x, order)
    assert start.shape == (DAY + 1, 3)
    np.testing.assert_array_equal(diff[:, 0], (x[DAY:, 0] - x[:-DAY, 0])[1:])
    # The trend t steps by 1 every hour, the cycle repeats every day.
    np.testing.assert_array_equal(diff[:, 1], np.zeros(500 - DAY - 1))
    np.testing.assert_array_equal(diff[:, 2], np.diff(x[:, 2], n=2)[DAY - 1 :])
    np.testing.assert_array_equal(integrate(start, diff, order), x)


def _exact_max(x: np.ndarray, lags: tuple[int, ...]) -> Fraction:
    weights = _coefficients(lags)[::-1]
    p = len(weights) - 1
    return max(
        abs(sum(int(w) * Fraction(x[t + j]) for j, w in enumerate(weights)))
        for t in range(len(x) - p)
    )


@pytest.mark.parametrize("order", [Difference(2), Difference(1, 3), CANCELLING])
def test_the_domain_max_is_exact_for_operators(order: Difference) -> None:
    """Check the max domain against the exact c(B) x of random walks."""
    rng = np.random.default_rng(3)
    for _ in range(30):
        x = np.cumsum(rng.normal(size=(40, 1)), axis=0) / 10
        exact = _exact_max(x[:, 0], order.lags)
        dod = get_domain_of_definition(x, order)[0]
        assert Fraction(dod) >= exact
        assert Fraction(np.nextafter(dod, 0.0)) < exact


# A domain whose scale V/(2D) = 32 is a float: it must step below it.
ORDER_0_V = 64
ORDER_0_DOMAIN = 1.0


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_order_0_uses_every_token_and_stays_within_d_over_v(tokenizer: str) -> None:
    """
    Check the order 0: tokens floor(s·x) + V/2 in [0, V-1], error ≈ D/V.

    The scale stays below V/(2D), so x = D lands in the top cell and x = -D
    in the bottom one, and the middles of the cells lie within
    1/(2s) = D/V·(1 + 2^-52) of the values. The rounding of version 3 used
    V - 1 tokens and reached D/(V - 2).
    """
    x = np.concatenate(
        [
            [[-ORDER_0_DOMAIN], [ORDER_0_DOMAIN], [0.0]],
            np.random.default_rng(4).uniform(-1, 1, size=(2_000, 1)),
        ]
    )
    dod = get_domain_of_definition(x, 0)
    assert dod[0] == ORDER_0_DOMAIN
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        start, scale, tokens = encode(x, ORDER_0_V, dod, 0, tokenizer=tokenizer)
    assert Fraction(scale[0]) * Fraction(dod[0]) < Fraction(ORDER_0_V, 2)
    assert tokens[:3, 0].tolist() == [0, ORDER_0_V - 1, ORDER_0_V // 2]
    dec = decode(start, scale, tokens, ORDER_0_V, 0, tokenizer=tokenizer)
    half_step = 1 / (2 * scale[0])
    assert half_step <= ORDER_0_DOMAIN / ORDER_0_V * (1 + 2**-50)
    assert np.abs(dec - x).max() <= half_step * DECODE_SLACK


def test_order_0_is_the_same_for_both_tokenizers() -> None:
    """Check that both tokenizers take the anchor 1/2 at the order 0."""
    x = _hourly(1_000)
    dod = get_domain_of_definition(x, 0)
    lgt = encode(x, 256, dod, 0, tokenizer="lgt")
    mash = encode(x, 256, dod, 0, tokenizer="mash")
    assert np.array_equal(lgt[2], mash[2])
    np.testing.assert_array_equal(
        decode(*lgt, 256, 0, tokenizer="lgt"),
        decode(*mash, 256, 0, tokenizer="mash"),
    )
