"""
Exact levels: steps within the domain never clip, at rounding ties included.

README Theorem 1 holds in float64 because nothing between x and the tokens
is rounded: `_levels` reads the levels off fl(s·x) and its exact error, the
scale rounds down so that s·D <= (V - ‖c‖₁)/2, and the max of
`get_domain_of_definition` rounds up so that |c(B) x| <= D. Version 2 got
each of these wrong at some tie, and its token V clipped a step within D.
"""

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from diff_gpt.encoder_decoder import (
    TokenClippingWarning,
    _levels,
    decode,
    encode,
    get_domain_of_definition,
)


def _exact_levels(x: np.ndarray, scale: float, fraction: np.ndarray) -> np.ndarray:
    """Return floor(s·x - F + 1/2) in rationals."""
    half = Fraction(1, 2)
    return np.array(
        [
            math.floor(Fraction(scale) * Fraction(v) - Fraction(f) + half)
            for v, f in zip(x, fraction, strict=True)
        ],
        dtype=np.int64,
    )


def _nudge(x: np.ndarray, ulps: np.ndarray) -> np.ndarray:
    """Move each x[i] by ulps[i] floats up or down."""
    x = x.copy()
    for step in range(1, int(np.abs(ulps).max()) + 1):
        moving = np.abs(ulps) >= step
        x[moving] = np.nextafter(x[moving], np.sign(ulps[moving]) * np.inf)
    return x


# The products s·x tried below: around the ties of the level grid, where |p|
# crosses 1/2, below 2^-53 and near 2^52, where float64's step reaches 1.
_TIE_RANGE = 1000
_HALF_ULPS = 3
# Version 2 rounded the fractional part of fl(s·x) - F up from here.
_ROUND_UP_FROM = 0.5


@pytest.mark.parametrize("scale", [63.5 / 1.8691333659165825, 2.0**20, 3.0**33 / 2**52])
def test_levels_are_exact_around_rounding_ties(scale: float) -> None:
    """
    Check `_levels` against rationals where rounding s·x first would go wrong.

    The fractions lie on the finest grid of the "mash" anchor, 2^-53, so
    that F - 1/2 has a bit as low as the products'. Rounding s·x first, as
    version 2 did, gets several percent of these levels wrong.
    """
    rng = np.random.default_rng(0)
    n = 4000
    fraction = np.ldexp(rng.integers(0, 2**53, n).astype(np.float64), -53)
    fraction[: n // 4] = 0.5  # F - 1/2 = 0: the level follows the sign of s·x
    ties = rng.integers(-_TIE_RANGE, _TIE_RANGE, n) + fraction - 0.5
    targets = np.concatenate(
        [
            ties[: n // 2],
            rng.choice([-0.5, 0.5], n // 4),
            rng.choice([-1.0, 1.0], n // 8) * 2.0**52,
            rng.normal(size=n // 8) * 2.0**-60,
        ]
    )
    x = _nudge(targets / scale, rng.integers(-_HALF_ULPS, _HALF_ULPS + 1, n))
    want = _exact_levels(x, scale, fraction)
    np.testing.assert_array_equal(_levels(x, scale, fraction, factors=0), want)
    u = x * scale - fraction
    rounded_first = (np.floor(u) + (u - np.floor(u) >= _ROUND_UP_FROM)).astype(np.int64)
    assert np.count_nonzero(rounded_first != want) > n // 50


def _error_in_half_steps(
    x: np.ndarray,
    vocab_size: int,
    domain: np.ndarray,
    order: int,
    tokenizer: str,
) -> float:
    """Encode `x` on `domain` with warnings as errors; return max |x̂ - x| / (Δ/2)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        start, scale, tokens = encode(x, vocab_size, domain, order, tokenizer=tokenizer)
    assert tokens.min() >= 1
    assert tokens.max() <= vocab_size - 1
    dec = decode(start, scale, tokens, vocab_size, order, tokenizer=tokenizer)
    return float(np.abs(dec - x).max() * 2 * scale[0])


# The k = 1 examples use V = 64, where s·D may reach V/2 - 1 = 31.
V_K1 = 64
ROOM_K1 = Fraction(V_K1 - 2, 2)
# The decode rounds x̂ = (R + F)/s, which may add a few ulps to Δ/2.
DECODE_SLACK = 1 + 1e-9

# A k = 1 step within D whose ends round onto ties: s·x1 lies just below 31.5
# and rounds to it, s·x0 lies just below 1/2.
LEVEL_TIE_DOMAIN = 1.8691333659165825
LEVEL_TIE_SERIES = [0.030147312353493265, 1.8992806782700757]


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_a_step_within_the_domain_at_ties_does_not_clip(tokenizer: str) -> None:
    """
    Check a k = 1 step within D whose ends sit just below ties.

    s·D <= 31 here, but fl(s·x1) rounds up onto the tie 31.5: rounding s·x
    first, as version 2 did, made the level step 32 and the token V = 64.
    """
    x = np.array(LEVEL_TIE_SERIES)[:, None]
    assert Fraction(x[1, 0]) - Fraction(x[0, 0]) <= Fraction(LEVEL_TIE_DOMAIN)
    domain = np.array([LEVEL_TIE_DOMAIN])
    assert _error_in_half_steps(x, V_K1, domain, 1, tokenizer) <= DECODE_SLACK


# A k = 1 step of exactly D = fl(x1 - x0): the scale 31/D rounds up, past it.
SCALE_TIE_SERIES = [0.06800385238021311, 1.4734168015712845]


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_the_scale_rounds_down(tokenizer: str) -> None:
    """
    Check that s·D <= V/2 - 1 holds exactly, so a step of exactly D fits.

    The nearest float to 31/D gives s·D = 31 + 1.1·10^-15, and the step
    x1 - x0 = D then reaches the level step 32, token V = 64, even with
    exact levels. Version 2 clipped it, and decoded the rows after it up to
    2.65·Δ/2 off.
    """
    x = np.array(SCALE_TIE_SERIES)[:, None]
    dod = get_domain_of_definition(x, 1)
    nearest = (V_K1 - 2) / dod[0] / 2
    assert Fraction(nearest) * Fraction(dod[0]) > ROOM_K1
    _, scale, _ = encode(x, V_K1, dod, 1, tokenizer=tokenizer)
    assert Fraction(scale[0]) * Fraction(dod[0]) <= ROOM_K1
    assert _error_in_half_steps(x, V_K1, dod, 1, tokenizer) <= DECODE_SLACK


def test_the_domain_max_rounds_up() -> None:
    """
    Check that the max of `get_domain_of_definition` is never below the steps.

    The step from 1/2 - 2^-54 to 3/2 is 1 + 2^-54, which np.diff rounds to
    1: on that domain, with s = 1 at V = 4, "lgt" steps from level 0 to
    level 2 and clips. The domain rounded up holds the step, for both
    tokenizers.
    """
    x = np.array([np.nextafter(0.5, 0.0), 1.5])[:, None]
    assert np.diff(x[:, 0])[0] == 1.0
    with pytest.warns(TokenClippingWarning, match="1 of 1 tokens"):
        encode(x, 4, np.array([1.0]), 1, tokenizer="lgt")
    dod = get_domain_of_definition(x, 1)
    assert dod[0] == np.nextafter(1.0, 2.0)
    for tokenizer in ("lgt", "mash"):
        assert _error_in_half_steps(x, 4, dod, 1, tokenizer) <= DECODE_SLACK


# A k = 2 series with a second difference of nearly D at ties: version 2's
# scale put fl(s·x0) on the tie 63.5 and kept fl(s·x1) below 1/2.
DRIFT_DOMAIN = 0.5614602859042921
DRIFT_X0, DRIFT_X1 = 0.28295815995970275, 0.002228017007556714


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_k2_tie_clip_no_longer_drifts(tokenizer: str) -> None:
    """
    Check the k = 2 series whose clipped token drifted without bound.

    Version 2 clipped the first token, V = 256, and at k = 2 the lost level
    grows with every row: the decode was 17·Δ/2 off at row 10 and 1999·Δ/2
    at row 1001. All its steps lie within D.
    """
    x = np.array([DRIFT_X0, DRIFT_X1] + [DRIFT_X0] * 1000)[:, None]
    step = Fraction(DRIFT_X0) - 2 * Fraction(DRIFT_X1) + Fraction(DRIFT_X0)
    assert abs(step) <= Fraction(DRIFT_DOMAIN)
    domain = np.array([DRIFT_DOMAIN])
    assert _error_in_half_steps(x, 256, domain, 2, tokenizer) <= DECODE_SLACK


def _exact_max_abs_difference(x: np.ndarray, k: int) -> Fraction:
    weights = [(-1) ** (k - j) * math.comb(k, j) for j in range(k + 1)]
    return max(
        abs(sum(w * Fraction(x[t + j]) for j, w in enumerate(weights)))
        for t in range(len(x) - k)
    )


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_the_domain_max_is_the_least_float_at_or_above_the_exact_max(k: int) -> None:
    """
    Check the max domain against the exact k-th differences of random walks.

    np.diff falls short of the exact max on about one walk in ten at k = 1
    and one in four at k = 3.
    """
    rng = np.random.default_rng(k)
    short = 0
    for _ in range(50):
        x = np.cumsum(rng.normal(size=(40, 1)), axis=0) / 10
        exact = _exact_max_abs_difference(x[:, 0], k)
        dod = get_domain_of_definition(x, k)[0]
        assert Fraction(dod) >= exact
        assert Fraction(np.nextafter(dod, 0.0)) < exact
        short += Fraction(np.abs(np.diff(x[:, 0], n=k)).max()) < exact
    assert (short > 0) == (k > 0)
