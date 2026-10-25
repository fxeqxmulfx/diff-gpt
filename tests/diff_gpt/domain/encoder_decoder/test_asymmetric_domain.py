"""
Ranges [lo, hi] for the steps of a column: the scale, the zero point, the fit.

A range, shape (F, 2), need not hold 0 in its middle: the scale puts
hi - lo on the V - ‖c‖₁ levels instead of 2·max(|lo|, |hi|), and the zero
point z moves the range to the middle of the vocabulary.
"""

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from diff_gpt.domain.encoder_decoder import (
    CONSTANT_COLUMN_SCALE,
    Difference,
    decode,
    encode,
    get_domain_of_definition,
    level_grid,
)

V = 64
# Each order with ‖c‖₁, the levels its scale leaves for the rounding noise:
# none at the order 0, and 4 for (1 - B)(1 - B^3) = 1 - B - B^3 + B^4.
ORDERS = [(0, 0), (1, 2), (2, 4), (Difference(1, 3), 4)]
VOCAB_SIZES = (8, 64, 4096)
EPS = float(np.finfo(np.float64).eps)
# The decode rounds x̂ = (R + F)/s, which may add a few ulps to Δ/2.
DECODE_SLACK = 1 + 1e-9


def _symmetric_scale(vocab_size: int, bound: float, headroom: int) -> float:
    """
    Return the scale of [-D, D] that README Theorem 1 gives.

    The largest float64 s with s·D <= (V - ‖c‖₁)/2, and s·D < V/2 at the
    order 0; `CONSTANT_COLUMN_SCALE` for D <= eps.
    """
    if bound <= EPS:
        return CONSTANT_COLUMN_SCALE
    room = Fraction(vocab_size - headroom, 2)

    def fits(s: float) -> bool:
        product = Fraction(s) * Fraction(bound)
        return product <= room if headroom else product < room

    s = math.nextafter(float(room / Fraction(bound)), math.inf)
    while not fits(s):
        s = math.nextafter(s, 0.0)
    return s


def _level_steps(
    s: float,
    lo: Fraction,
    hi: Fraction,
    headroom: int,
    *,
    shift: Fraction = Fraction(0),
) -> list[int]:
    """
    Return the least and the greatest c(B) L of the steps in [lo, hi] on the scale s.

    Above the order 0, c(B) L lies within ‖c‖₁/2 of s·c(B) x - `shift`,
    strictly, where the shift c(B) A is 0, or 1/2 for "mash-half"; at the
    order 0 the levels are floor(s·x).
    """
    scale = Fraction(s)
    if headroom == 0:
        return [math.floor(scale * lo), math.floor(scale * hi)]
    return [
        math.floor(scale * lo - shift) + 1 - headroom // 2,
        math.ceil(scale * hi - shift) - 1 + headroom // 2,
    ]


def _ranges(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return 5n ranges [lo, hi]: across 0, on either side of it, narrow far from it."""
    size = 10.0 ** rng.uniform(-3.0, 3.0, (n, 1))
    across = np.sort(rng.normal(size=(n, 2)), axis=1) * size
    low = rng.lognormal(0.0, 2.0, (n, 1))
    above = np.hstack([low, low + rng.lognormal(0.0, 2.0, (n, 1))]) * size
    centre = 10.0 ** rng.uniform(0.0, 6.0, (n, 1))
    narrow = np.hstack([centre, centre * (1 + 10.0 ** rng.uniform(-8.0, 0.0, (n, 1)))])
    return np.vstack([across, above, -above[:, ::-1], narrow, -narrow[:, ::-1]])


def _exact_steps(column: np.ndarray, order: int | Difference) -> list[Fraction]:
    """Return the steps c(B) x of `column` in rationals, without rounding."""
    lags = order.lags if isinstance(order, Difference) else (1,) * order
    steps = [Fraction(value) for value in column]
    for lag in lags:
        steps = [a - b for a, b in zip(steps[lag:], steps[:-lag], strict=True)]
    return steps


@pytest.mark.parametrize(("order", "headroom"), ORDERS)
def test_a_symmetric_domain_keeps_its_scale_in_either_form(
    order: int | Difference,
    headroom: int,
) -> None:
    """
    Check that D and the range [-D, D] both give the scale of Theorem 1, z = V/2.

    Symmetric domains keep the scale they had before ranges, so their tokens
    stay those of ENCODER_VERSION 4.
    """
    rng = np.random.default_rng(0)
    bounds = np.concatenate(
        [rng.lognormal(0.0, 8.0, 300), [0.0, EPS, 2 * EPS, 1.0, 1e300]],
    )
    for vocab_size in VOCAB_SIZES:
        expected = [_symmetric_scale(vocab_size, d, headroom) for d in bounds]
        for domain in (bounds, np.stack([-bounds, bounds], axis=1)):
            grid = level_grid(vocab_size, domain, order)
            np.testing.assert_array_equal(grid.scale, expected)
            np.testing.assert_array_equal(grid.zero, vocab_size // 2)


@pytest.mark.parametrize(("order", "headroom"), ORDERS)
def test_a_range_fits_its_steps_at_most_a_level_short_of_the_best(
    order: int | Difference,
    headroom: int,
) -> None:
    """
    Check the grid of ranges across 0, on one side of it and narrow far from it.

    Every step in [lo, hi] gets a token in range, the scale falls short of
    (V - ‖c‖₁ - 1)/(hi - lo) by less than an ulp, and across 0, where the
    scales that fit run up from 0, no larger float64 fits.
    """
    ranges = _ranges(np.random.default_rng(1), 100)
    first = 0 if headroom == 0 else 1
    for vocab_size in VOCAB_SIZES:
        grid = level_grid(vocab_size, ranges, order)
        room = vocab_size - headroom
        for (lo, hi), s, z in zip(ranges, grid.scale, grid.zero, strict=True):
            low, high = Fraction(lo), Fraction(hi)
            least, most = _level_steps(s, low, high, headroom)
            assert first <= least + int(z)
            assert most + int(z) <= vocab_size - 1
            up = math.nextafter(s, math.inf)
            assert Fraction(up) * (high - low) > room - 1
            if lo < 0 < hi:
                least, most = _level_steps(up, low, high, headroom)
                assert most - least + 1 > vocab_size - first


@pytest.mark.parametrize(("order", "headroom"), ORDERS[1:])
def test_mash_half_fits_a_range_on_all_the_tokens(
    order: int | Difference,
    headroom: int,
) -> None:
    """
    Check the "mash-half" grid of the ranges above, whose steps c(B) L sit 1/2 lower.

    Every step in [lo, hi] gets a token in [0, V-1], the scale is at least
    that of "mash" and falls short of (V - ‖c‖₁)/(hi - lo) by less than an
    ulp, and across 0 no larger float64 fits.
    """
    ranges = _ranges(np.random.default_rng(1), 100)
    half = Fraction(1, 2)
    for vocab_size in VOCAB_SIZES:
        grid = level_grid(vocab_size, ranges, order, tokenizer="mash-half")
        mash = level_grid(vocab_size, ranges, order, tokenizer="mash")
        assert (grid.scale >= mash.scale).all()
        for (lo, hi), s, z in zip(ranges, grid.scale, grid.zero, strict=True):
            low, high = Fraction(lo), Fraction(hi)
            least, most = _level_steps(s, low, high, headroom, shift=half)
            assert least + int(z) >= 0
            assert most + int(z) <= vocab_size - 1
            up = math.nextafter(s, math.inf)
            assert Fraction(up) * (high - low) > vocab_size - headroom
            if lo < 0 < hi:
                least, most = _level_steps(up, low, high, headroom, shift=half)
                assert most - least + 1 > vocab_size


@pytest.mark.parametrize(
    ("domain", "order", "scale", "zero"),
    [
        # [1, 3] takes s = 62/2, three times the 62/6 of [-3, 3].
        ([1.0, 3.0], 1, 31.0, -30),
        ([0.0, 1.0], 1, 62.0, 1),
        ([-1.0, 0.0], 1, 62.0, 63),
        # At the order 0, x = 1 must floor below V: s < 64.
        ([0.0, 1.0], 0, math.nextafter(64.0, 0.0), 0),
        # x = 0 takes a level of its own above the 63 of [-1, 0).
        ([-1.0, 0.0], 0, 63.0, 63),
    ],
)
def test_one_sided_ranges_take_the_whole_vocabulary(
    domain: list[float],
    order: int,
    scale: float,
    zero: int,
) -> None:
    """Check the scale and the zero point of one-sided ranges at V = 64."""
    grid = level_grid(V, np.array([domain]), order)
    assert grid.scale[0] == scale
    assert grid.zero[0] == zero


def test_counts_take_every_token_at_half_the_step_of_their_symmetric_hull() -> None:
    """
    Check counts 0..100 at the order 0 on their range [0, 100] and on D = 100.

    The domain [-100, 100] leaves the 32 tokens of negative counts unused.
    """
    x = np.arange(101, dtype=np.float64)[:, None]
    domain = get_domain_of_definition(x, 0, symmetric=False)
    np.testing.assert_array_equal(domain, [[0.0, 100.0]])
    _, grid, tokens = encode(x, V, domain, 0)
    _, hull, hull_tokens = encode(x, V, np.array([100.0]), 0)
    assert np.unique(tokens).size == V
    assert np.unique(hull_tokens).size == V // 2
    assert grid.scale[0] == 2 * hull.scale[0]


@pytest.mark.parametrize("tokenizer", ["lgt", "mash", "mash-half"])
@pytest.mark.parametrize(("order", "headroom"), ORDERS)
def test_series_on_their_exact_ranges_decode_within_half_a_level(
    order: int | Difference,
    headroom: int,
    tokenizer: str,
) -> None:
    """
    Check encode and decode of drifting series on their exact ranges.

    No step clips, x̂ stays within Δ/2 = 1/(2s) of x, and Δ/2 within
    (hi - lo)/(2(V - ‖c‖₁ - 1)).
    """
    rng = np.random.default_rng(2)
    walk = np.cumsum(rng.exponential(size=(300, 3)), axis=0)
    x = walk * np.array([1.0, -1e-3, 1e3])
    domain = get_domain_of_definition(x, order, symmetric=False)
    width = domain[:, 1] - domain[:, 0]
    for vocab_size in VOCAB_SIZES:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            start, grid, tokens = encode(
                x,
                vocab_size,
                domain,
                order,
                tokenizer=tokenizer,
            )
        dec = decode(start, grid, tokens, vocab_size, order, tokenizer=tokenizer)
        half_step = 1 / (2 * grid.scale)
        assert (np.abs(dec - x).max(axis=0) <= half_step * DECODE_SLACK).all()
        bound = width / (2 * (vocab_size - headroom - 1))
        assert (half_step <= bound * DECODE_SLACK).all()


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_a_constant_step_takes_the_middle_token(tokenizer: str) -> None:
    """
    Check a line, whose range is the point [0.5, 0.5], at the order 1.

    A point takes the grid of [-0.5, 0.5], and z puts its step on V/2, so
    the line decodes exactly.
    """
    x = 0.5 * np.arange(50, dtype=np.float64)[:, None]
    domain = get_domain_of_definition(x, 1, symmetric=False)
    np.testing.assert_array_equal(domain, [[0.5, 0.5]])
    start, grid, tokens = encode(x, V, domain, 1, tokenizer=tokenizer)
    assert grid.scale[0] == level_grid(V, np.array([0.5]), 1).scale[0]
    np.testing.assert_array_equal(tokens, V // 2)
    dec = decode(start, grid, tokens, V, 1, tokenizer=tokenizer)
    np.testing.assert_array_equal(dec, x)


@pytest.mark.parametrize(("order", "headroom"), ORDERS)
def test_a_point_at_zero_takes_the_constant_column_scale(
    order: int | Difference,
    headroom: int,
) -> None:
    """Check that ranges within eps of 0 take `CONSTANT_COLUMN_SCALE` and z = V/2."""
    del headroom
    domain = np.array([[0.0, 0.0], [-EPS / 2, EPS / 2], [EPS, EPS]])
    grid = level_grid(V, domain, order)
    np.testing.assert_array_equal(grid.scale, CONSTANT_COLUMN_SCALE)
    np.testing.assert_array_equal(grid.zero, V // 2)


@pytest.mark.parametrize(
    ("domain", "error", "match"),
    [
        (np.array([[1.0, -1.0]]), ValueError, r"needs D >= 0, or lo <= hi"),
        (np.array([-1.0]), ValueError, r"needs D >= 0, or lo <= hi"),
        (np.array([[0.0, 1.0, 2.0]]), ValueError, r"has shape \(1, 3\) for 1 col"),
        (np.array([1.0, 1.0]), ValueError, r"has shape \(2,\) for 1 col"),
        (np.array([[0.0, np.nan]]), ValueError, "must be finite"),
        (np.array([[-np.inf, 0.0]]), ValueError, "must be finite"),
        (np.array([[0, 1]]), TypeError, "must be float64"),
    ],
)
def test_bad_domains_are_rejected(
    domain: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    """Check that encode rejects domains of the wrong shape, order or values."""
    x = np.zeros((5, 1))
    with pytest.raises(error, match=match):
        encode(x, V, domain, 1)


def test_a_narrow_range_far_from_zero_raises() -> None:
    """
    Check that a range too narrow for its distance from 0 raises, not wraps.

    [1e6, 1e6 + ulp] at V = 2^16 puts its zero point past 2^62 levels;
    [1e6, 1e6 + 1e-6] at V = 64 leaves z in int64, but steps of 1e6 then
    carry s·x past 2^53 within 150 rows.
    """
    far = np.array([[1e6, np.nextafter(1e6, np.inf)]])
    with pytest.raises(ValueError, match="widen the range"):
        level_grid(2**16, far, 1)
    x = 1e6 * np.arange(200, dtype=np.float64)[:, None]
    with pytest.raises(ValueError, match="widen a narrow domain_of_definition"):
        encode(x, V, np.array([[1e6, 1e6 + 1e-6]]), 1)


@pytest.mark.parametrize(("order", "headroom"), ORDERS)
def test_the_range_of_the_steps_is_exact(
    order: int | Difference,
    headroom: int,
) -> None:
    """
    Check that symmetric=False returns the tightest float64 range of the exact steps.

    The float differences round, so their extremes can miss the exact ones.
    The range must hold every exact step with each bound the float next to
    its extreme, and symmetric=True gives the larger of |lo| and |hi|.
    """
    del headroom
    rng = np.random.default_rng(3)
    n_rows = 60
    x = np.column_stack(
        [
            np.cumsum(rng.normal(0.1, 1.0, n_rows)),
            np.cumsum(rng.exponential(size=n_rows)) * 1e-7,
            rng.normal(size=n_rows) * 10.0 ** rng.integers(-8, 8, n_rows),
        ],
    )
    ranges = get_domain_of_definition(x, order, symmetric=False)
    bounds = get_domain_of_definition(x, order)
    np.testing.assert_array_equal(bounds, np.maximum(-ranges[:, 0], ranges[:, 1]))
    for c in range(x.shape[1]):
        steps = _exact_steps(x[:, c], order)
        lo, hi = ranges[c]
        assert Fraction(lo) <= min(steps) < Fraction(np.nextafter(lo, np.inf))
        assert Fraction(np.nextafter(hi, -np.inf)) < max(steps) <= Fraction(hi)


def test_a_quantile_range_is_the_central_interval_of_the_steps() -> None:
    """
    Check quantile with symmetric=False: the steps between (1 - q)/2 and (1 + q)/2.

    A series that only rises has its range above 0, where the symmetric
    quantile D spends half the levels on falls that never come.
    """
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.exponential(size=(1000, 1)) + 0.5, axis=0)
    q = 0.98
    steps = np.diff(x[:, 0])
    ranges = get_domain_of_definition(x, 1, quantile=q, symmetric=False)
    np.testing.assert_array_equal(
        ranges,
        [np.quantile(steps, [(1 - q) / 2, (1 + q) / 2])],
    )
    assert ranges[0, 0] > 0
    np.testing.assert_array_equal(
        get_domain_of_definition(x, 1, quantile=q),
        [np.quantile(np.abs(steps), q)],
    )
