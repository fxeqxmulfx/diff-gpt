"""
The opt-in tokenizer "mash-half": the grid of "mash" with c(B) A = 1/2.

The half step of the order 0 moves the integers c(B) L half a level off the
steps, so the tokens take all of [0, V-1] and the scale gets one level
more: s·D <= (V - ‖c‖₁ + 1)/2, and the error stays within
D/(V - ‖c‖₁ + 1). In exchange a constant encodes as a periodic pattern of
tokens and comes back up to 1/(2s) off.
"""

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from diff_gpt.encoder_decoder import (
    CONSTANT_COLUMN_SCALE,
    Difference,
    decode,
    encode,
    get_domain_of_definition,
    level_grid,
)

V = 64
EPS = float(np.finfo(np.float64).eps)
# The decode rounds x̂ = (R + F)/s, which may add a few ulps to Δ/2.
DECODE_SLACK = 1 + 1e-9
# Each operator with ‖c‖₁: 2^k for Δ^k, and 6 for (1 - B)(1 - B^2)(1 - B^3),
# whose B^3 terms cancel.
OPERATORS = [
    (Difference(1), 2),
    (Difference(1, 1), 4),
    (Difference(1, 1, 1), 8),
    (Difference(1, 1, 1, 1), 16),
    (Difference(1, 1, 1, 1, 1), 32),
    (Difference(24), 2),
    (Difference(1, 24), 4),
    (Difference(1, 2, 3), 6),
]


def _half_scale(vocab_size: int, bound: float, norm: int) -> float:
    """
    Return the scale of [-D, D] under "mash-half".

    The largest float64 s with s·D <= (V - ‖c‖₁ + 1)/2;
    `CONSTANT_COLUMN_SCALE` for D <= eps.
    """
    if bound <= EPS:
        return CONSTANT_COLUMN_SCALE
    room = Fraction(vocab_size - norm + 1, 2)
    s = math.nextafter(float(room / Fraction(bound)), math.inf)
    while Fraction(s) * Fraction(bound) > room:
        s = math.nextafter(s, 0.0)
    return s


def _walks(n_rows: int, seed: int) -> np.ndarray:
    """Return two random walks integrated once more, smooth enough for Δ^5."""
    steps = np.random.default_rng(seed).normal(size=(n_rows, 2))
    return np.cumsum(np.cumsum(steps, axis=0), axis=0)


@pytest.mark.parametrize(("order", "norm"), OPERATORS)
def test_the_error_stays_within_d_over_one_more_level(
    order: Difference,
    norm: int,
) -> None:
    """
    Check the bound of "mash-half" from V = ‖c‖₁ + 2 up.

    On its own domain no step clips, the tokens lie in [0, V-1], the scale
    is the largest with s·D <= (V - ‖c‖₁ + 1)/2 and z = V/2, the start rows
    come back as given and every column stays within D/(V - ‖c‖₁ + 1).
    """
    x = _walks(3_000, seed=0)
    dod = get_domain_of_definition(x, order)
    p = sum(order.lags)
    for vocab_size in (norm + 2, V, 4096):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            start, grid, tokens = encode(
                x,
                vocab_size,
                dod,
                order,
                tokenizer="mash-half",
            )
        assert tokens.min() >= 0
        assert tokens.max() <= vocab_size - 1
        expected = [_half_scale(vocab_size, d, norm) for d in dod]
        np.testing.assert_array_equal(grid.scale, expected)
        np.testing.assert_array_equal(grid.zero, vocab_size // 2)
        dec = decode(start, grid, tokens, vocab_size, order, tokenizer="mash-half")
        assert np.array_equal(dec[:p], x[:p])
        err = np.abs(dec - x).max(axis=0)
        assert (err <= dod / (vocab_size - norm + 1) * DECODE_SLACK).all()


@pytest.mark.parametrize(("order", "norm"), [OPERATORS[0], OPERATORS[6]])
def test_a_symmetric_domain_takes_the_scale_of_one_more_level(
    order: Difference,
    norm: int,
) -> None:
    """Check the scale and z = V/2 of D and [-D, D] from 0 to 1e300."""
    rng = np.random.default_rng(1)
    bounds = np.concatenate(
        [rng.lognormal(0.0, 8.0, 300), [0.0, EPS, 2 * EPS, 1.0, 1e300]],
    )
    for vocab_size in (norm + 2, V, 4096):
        expected = [_half_scale(vocab_size, d, norm) for d in bounds]
        for domain in (bounds, np.stack([-bounds, bounds], axis=1)):
            grid = level_grid(vocab_size, domain, order, tokenizer="mash-half")
            np.testing.assert_array_equal(grid.scale, expected)
            np.testing.assert_array_equal(grid.zero, vocab_size // 2)


def test_ramps_at_the_edge_of_the_domain_take_the_first_and_the_last_token() -> None:
    """
    Check steps of D = 1 up and down at Δ: tokens V - 1 and 0.

    At s = (V - 1)/2 a step of ±1 makes c(B) L = ±s - 1/2, 31 or -32 at
    V = 64, exactly. "mash" never emits token 0: at s = V/2 - 1 its steps of
    -1 take token 1.
    """
    x = np.concatenate([np.arange(11.0), np.arange(9.0, -1.0, -1.0)])[:, None]
    dod = get_domain_of_definition(x, 1)
    assert dod[0] == 1.0
    ramps = {"mash": [V - 1] * 10 + [1] * 10, "mash-half": [V - 1] * 10 + [0] * 10}
    for tokenizer, expected in ramps.items():
        start, grid, tokens = encode(x, V, dod, 1, tokenizer=tokenizer)
        np.testing.assert_array_equal(tokens[:, 0], expected)
        dec = decode(start, grid, tokens, V, 1, tokenizer=tokenizer)
        assert np.abs(dec - x).max() <= 1 / (2 * grid.scale[0])


@pytest.mark.parametrize(
    ("k", "period"),
    [
        (1, [32, 31]),
        (2, [32, 31, 31, 32]),
        (3, [32, 30, 33, 31]),
        (4, [32, 30, 33, 31, 31, 33, 30, 32]),
    ],
)
def test_a_constant_encodes_as_a_periodic_pattern(k: int, period: list[int]) -> None:
    """
    Check the price of the half step on zeros at Δ^k, V = 64, D = 1.

    "mash" keeps the zeros on its grid: token V/2 throughout, decoded
    exactly. Under "mash-half" the anchor C(t, k)/2 is a half-integer at
    some t, where the zeros fall on a tie between two levels and round up:
    the tokens repeat with a period, and the zeros come back as 0 or 1/(2s).
    """
    x = np.zeros((k + 4 * len(period), 1))
    domain = np.array([1.0])
    _, _, mash = encode(x, V, domain, k, tokenizer="mash")
    np.testing.assert_array_equal(mash, V // 2)
    start, grid, tokens = encode(x, V, domain, k, tokenizer="mash-half")
    np.testing.assert_array_equal(tokens[:, 0], period * 4)
    dec = decode(start, grid, tokens, V, k, tokenizer="mash-half")
    np.testing.assert_array_equal(np.unique(dec), [0.0, 0.5 / grid.scale[0]])


def test_the_order_0_takes_the_half_step_already() -> None:
    """
    Check that "mash-half" is "mash" in a column of the order 0.

    Every tokenizer takes A = 1/2 there: c(B) A = 1/2 with c = 1. The
    column of the order 1 next to it gets the larger scale.
    """
    x = _walks(500, seed=2)
    order = [0, 1]
    dod = get_domain_of_definition(x, order)
    _, mash_grid, mash = encode(x, V, dod, order, tokenizer="mash")
    _, half_grid, half = encode(x, V, dod, order, tokenizer="mash-half")
    assert half_grid.scale[0] == mash_grid.scale[0]
    assert half_grid.zero[0] == mash_grid.zero[0]
    np.testing.assert_array_equal(half[:, 0], mash[:, 0])
    assert half_grid.scale[1] > mash_grid.scale[1]
