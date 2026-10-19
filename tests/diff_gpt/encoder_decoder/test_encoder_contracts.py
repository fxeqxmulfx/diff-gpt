# Copyright 2025 Mikhail Borisov
"""Input checks, the clipping report and edge cases of the encoder."""

import warnings
from decimal import Decimal

import numpy as np
import pytest

from diff_gpt.encoder_decoder import (
    TokenClippingWarning,
    decode,
    differentiate,
    encode,
    get_domain_of_definition,
    integrate,
    np_is_decimal,
    np_to_decimal,
)

V = 64


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_encode_and_domain_reject_nan_and_infinity(bad: float) -> None:
    """
    Check that encode and get_domain_of_definition raise on NaN and infinity.

    The domain came out NaN, the column was then taken for a constant one,
    and LGT emitted garbage tokens with only numpy's cast warning.
    """
    x = np.array([[0.0], [1.0], [bad], [2.0]])
    with pytest.raises(ValueError, match="inp must be finite"):
        encode(x, V, np.array([1.0]), 1, use_decimal=False)
    with pytest.raises(ValueError, match="inp must be finite"):
        get_domain_of_definition(x, 1, use_decimal=False)
    finite = np.array([[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="domain_of_definition must be finite"):
        encode(finite, V, np.array([bad]), 1, use_decimal=False)


def test_encode_rejects_decimal_nan() -> None:
    """Check that a Decimal NaN raises as the float one does."""
    x = np_to_decimal(np.array([[0.0], [1.0], [2.0]]))
    x[1, 0] = Decimal("NaN")
    with pytest.raises(ValueError, match="inp must be finite"):
        encode(x, V, np_to_decimal(np.array([1.0])), 1, use_decimal=True)


def test_lgt_rejects_levels_past_the_int64_grid() -> None:
    """
    Check that LGT raises where |s·x| would leave the int64 level grid.

    A constant column gets the scale 2^20, so a level of 2^44 used to wrap
    around in the cast to int64 and decode to garbage.
    """
    x = np.full((10, 1), 2.0**44)
    dod = get_domain_of_definition(x, 1, use_decimal=False)
    with pytest.raises(ValueError, match="level grid"):
        encode(x, V, dod, 1, use_decimal=False, tokenizer="lgt")


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_encode_warns_with_the_number_of_clipped_tokens(tokenizer: str) -> None:
    """
    Check that encode reports the tokens it clips.

    A step of 8 against a domain of 1 is out of range; it used to be clipped
    silently, leaving a level error in everything decoded after it.
    """
    x = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    with pytest.warns(TokenClippingWarning, match="1 of 4 tokens"):
        encode(x, V, np.array([1.0]), 1, use_decimal=False, tokenizer=tokenizer)


# Steps that MASH at k = 1 takes to make up the jump below: it needs about
# 1240 levels at V = 64, and each later step leaves about 31 levels of room.
CATCH_UP_STEPS = 50


@pytest.mark.parametrize("use_decimal", [False, True])
def test_mash_k1_makes_up_a_step_outside_the_domain(*, use_decimal: bool) -> None:
    """
    Check that MASH at k = 1 makes a jump of 40·D up within a few dozen steps.

    The jump's token was clipped after the feedback loop, so every later
    decoded value stayed off by 39.0, against Δ/2 = 0.016. The tokens before
    the jump stay as they were.
    """
    steps = np.random.default_rng(0).uniform(-1, 1, size=(400, 1))
    steps[200] = 40.0
    x = np.concatenate([np.zeros((1, 1)), np.cumsum(steps, axis=0)])
    dod = np.array([1.0])
    if use_decimal:
        x, dod = np_to_decimal(x), np_to_decimal(dod)
    with pytest.warns(TokenClippingWarning, match="1 of 400 tokens"):
        st, sc, tok = encode(x, V, dod, 1, use_decimal=use_decimal, tokenizer="mash")
    _, _, before = encode(x[:201], V, dod, 1, use_decimal=use_decimal, tokenizer="mash")
    assert np.array_equal(tok[:200], before)
    dec = decode(st, sc, tok, V, 1, use_decimal=use_decimal, tokenizer="mash")
    err = np.abs((dec - x).astype(np.float64))[:, 0]
    half_step = 1.0 / (V - 2) * (1 + 1e-9)  # Δ/2 = D / (V - 2), D = 1
    assert err[:201].max() <= half_step
    assert err[201 + CATCH_UP_STEPS :].max() <= half_step


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_encode_in_domain_does_not_warn(tokenizer: str) -> None:
    """Check that a series inside its domain encodes without a warning."""
    x = np.array([[0.0], [1.0], [2.0], [1.5], [0.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encode(x, V, np.array([1.0]), 1, use_decimal=False, tokenizer=tokenizer)


def test_mash_scalar_order_0_rounds_like_the_per_column_order_0() -> None:
    """
    Check that MASH with a scalar order 0 rounds plainly, within Δ/2.

    The scalar order went through the carry scheme instead: its tokens
    differed from those of the order [0] and its error reached 1.97·Δ/2,
    against README Theorem 1.
    """
    x = np.random.default_rng(0).normal(size=(2000, 1))
    dod = get_domain_of_definition(x, 0, use_decimal=False)
    st, sc, scalar = encode(x, V, dod, 0, use_decimal=False, tokenizer="mash")
    _, _, per_column = encode(
        x,
        V,
        dod,
        np.array([0]),
        use_decimal=False,
        tokenizer="mash",
    )
    assert np.array_equal(scalar, per_column)
    dec = decode(st, sc, scalar, V, 0, use_decimal=False, tokenizer="mash")
    # Δ/2 = 1 / (2s) = D / (V - 2)
    assert np.abs(dec - x).max() <= dod[0] / (V - 2) * (1 + 1e-12)


def test_integrate_inverts_differentiate_across_order_forms() -> None:
    """
    Check that a scalar-order differentiate inverts with the per-column order.

    `start` used to take another form for scalar orders than for per-column
    ones, and integrate did not tell them apart: differentiate(x, 2)
    followed by integrate with the order [2, 2, 2] was off by 2997. No column
    starts at 0: there the old form [x0, x1 - x0] equals the raw [x0, x1].
    """
    t = np.arange(1000, dtype=np.float64)
    x = np.stack([t + 3.0, t**2 / 1000 - 2.0, np.cos(t / 10)], axis=1)
    start, diff = differentiate(x, 2, use_decimal=False)
    assert np.array_equal(start, x[:2])
    back = integrate(start, diff, np.array([2, 2, 2]), use_decimal=False)
    np.testing.assert_allclose(back, x, atol=1e-6)


def test_integrate_rejects_a_start_of_the_wrong_shape() -> None:
    """Check that integrate raises on a start with too few rows or columns."""
    t = np.arange(50, dtype=np.float64)
    x = np.stack([t, t**2, -t], axis=1)
    start, diff = differentiate(x, 2, use_decimal=False)
    with pytest.raises(ValueError, match="exceeds the 1-row start prefix"):
        integrate(start[1:], diff, 2, use_decimal=False)
    with pytest.raises(ValueError, match="start has shape"):
        integrate(start[:, :2], diff, 2, use_decimal=False)


@pytest.mark.parametrize("order", [2, np.array([2, 1])])
def test_domain_needs_a_row_past_the_order(order: int | np.ndarray) -> None:
    """
    Check that a series of max_k rows raises a clear ValueError for any order form.

    The scalar order failed inside numpy ("zero-size array"), the
    per-column one returned a domain of 0.
    """
    x = np.array([[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="needs a series longer than 2 rows"):
        get_domain_of_definition(x, order, use_decimal=False)


def test_np_is_decimal_on_arrays_without_columns() -> None:
    """Check that np_is_decimal goes by the dtype of an empty array, not inp[0, 0]."""
    assert not np_is_decimal(np.zeros((3, 0)))
    assert np_is_decimal(np.empty((3, 0), dtype=object))
