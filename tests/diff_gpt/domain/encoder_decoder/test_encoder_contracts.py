"""Input checks, the clipping report and edge cases of the encoder."""

import warnings

import numpy as np
import pytest

from diff_gpt.domain.encoder_decoder import (
    TokenClippingWarning,
    decode,
    differentiate,
    encode,
    get_domain_of_definition,
    integrate,
)

V = 64
TOKENIZERS = ["lgt", "mash", "mash-half"]


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_encode_and_domain_reject_infinity(bad: float) -> None:
    """
    Check that encode and get_domain_of_definition raise on infinity.

    The domain came out NaN, the column was then taken for a constant one,
    and LGT emitted garbage tokens with only numpy's cast warning.
    """
    x = np.array([[0.0], [1.0], [bad], [2.0]])
    with pytest.raises(ValueError, match="inp must be finite"):
        encode(x, V, np.array([1.0]), 1)
    with pytest.raises(ValueError, match="holds infinity"):
        get_domain_of_definition(x, 1)
    finite = np.array([[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="domain_of_definition must be finite"):
        encode(finite, V, np.array([bad]), 1)


def test_encode_rejects_nan_and_names_the_fill() -> None:
    """Check that encode raises on NaN, naming fill_missing, and on a NaN domain."""
    x = np.array([[0.0], [1.0], [np.nan], [2.0]])
    with pytest.raises(ValueError, match="fill_missing"):
        encode(x, V, np.array([1.0]), 1)
    finite = np.array([[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="domain_of_definition must be finite"):
        encode(finite, V, np.array([np.nan]), 1)


def test_values_other_than_float64_are_rejected() -> None:
    """Check that the encoder takes float64 values only, e.g. no object arrays."""
    x = np.array([[0.0], [1.0], [2.0]])
    with pytest.raises(TypeError, match="inp must be float64"):
        encode(x.astype(object), V, np.array([1.0]), 1)
    with pytest.raises(TypeError, match="inp must be float64"):
        get_domain_of_definition(x.astype(object), 1)
    st, grid, tok = encode(x, V, np.array([1.0]), 1)
    with pytest.raises(TypeError, match="start must be float64"):
        decode(st.astype(np.float32), grid, tok, V, 1)


# The order from which the int64 limit on the levels, 2^(62 - k), binds
# before the float64 one, 2^53.
INT64_BOUND_ORDER = 12


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_levels_past_the_int64_grid_are_rejected(tokenizer: str) -> None:
    """
    Check that encode raises where |s·x| would leave the int64 level grid.

    A constant column gets the scale 2^20, so a level of 2^44 used to wrap
    around in the cast to int64 and decode to garbage. From k = 12 on, the
    levels must stay below 2^(62 - k), so that their k-th differences fit.
    """
    x = np.full((20, 1), 2.0**44)
    with pytest.raises(ValueError, match="level grid"):
        encode(x, V, get_domain_of_definition(x, 1), 1, tokenizer=tokenizer)
    x = np.full((20, 1), 2.0**30)  # levels of 2^50
    vocab_size = 2 ** (INT64_BOUND_ORDER + 1)
    for k in (INT64_BOUND_ORDER - 1, INT64_BOUND_ORDER):
        dod = get_domain_of_definition(x, k)
        if k < INT64_BOUND_ORDER:
            encode(x, vocab_size, dod, k, tokenizer=tokenizer)
        else:
            with pytest.raises(ValueError, match=r"\|s·x\| < 2\^50"):
                encode(x, vocab_size, dod, k, tokenizer=tokenizer)


# Domains of x = 1 at V = 64, where s·x = 31 / D, on either side of 2^53.
FINEST_DOMAIN = 4e-15  # s·x = 7.8e15
TOO_FINE_DOMAIN = 3e-15  # s·x = 1.03e16


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_a_grid_finer_than_float64_resolves_is_rejected(tokenizer: str) -> None:
    """
    Check that encode raises where |s·x| reaches 2^53 and float64 misses levels.

    Sampled every 10^-6, one sin/cos period has Δ³x of 10^-18, below the
    float64 noise of the values, so D is that noise and s·x reaches 7·10^16.
    The rounding of s·x then moves levels by whole steps: with the int64
    guard alone, over a thousand tokens clipped and LGT decoded 0.03 off,
    against Δ/2 of 10^-17.
    """
    x = np.ones((10, 1))
    encode(x, V, np.array([FINEST_DOMAIN]), 1, tokenizer=tokenizer)
    with pytest.raises(ValueError, match=r"\|s·x\| < 2\^53"):
        encode(x, V, np.array([TOO_FINE_DOMAIN]), 1, tokenizer=tokenizer)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encode_warns_with_the_number_of_clipped_tokens(tokenizer: str) -> None:
    """
    Check that encode reports the tokens it clips.

    A step of 8 against a domain of 1 is out of range; it used to be clipped
    silently, leaving a level error in everything decoded after it.
    """
    x = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    with pytest.warns(TokenClippingWarning, match="1 of 4 tokens"):
        encode(x, V, np.array([1.0]), 1, tokenizer=tokenizer)


# The jump below exceeds the largest step the grid holds, D = 1, by this much.
JUMP_EXCESS = 39.0


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_a_clipped_step_offsets_the_values_after_it(tokenizer: str) -> None:
    """
    Check that a clipped jump of 40·D shifts every later value by its excess.

    The tokenizers clip alike for every k. Before version 2, MASH at k = 1
    alone made the excess up over the next steps instead.
    """
    steps = np.random.default_rng(0).uniform(-1, 1, size=(400, 1))
    steps[200] = 40.0
    x = np.concatenate([np.zeros((1, 1)), np.cumsum(steps, axis=0)])
    dod = np.array([1.0])
    with pytest.warns(TokenClippingWarning, match="1 of 400 tokens"):
        st, grid, tok = encode(x, V, dod, 1, tokenizer=tokenizer)
    _, _, before = encode(x[:201], V, dod, 1, tokenizer=tokenizer)
    assert np.array_equal(tok[:200], before)
    err = (decode(st, grid, tok, V, 1, tokenizer=tokenizer) - x)[:, 0]
    half_step = 1.0 / (V - 2) * (1 + 1e-9)  # Δ/2 = D / (V - 2), D = 1
    assert np.abs(err[:201]).max() <= half_step
    # The excess is a whole number of levels, so after the jump the error
    # stays within Δ/2 of one offset.
    offset = np.median(err[201:])
    assert abs(offset + JUMP_EXCESS) <= 2 * half_step
    assert np.abs(err[201:] - offset).max() <= 2 * half_step


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_encode_in_domain_does_not_warn(tokenizer: str) -> None:
    """Check that a series inside its domain encodes without a warning."""
    x = np.array([[0.0], [1.0], [2.0], [1.5], [0.5]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encode(x, V, np.array([1.0]), 1, tokenizer=tokenizer)


def test_mash_scalar_order_0_rounds_like_the_per_column_order_0() -> None:
    """
    Check that MASH with a scalar order 0 rounds plainly, within Δ/2.

    The scalar order went through the carry scheme instead: its tokens
    differed from those of the order [0] and its error reached 1.97·Δ/2,
    against README Theorem 1. At the order 0 Δ/2 = 1/(2s) is a hair above
    D/V.
    """
    x = np.random.default_rng(0).normal(size=(2000, 1))
    dod = get_domain_of_definition(x, 0)
    st, grid, scalar = encode(x, V, dod, 0, tokenizer="mash")
    _, _, per_column = encode(
        x,
        V,
        dod,
        np.array([0]),
        tokenizer="mash",
    )
    assert np.array_equal(scalar, per_column)
    dec = decode(st, grid, scalar, V, 0, tokenizer="mash")
    assert np.abs(dec - x).max() <= dod[0] / V * (1 + 1e-12)


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
    start, diff = differentiate(x, 2)
    assert np.array_equal(start, x[:2])
    back = integrate(start, diff, np.array([2, 2, 2]))
    np.testing.assert_allclose(back, x, atol=1e-6)


def test_integrate_rejects_a_start_of_the_wrong_shape() -> None:
    """Check that integrate raises on a start with too few rows or columns."""
    t = np.arange(50, dtype=np.float64)
    x = np.stack([t, t**2, -t], axis=1)
    start, diff = differentiate(x, 2)
    with pytest.raises(ValueError, match="exceeds the 1-row start prefix"):
        integrate(start[1:], diff, 2)
    with pytest.raises(ValueError, match="start has shape"):
        integrate(start[:, :2], diff, 2)


@pytest.mark.parametrize("order", [2, np.array([2, 1])])
def test_domain_needs_a_row_past_the_order(order: int | np.ndarray) -> None:
    """
    Check that a series of max_k rows raises a clear ValueError for any order form.

    The scalar order failed inside numpy ("zero-size array"), the
    per-column one returned a domain of 0.
    """
    x = np.array([[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="needs a series longer than 2 rows"):
        get_domain_of_definition(x, order)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize(("k", "vocab_size"), [(0, 2), (1, 2), (2, 4), (3, 8)])
def test_the_vocabulary_must_exceed_the_norm_of_c(
    tokenizer: str,
    k: int,
    vocab_size: int,
) -> None:
    """
    Check that V <= ‖c‖₁ = 2^k, with no room for the rounding noise, raises.

    The order 0 needs no room, but V must be even and at least 4.
    """
    x = np.random.default_rng(0).normal(size=(20, 1))
    dod = get_domain_of_definition(x, k)
    with pytest.raises(ValueError, match=r"vocab_size > ‖c‖₁|even, >= 4"):
        encode(x, vocab_size, dod, k, tokenizer=tokenizer)
    encode(x, vocab_size + 2, dod, k, tokenizer=tokenizer)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_the_vocabulary_stays_below_2_to_the_62(tokenizer: str) -> None:
    """
    Check that V >= 2^62 raises, in encode and in decode.

    The clip bound V - z of `encode` must stay in int64, and the Lean proof
    of the round trip (`encode_decode`) takes V < 2^62.
    """
    x = np.random.default_rng(0).normal(size=(20, 1))
    dod = get_domain_of_definition(x, 1)
    st, grid, tokens = encode(x, V, dod, 1, tokenizer=tokenizer)
    with pytest.raises(ValueError, match=r"< 2\^62"):
        encode(x, 2**62, dod, 1, tokenizer=tokenizer)
    with pytest.raises(ValueError, match=r"< 2\^62"):
        decode(st, grid, tokens, 2**62, 1, tokenizer=tokenizer)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_decode_raises_where_the_tokens_leave_int64(tokenizer: str) -> None:
    """
    Check that decode raises on tokens whose levels would wrap around in int64.

    At k = 3, 3·10^6 tokens of the largest step reach 31·(3·10^6)^3/6 levels,
    past 2^63; the int64 sums would wrap and decode to garbage.
    """
    x = np.random.default_rng(0).normal(size=(10, 1))
    st, grid, _ = encode(x, V, np.array([10.0]), 3, tokenizer=tokenizer)
    tokens = np.full((3_000_000, 1), V - 1, dtype=np.int64)
    with pytest.raises(ValueError, match="outside int64"):
        decode(st, grid, tokens, V, 3, tokenizer=tokenizer)
    assert np.isfinite(decode(st, grid, tokens[:1000], V, 3, tokenizer=tokenizer)).all()
