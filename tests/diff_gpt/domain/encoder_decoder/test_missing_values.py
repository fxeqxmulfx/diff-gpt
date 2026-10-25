"""
Missing values: NaN marks them, and the steps that read one stay out.

`encode` takes no NaN; a series with gaps is encoded as `fill_missing` fills
it, `observed_tokens` marks the tokens that read no NaN, and
`get_domain_of_definition` bounds only those steps.
"""

import warnings

import numpy as np
import pytest

from diff_gpt.domain.encoder_decoder import (
    Difference,
    TokenClippingWarning,
    encode,
    fill_missing,
    get_domain_of_definition,
    observed_tokens,
)

V = 64
NAN = np.nan


def test_fill_missing_draws_the_line_across_a_gap_and_holds_the_ends() -> None:
    """Check the fill: the line across a gap, the nearest value at either end."""
    x = np.array(
        [
            [NAN, 1.0],
            [2.0, 1.0],
            [NAN, 1.0],
            [NAN, 1.0],
            [8.0, 1.0],
            [NAN, 1.0],
        ],
    )
    expected = np.array(
        [
            [2.0, 1.0],
            [2.0, 1.0],
            [4.0, 1.0],
            [6.0, 1.0],
            [8.0, 1.0],
            [8.0, 1.0],
        ],
    )
    np.testing.assert_array_equal(fill_missing(x), expected)
    assert np.isnan(x[[0, 2, 3, 5], 0]).all()


def test_fill_missing_rejects_a_column_of_nan_and_infinities() -> None:
    """Check that a column without a value and an infinity raise, as other dtypes do."""
    with pytest.raises(ValueError, match="column 1 holds no value"):
        fill_missing(np.array([[1.0, NAN], [2.0, NAN]]))
    with pytest.raises(ValueError, match="only NaN marks a missing value"):
        fill_missing(np.array([[1.0], [np.inf]]))
    with pytest.raises(TypeError, match="inp must be float64"):
        fill_missing(np.array([[1.0], [NAN]], dtype=np.float32))


@pytest.mark.parametrize(
    ("order", "degree", "reach"),
    [
        (0, 0, [0]),
        (1, 1, [0, 1]),
        (2, 2, [0, 1, 2]),
        # (1 - B)(1 - B^3) = 1 - B - B^3 + B^4 does not read x_{t-2}.
        (Difference(1, 3), 4, [0, 1, 3, 4]),
    ],
)
def test_a_nan_leaves_out_the_tokens_of_the_steps_that_read_it(
    order: int | Difference,
    degree: int,
    reach: list[int],
) -> None:
    """Check that a NaN at t leaves out the tokens at t + j with c_j != 0 alone."""
    n_rows, nan_at = 12, 6
    x = np.arange(n_rows, dtype=np.float64)[:, None]
    x[nan_at] = NAN
    expected = np.ones(n_rows - degree, dtype=bool)
    expected[[nan_at + j - degree for j in reach]] = False
    np.testing.assert_array_equal(observed_tokens(x, order)[:, 0], expected)


def test_observed_tokens_line_up_with_the_tokens_of_per_column_orders() -> None:
    """
    Check the rows of the mask where the columns differ in degree.

    The tokens start after the largest degree, 4 here, so the order 1 NaN
    at t = 6 leaves out the rows of t = 6 and 7, rows 2 and 3.
    """
    x = np.column_stack([np.arange(12.0), np.arange(12.0) ** 2])
    x[6, 0] = NAN
    mask = observed_tokens(x, [1, Difference(1, 3)])
    expected = np.ones((8, 2), dtype=bool)
    expected[[2, 3], 0] = False
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize("tokenizer", ["lgt", "mash", "mash-half"])
def test_the_observed_tokens_do_not_depend_on_the_fill(tokenizer: str) -> None:
    """
    Check that any fill of the NaN after the prefix leaves the observed tokens be.

    Every grid is anchored before the gaps, "lgt" at 0 and the others at the
    prefix, so a token that reads no NaN reads no filled value either.
    """
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(80, 2)), axis=0)
    x[[10, 11, 12, 40, 79], 0] = NAN
    x[[30, 50], 1] = NAN
    order = Difference(1, 3)
    mask = observed_tokens(x, order)
    domain = get_domain_of_definition(x, order, symmetric=False)
    with warnings.catch_warnings():
        # The steps of the filled values may leave the domain and clip.
        warnings.simplefilter("ignore", TokenClippingWarning)
        _, _, linear = encode(fill_missing(x), V, domain, order, tokenizer=tokenizer)
        _, _, other = encode(
            np.where(np.isnan(x), 100.0, x),
            V,
            domain,
            order,
            tokenizer=tokenizer,
        )
    np.testing.assert_array_equal(linear[mask], other[mask])
    assert (linear[~mask] != other[~mask]).any()


def test_the_domain_bounds_only_the_steps_that_read_no_nan() -> None:
    """
    Check get_domain_of_definition across a gap.

    The fill of [1, NaN, 10] steps by 4.5; the steps observed are all 1.
    """
    x = np.array([[0.0], [1.0], [NAN], [10.0], [11.0]])
    np.testing.assert_array_equal(get_domain_of_definition(x, 1), [1.0])
    np.testing.assert_array_equal(
        get_domain_of_definition(x, 1, symmetric=False),
        [[1.0, 1.0]],
    )
    np.testing.assert_array_equal(get_domain_of_definition(x, 1, quantile=0.5), [1.0])


def test_the_domain_of_a_column_without_an_observed_step_raises() -> None:
    """Check that a column whose every step reads a NaN raises, at the order 1."""
    x = np.array([[0.0, 0.0], [NAN, 1.0], [2.0, 2.0], [NAN, 3.0]])
    with pytest.raises(ValueError, match="column 0 has no step"):
        get_domain_of_definition(x, 1)
    np.testing.assert_array_equal(get_domain_of_definition(x, 0), [2.0, 3.0])
