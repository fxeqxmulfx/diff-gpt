"""Tests for the differentiate/integrate round trip."""

import numpy as np
import pytest

from diff_gpt.domain.encoder_decoder import differentiate, integrate

# float64 round trips of sin/cos stay this close; the measured error is 1e-15.
ROUND_TRIP_TOLERANCE = 1e-12


def _sin_cos(T: int) -> np.ndarray:
    idx = np.arange(T, dtype=np.float64)
    return np.stack((np.sin(idx), np.cos(idx)), axis=1)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 6, 7])
def test_differentiate_integrate_sin_cos_round_trips(k: int) -> None:
    """Check that the order-k round trip recovers sin/cos to float64 precision."""
    inp = _sin_cos(1_000)
    start, diff = differentiate(inp=inp, order_of_derivative=k)
    result = integrate(start=start, diff=diff, order_of_derivative=k)
    assert np.abs(result - inp).max() <= ROUND_TRIP_TOLERANCE


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 6, 7])
def test_differentiate_integrate_is_exact_on_integer_series(k: int) -> None:
    """Check that integer series round-trip exactly: every sum stays below 2^53."""
    idx = np.arange(1_000, dtype=np.float64)
    inp = np.stack((np.square(idx), idx**3), axis=1)
    start, diff = differentiate(inp=inp, order_of_derivative=k)
    result = integrate(start=start, diff=diff, order_of_derivative=k)
    assert np.array_equal(result, inp)


def test_overflow_differentiate_integrate_square_x() -> None:
    """Check that order 7 on 7 points round-trips and order 8 raises ValueError."""
    idx = np.arange(7, dtype=np.float64)
    inp = np.stack((np.square(idx),), axis=1)
    start, diff = differentiate(inp=inp, order_of_derivative=7)
    result = integrate(start=start, diff=diff, order_of_derivative=7)
    assert np.array_equal(result, inp)
    with pytest.raises(ValueError, match="series length"):
        differentiate(inp=inp, order_of_derivative=8)
