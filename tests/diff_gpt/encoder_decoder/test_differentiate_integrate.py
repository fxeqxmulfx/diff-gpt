# Copyright 2025 Mikhail Borisov
"""Tests for the differentiate/integrate round trip on Decimal series."""

from decimal import Decimal

import numpy as np
import pytest

from diff_gpt.encoder_decoder import differentiate, integrate, np_to_decimal


@pytest.mark.slow
def test_zero_order_differentiate_integrate_sin_cos() -> None:
    """Check that the order-0 round trip reproduces sin/cos exactly."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 0
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) == Decimal(0)


@pytest.mark.slow
def test_first_order_differentiate_integrate_sin_cos() -> None:
    """Check that the order-1 round trip recovers sin/cos within its error bound."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) <= Decimal("7.802992583885788917541503906E-26")


@pytest.mark.slow
def test_second_order_differentiate_integrate_sin_cos() -> None:
    """Check that the order-2 round trip recovers sin/cos within its error bound."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 2
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


@pytest.mark.slow
def test_third_order_differentiate_integrate_sin_cos() -> None:
    """Check that the order-3 round trip recovers sin/cos within its error bound."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 3
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


@pytest.mark.slow
def test_fourth_order_differentiate_integrate_sin_cos() -> None:
    """Check that the order-4 round trip recovers sin/cos within its error bound."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 4
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


@pytest.mark.slow
def test_fifth_order_differentiate_integrate_sin_cos() -> None:
    """Check the sin/cos round trip; despite the name, it runs order 1."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 1
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) <= Decimal("7.802992583885788917541503906E-26")


@pytest.mark.slow
def test_sixth_order_differentiate_integrate_sin_cos() -> None:
    """Check that the order-6 round trip recovers sin/cos within its error bound."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 6
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


@pytest.mark.slow
def test_seventh_order_differentiate_integrate_sin_cos() -> None:
    """Check that the order-7 round trip recovers sin/cos within its error bound."""
    idx = np.arange(1_000_000, dtype=np.float64)
    inp = np.stack(
        (
            np.sin(idx),
            np.cos(idx),
        ),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 7
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) <= Decimal("2.913827378310570120811462402E-21")


def test_overflow_differentiate_integrate_square_x() -> None:
    """Check that order 7 on 7 points round-trips and order 8 raises ValueError."""
    idx = np.arange(7, dtype=np.float64)
    inp = np.stack(
        (np.square(idx),),
        axis=1,
    )
    inp = np_to_decimal(inp)
    order_of_derivative = 7
    use_decimal = True
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    result = integrate(
        start=start,
        diff=diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    assert np.max(np.abs(result - inp)) == Decimal(0)
    order_of_derivative = 8
    with pytest.raises(ValueError, match="series length"):
        differentiate(
            inp=inp,
            order_of_derivative=order_of_derivative,
            use_decimal=use_decimal,
        )
