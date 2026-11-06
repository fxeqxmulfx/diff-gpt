"""Validate the independent optimizer reference against closed-form solutions."""

# ruff: noqa: FURB164 -- Explicit conversion from represented binary floats is intentional.

from decimal import Decimal, localcontext

from research.adamw_gradient_scaling.reference import decimal_trace


def test_constant_input_matches_closed_form_with_zero_rate_start() -> None:
    """The constant direction yields an independent product formula for parameters."""
    initial, supplied = [0.02, -0.03, 0.0], [1e-12, -1.0, 0.0]
    betas, eps, decay = [0.9, 0.98], 1e-8, 0.1
    rates = [0.002 * min(1.0, k / 50) for k in range(64)]
    result = decimal_trace(initial, [supplied] * len(rates), rates, betas, eps, decay)
    with localcontext() as context:
        context.prec = 80
        b1, b2 = map(Decimal.from_float, betas)
        epsilon, wd = Decimal.from_float(eps), Decimal.from_float(decay)
        product = Decimal(1)
        for row, rate in zip(result, rates, strict=True):
            product *= 1 - Decimal.from_float(rate) * wd
            for index, (p, g) in enumerate(zip(initial, supplied, strict=True)):
                point, gradient = Decimal.from_float(p), Decimal.from_float(g)
                direction = gradient / (abs(gradient) + epsilon)
                expected = (point + direction / wd) * product - direction / wd
                assert abs(row["parameter"][index] - expected) < Decimal("1e-70")
                assert abs(
                    row["moment"][index] - (1 - b1 ** row["clock"]) * gradient
                ) < Decimal("1e-70")
                assert abs(
                    row["variance"][index] - (1 - b2 ** row["clock"]) * gradient**2
                ) < Decimal("1e-70")
    assert result[0]["parameter"] == list(map(Decimal.from_float, initial))
    assert any(result[0]["moment"])
    assert any(result[0]["variance"])


def test_zero_betas_signed_inputs_match_exact_rational_fixture() -> None:
    """An abrupt sign change and zero gradient have a separate rational solution."""
    result = decimal_trace(
        [2.0], [[2.0], [-2.0], [0.0]], [0.0, 0.125, 0.125], [0.0, 0.0], 0.5, 0.25
    )
    assert result[0] == {
        "clock": 1,
        "parameter": [Decimal(2)],
        "moment": [Decimal(2)],
        "variance": [Decimal(4)],
    }
    assert result[1] == {
        "clock": 2,
        "parameter": [Decimal("2.0375")],
        "moment": [Decimal(-2)],
        "variance": [Decimal(4)],
    }
    assert result[2] == {
        "clock": 3,
        "parameter": [Decimal("1.973828125")],
        "moment": [Decimal(0)],
        "variance": [Decimal(0)],
    }
