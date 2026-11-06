"""Check the independent scalar reference before comparing native operations."""

from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from research.layernorm_affine_guarantees.reference import LAYER, RMS, evaluate


@pytest.mark.parametrize("normalizer", [RMS, LAYER])
def test_analytic_vjps_match_high_precision_finite_differences(normalizer: str) -> None:
    """Differentiate the forward formula independently in every parameter/input."""
    with localcontext() as context:
        context.prec = 80
        values = list(map(Decimal, ["0.125", "-0.25", "0.75"]))
        gamma = list(map(Decimal, ["2", "-1", "0.5"]))
        beta = list(map(Decimal, ["0.125", "0.5", "-0.25"]))
        upstream = list(map(Decimal, ["0.75", "-0.5", "0.25"]))
        epsilon, step = Decimal("0.125"), Decimal("1e-20")
        result = evaluate(values, epsilon, normalizer, upstream, gamma, beta)
        for index, key in enumerate(
            ["input_gradient", "gamma_gradient", "beta_gradient"]
        ):
            for coordinate in range(len(values)):
                low, high = [], []
                for delta, destination in [(-step, low), (step, high)]:
                    arguments = [values.copy(), gamma.copy(), beta.copy()]
                    arguments[index][coordinate] += delta
                    measured = evaluate(
                        arguments[0],
                        epsilon,
                        normalizer,
                        upstream,
                        arguments[1],
                        arguments[2],
                    )
                    destination.extend(measured["output"])
                derivative = sum(
                    u * (b - a) / (2 * step)
                    for u, a, b in zip(upstream, low, high, strict=True)
                )
                assert abs(derivative - result[key][coordinate]) < Decimal("1e-35")


def test_singleton_layernorm_and_zero_rms_are_exact() -> None:
    """Centering erases singleton input; finite-epsilon RMS retains its derivative."""
    with localcontext() as context:
        context.prec = 80
        gamma, beta, upstream = [Decimal(2)], [Decimal(3)], [Decimal(4)]
        layer = evaluate([Decimal(7)], Decimal(1), LAYER, upstream, gamma, beta)
        assert layer == {
            "output": [Decimal(3)],
            "input_gradient": [Decimal(0)],
            "gamma_gradient": [Decimal(0)],
            "beta_gradient": [Decimal(4)],
        }
        rms = evaluate([Decimal(0)], Decimal(1), RMS, upstream, gamma, beta)
        assert rms == {**layer, "input_gradient": [Decimal(8)]}
