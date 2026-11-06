"""Independent Decimal evaluation of the documented native AdamW recurrence."""

# ruff: noqa: FURB164 -- Explicit conversion from represented binary floats is intentional.

from __future__ import annotations

from decimal import Decimal, localcontext


def decimal_trace(
    initial: list[float],
    gradients: list[list[float]],
    rates: list[float],
    betas: list[float],
    epsilon: float,
    decay: float,
) -> list[dict]:
    """Evaluate exact represented inputs with 80-digit scalar arithmetic."""
    with localcontext() as context:
        context.prec = 80
        one, zero = Decimal(1), Decimal(0)
        b1, b2 = (Decimal.from_float(value) for value in betas)
        eps, wd = Decimal.from_float(epsilon), Decimal.from_float(decay)
        parameter = [Decimal.from_float(value) for value in initial]
        moment, variance = [zero] * len(initial), [zero] * len(initial)
        history = []
        for clock, (supplied, rate) in enumerate(zip(gradients, rates, strict=True), 1):
            eta = Decimal.from_float(rate)
            gradient = [Decimal.from_float(value) for value in supplied]
            if len(gradient) != len(initial):
                message = "Gradient coordinate count changed"
                raise ValueError(message)
            moment = [
                b1 * m + (one - b1) * g for m, g in zip(moment, gradient, strict=True)
            ]
            variance = [
                b2 * v + (one - b2) * g * g
                for v, g in zip(variance, gradient, strict=True)
            ]
            direction = [
                (m / (one - b1**clock)) / ((v / (one - b2**clock)).sqrt() + eps)
                for m, v in zip(moment, variance, strict=True)
            ]
            parameter = [
                (one - eta * wd) * p - eta * u
                for p, u in zip(parameter, direction, strict=True)
            ]
            history.append(
                {
                    "clock": clock,
                    "parameter": parameter,
                    "moment": moment,
                    "variance": variance,
                }
            )
        return history


def represented_trace(history: list[dict]) -> list[dict]:
    """Round only the reference's exported observations to Python binary64."""
    return [
        {
            "clock": row["clock"],
            **{
                name: [float(value) for value in row[name]]
                for name in ["parameter", "moment", "variance"]
            },
        }
        for row in history
    ]
