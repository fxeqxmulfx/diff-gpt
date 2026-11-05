"""Check the corrected multiplier against decimal calculus and finite differences."""

import pytest
import torch
from torch.testing import assert_close

from research.transformer_gradient_stress.frequency import (
    decimal_reference,
    stable_multiplier,
)


@pytest.mark.parametrize("raw", [-90, -4, 0, 4, 90])
def test_decimal_frequency_and_gradient(raw: int) -> None:
    """Keep a true nonzero derivative even when native tanh has rounded to one."""
    parameter = torch.tensor(float(raw), requires_grad=True)
    value = stable_multiplier(parameter.double())
    gradient = torch.autograd.grad(value, parameter)[0]
    expected_value, expected_gradient = decimal_reference(raw)
    assert_close(
        value, torch.tensor(expected_value, dtype=torch.float64), rtol=1e-14, atol=0
    )
    assert_close(
        gradient.double(),
        torch.tensor(expected_gradient, dtype=torch.float64),
        rtol=1e-6,
        atol=0,
    )
    assert gradient > 0


def test_multiplier_gradcheck_and_second_derivative() -> None:
    """Verify the usual optimizer derivative and one further derivative locally."""
    value = torch.tensor([0.0, 0.3, -0.7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(stable_multiplier, (value,))
    assert torch.autograd.gradgradcheck(stable_multiplier, (value,))
