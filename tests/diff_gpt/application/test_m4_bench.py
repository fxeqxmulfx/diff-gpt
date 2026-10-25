"""Check m4_bench's NLL of the values on a grid from the encoder, without M4."""

import math

import numpy as np
import pytest

from benchmarks.m4_bench import nll_x
from diff_gpt.domain.encoder_decoder import encode


@pytest.mark.parametrize("bound", [3.0, 0.25])
def test_nll_x_of_uniform_tokens_is_the_uniform_density_of_the_steps(
    bound: float,
) -> None:
    """Tokens uniform on the V - 2 levels of [-D, D] give the values NLL log 2D."""
    _, grid, _ = encode(
        inp=np.zeros((4, 1)),
        vocab_size=256,
        domain_of_definition=np.array([bound]),
        order_of_derivative=1,
    )
    # At the order 1, s = (V - 2) / 2D, so the V - 2 levels span [-D, D]:
    # uniform tokens, NLL log(V - 2), are the density 1 / 2D of the steps.
    assert f"{nll_x(math.log(256 - 2), grid):.4f}" == f"{math.log(2 * bound):.4f}"
