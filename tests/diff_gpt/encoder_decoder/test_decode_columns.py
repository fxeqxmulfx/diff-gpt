"""
`decode` takes the columns of one operator together, each as it would alone.

The reference decodes one column at a time, with the helpers on 1-D arrays,
as `decode` did before it took the columns of an operator side by side: the
values must come back bit for bit.
"""

import numpy as np
import pytest

from diff_gpt.encoder_decoder import (
    Difference,
    Grid,
    _anchor_fraction,
    _invert_difference,
    _levels,
    column_lags,
    decode,
    encode,
    get_domain_of_definition,
)

V = 256
ORDERS = {
    "0": 0,
    "1": 1,
    "2": 2,
    "seasonal": Difference(1, 24),
    "mixed": np.array(
        [0, 1, 2, Difference(1, 24), 1, 2, Difference(24), 0],
        dtype=object,
    ),
}


def _column_by_column(
    start: np.ndarray,
    grid: Grid,
    tokens: np.ndarray,
    order: object,
    tokenizer: str,
) -> np.ndarray:
    """Decode `tokens` one column at a time, the helpers on 1-D arrays."""
    lags = column_lags(order, tokens.shape[1])
    max_p, (n, f) = start.shape[0], tokens.shape
    steps = tokens - grid.zero
    out = np.empty((max_p + n, f))
    out[:max_p] = start
    for c, column in enumerate(lags):
        p = sum(column)
        scale = grid.scale[c]
        prefix = start[max_p - p :, c]
        fraction, fraction_steps = _anchor_fraction(
            prefix * scale,
            n,
            lags=column,
            tokenizer=tokenizer,
        )
        levels = _invert_difference(
            _levels(prefix, scale, fraction[:p], factors=len(column)),
            steps[:, c] - fraction_steps,
            column,
        )
        out[max_p:, c] = (levels[p:] + fraction[p:]) / scale
    return out


def _series(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    """Random walks of many scales and levels, and a constant column."""
    scales = 10.0 ** rng.uniform(-3, 4, size=columns)
    walks = np.cumsum(rng.normal(size=(rows, columns)), axis=0)
    series = (walks + 50 * rng.normal(size=columns)) * scales
    series[:, 3] = 7.25
    return series


@pytest.mark.parametrize("tokenizer", ["mash", "lgt", "mash-half"])
@pytest.mark.parametrize("order", list(ORDERS))
def test_the_columns_of_an_operator_decode_together_as_each_alone(
    tokenizer: str,
    order: str,
) -> None:
    """Tokens drawn anywhere in the vocabulary decode as column by column."""
    rng = np.random.default_rng(14)
    series = _series(rng, 200, 8)
    domain = get_domain_of_definition(series, ORDERS[order])
    start, grid, _ = encode(series, V, domain, ORDERS[order], tokenizer=tokenizer)
    drawn = rng.integers(0, V, size=(300, 8))
    want = _column_by_column(start, grid, drawn, ORDERS[order], tokenizer)
    got = decode(start, grid, drawn, V, ORDERS[order], tokenizer=tokenizer)
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("tokenizer", ["mash", "lgt", "mash-half"])
def test_the_samples_of_a_context_decode_together_as_each_alone(tokenizer: str) -> None:
    """
    The samples of one context, as DiffGPT.predict_samples_many decodes them.

    The context's start rows and grid repeat for each of the 100 samples,
    whose tokens differ: the values match those decoded one sample at a time.
    """
    rng = np.random.default_rng(15)
    series = _series(rng, 120, 4)[:, :1]
    domain = get_domain_of_definition(series, 1)
    start, grid, _ = encode(series, V, domain, 1, tokenizer=tokenizer)
    samples = 100
    tiled = Grid(scale=np.tile(grid.scale, samples), zero=np.tile(grid.zero, samples))
    drawn = rng.integers(0, V, size=(64, samples))
    got = decode(np.tile(start, (1, samples)), tiled, drawn, V, 1, tokenizer=tokenizer)
    for s in range(samples):
        alone = _column_by_column(start, grid, drawn[:, [s]], 1, tokenizer)
        assert got[:, s].tobytes() == alone[:, 0].tobytes()
