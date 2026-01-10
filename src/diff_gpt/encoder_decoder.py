from decimal import Decimal

import numpy as np

ufunc_decimal = np.frompyfunc(Decimal, 1, 1)


def np_to_decimal(inp: np.ndarray) -> np.ndarray:
    result = ufunc_decimal(inp)
    if isinstance(result, Decimal):
        result = np.array(result)
    return result


def np_is_decimal(inp: np.ndarray) -> bool:
    len_inp_shape = len(inp.shape)
    if len_inp_shape == 0 or inp.shape[0] == 0:
        return inp.dtype == "object"
    if len_inp_shape == 1:
        return isinstance(inp[0], Decimal)
    return isinstance(inp[0, 0], Decimal)


def np_is_float64(inp: np.ndarray) -> bool:
    return inp.dtype == np.dtype("float64")


ufunc_round = np.frompyfunc(round, 1, 1)


def differentiate(
    inp: np.ndarray,
    order_of_derivative: int,
    use_decimal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    assert order_of_derivative >= 0
    assert order_of_derivative <= inp.shape[0]
    if use_decimal:
        assert np_is_decimal(inp)
    else:
        assert np_is_float64(inp)
    diff = inp
    start = np.zeros((0, inp.shape[1]), dtype=inp.dtype)
    for _ in range(order_of_derivative):
        start = np.concat((start, np.expand_dims(diff[0], axis=0)), axis=0)
        diff = np.diff(diff, axis=0)
    if use_decimal:
        assert np_is_decimal(start)
        assert np_is_decimal(diff)
    else:
        assert np_is_float64(start)
        assert np_is_float64(diff)
    result = start, diff
    return result


def integrate(
    start: np.ndarray,
    diff: np.ndarray,
    order_of_derivative: int,
    use_decimal: bool,
) -> np.ndarray:
    assert order_of_derivative == start.shape[0]
    if use_decimal:
        assert np_is_decimal(start)
        assert np_is_decimal(diff)
    else:
        assert np_is_float64(start)
        assert np_is_float64(diff)
    result = diff
    for i in range(order_of_derivative):
        _start = start[-i - 1]
        result = np.concat((np.expand_dims(_start, axis=0), result), axis=0)
        result = np.cumsum(result, axis=0)
    if use_decimal:
        assert np_is_decimal(result)
    else:
        assert np_is_float64(result)
    return result


def encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int,
    use_decimal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert vocab_size % 2 == 0
    assert vocab_size >= 4
    assert domain_of_definition.shape[0] == inp.shape[1]
    assert order_of_derivative >= 0
    if use_decimal:
        assert np_is_decimal(inp)
        assert np_is_decimal(domain_of_definition)

    else:
        assert np_is_float64(inp)
        assert np_is_float64(domain_of_definition)
    scale = np.ones(shape=domain_of_definition.shape, dtype=np.float64)
    if use_decimal:
        scale = np_to_decimal(scale)
    _filter = np.abs(domain_of_definition) > np.finfo(np.float64).eps
    _scale = (vocab_size - 2) / domain_of_definition[_filter] / 2
    scale[_filter] = _scale
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    scaled_diff = diff * scale
    trunced_scaled_diff = np.trunc(scaled_diff)
    residual = scaled_diff - trunced_scaled_diff
    residual = np.cumsum(residual, axis=0)
    residual = ufunc_round(residual)
    residual = np.concat(
        (
            np.zeros(shape=(1, residual.shape[1]), dtype=np.int64),
            residual,
        ),
        axis=0,
    )
    residual = np.diff(residual, axis=0)
    encoded_data = trunced_scaled_diff + vocab_size // 2 + residual
    encoded_data[encoded_data < 0] = 0
    encoded_data[encoded_data > vocab_size] = vocab_size - 1
    result = start, scale, encoded_data.astype(np.int64)
    return result


def decode(
    start: np.ndarray,
    scale: np.ndarray,
    inp: np.ndarray,
    vocab_size: int,
    order_of_derivative: int,
    use_decimal: bool,
) -> np.ndarray:
    assert vocab_size % 2 == 0
    assert vocab_size >= 4
    assert order_of_derivative >= 0
    if use_decimal:
        assert np_is_decimal(start)
        assert np_is_decimal(scale)
    else:
        assert np_is_float64(start)
        assert np_is_float64(scale)
    assert inp.dtype == np.int64
    inp = (inp - vocab_size // 2) / scale
    result = integrate(
        start=start,
        diff=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    return result


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: int,
    use_decimal: bool,
) -> np.ndarray:
    assert order_of_derivative >= 0
    if use_decimal:
        assert np_is_decimal(inp)
    else:
        assert np_is_float64(inp)
    diff = np.diff(inp, n=order_of_derivative, axis=0)
    result = np.max(np.abs(diff), axis=0)
    return result
