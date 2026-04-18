import math
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


def _sigma_delta_k(
    y: np.ndarray,
    k: int,
    vocab_size: int,
    use_decimal: bool,
) -> np.ndarray:
    """k-th-order MASH (cascade) Σ-Δ modulator.

    Each of the k stages is a stable 1st-order Σ-Δ loop; stages ≥ 2 operate
    on the previous stage's quantization error (|ε| ≤ 1/2). Outputs are
    combined with a digital differentiator network:

        q_t = q1_t + Δ q2_t + Δ² q3_t + ... + Δ^{k-1} qk_t,  Δ := (1 - z^{-1}).

    Algebra gives q_t - y_t = Δ^k ε_k_t with |ε_k| ≤ 1/2, so after k decoder
    integrations the residual is bounded uniformly in T. The cascade avoids
    the unconditional instability of direct high-order EFM at k ≥ 3.

    Supports both float64 (`use_decimal=False`, fast vectorized) and Decimal
    (`use_decimal=True`, slower but arbitrary precision — needed for k ≥ 5
    where float64's 53-bit mantissa loses bits faster than noise-shaping
    recovers them in the k-fold decoder cumsum).
    """
    assert k >= 1
    T, F = y.shape
    half = vocab_size // 2

    if use_decimal:
        dtype: type = object
        zero = Decimal(0)
        round_op = ufunc_round
    else:
        dtype = np.float64
        zero = 0.0
        round_op = np.rint

    # Run k cascaded 1st-order Σ-Δ stages. q_stage[i] holds integer-valued
    # outputs of stage i (stored in `dtype` — float64 or Decimal-as-object);
    # stage 0 receives y, stage i (>0) receives −ε of stage (i−1).
    q_stage = np.full((k, T, F), zero, dtype=dtype)
    eps_prev = np.full((k, F), zero, dtype=dtype)
    for t in range(T):
        stage_input = y[t]
        for i in range(k):
            y_eff = stage_input - eps_prev[i]
            q_rounded = round_op(y_eff)
            q_stage[i, t] = q_rounded
            eps_cur = q_rounded - y_eff
            eps_prev[i] = eps_cur
            stage_input = -eps_cur

    # Combine stages: combined_t = Σ_{i=0}^{k-1} (1 − z^{-1})^i q_stage[i].
    combined = np.full((T, F), zero, dtype=dtype)
    for i in range(k):
        for j in range(i + 1):
            raw = ((-1) ** j) * math.comb(i, j)
            coef = Decimal(raw) if use_decimal else float(raw)
            if j == 0:
                combined = combined + coef * q_stage[i]
            else:
                combined[j:] = combined[j:] + coef * q_stage[i, : T - j]

    if use_decimal:
        # Decimal values are already integer-valued (sum of ints); just cast.
        tokens = np.array(
            [[int(v) for v in row] for row in combined], dtype=np.int64
        )
    else:
        tokens = np.rint(combined).astype(np.int64)
    tokens = tokens + half
    np.clip(tokens, 0, vocab_size - 1, out=tokens)
    return tokens


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
    # MASH (k ≥ 2) produces combined = y + Δ^k ε_k with |Δ^k ε_k| ≤ 2^{k-1}.
    # To keep combined in the quantizer range we need |y| ≤ (V − 2 − 2^k)/2;
    # when V is too small for that, fall back to the 1st-order carry scheme
    # (which drifts as T^{k-1} at k ≥ 2 but at least fits in the vocabulary).
    use_mash = order_of_derivative >= 2 and vocab_size > 2 ** order_of_derivative + 2
    if use_mash:
        headroom = 2 ** order_of_derivative
        _scale = (vocab_size - 2 - headroom) / domain_of_definition[_filter] / 2
    else:
        _scale = (vocab_size - 2) / domain_of_definition[_filter] / 2
    scale[_filter] = _scale
    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    scaled_diff = diff * scale
    if not use_mash:
        # k ∈ {0, 1} or V too small: use the original carry-based 1st-order
        # Σ-Δ. Mathematically equivalent to the MASH k=1 case at k=1, and the
        # only option when V ≤ 2^k + 2 (no headroom for MASH corrections).
        trunced_scaled_diff = np.trunc(scaled_diff)
        residual = scaled_diff - trunced_scaled_diff
        residual = np.cumsum(residual, axis=0)
        residual = ufunc_round(residual) if use_decimal else np.rint(residual)
        residual = np.concat(
            (
                np.zeros(shape=(1, residual.shape[1]), dtype=residual.dtype),
                residual,
            ),
            axis=0,
        )
        residual = np.diff(residual, axis=0)
        encoded_data = trunced_scaled_diff + vocab_size // 2 + residual
        encoded_data[encoded_data < 0] = 0
        encoded_data[encoded_data >= vocab_size] = vocab_size - 1
        result = start, scale, encoded_data.astype(np.int64)
    else:
        # k ≥ 2: use MASH cascade so that after k decoder integrations the
        # reconstruction error stays uniformly bounded in T (1st-order Σ-Δ
        # would drift as T^{k-1}). Supports both float64 (fast) and Decimal
        # (arbitrary precision — required for k ≥ 5 where float64 cumsum
        # roundoff swamps the Δ/2 bound).
        encoded_data = _sigma_delta_k(
            scaled_diff, order_of_derivative, vocab_size, use_decimal
        )
        result = start, scale, encoded_data
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
