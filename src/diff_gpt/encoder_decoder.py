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


def _is_scalar_order(order_of_derivative: int | np.ndarray) -> bool:
    return isinstance(order_of_derivative, (int, np.integer))


def differentiate(
    inp: np.ndarray,
    order_of_derivative: int | np.ndarray,
    use_decimal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column k-th finite differences.

    `order_of_derivative` is either a scalar (applied uniformly) or a 1D
    array of shape (F,) giving per-column orders.

    When all columns share the same k, returns the classical form:
    `start` is the k-row stack of each order's first value,
    `diff` is the uniformly differenced (T - k, F) array.

    When orders differ, returns:
    `start` as the raw prefix inp[:max_k] (length max_k),
    `diff` with per-column trimming so all columns share length T - max_k.
    (`integrate` detects which form it received and inverts accordingly.)
    """
    if _is_scalar_order(order_of_derivative):
        k = int(order_of_derivative)
        assert k >= 0
        assert k <= inp.shape[0]
        if use_decimal:
            assert np_is_decimal(inp)
        else:
            assert np_is_float64(inp)
        diff = inp
        start = np.zeros((0, inp.shape[1]), dtype=inp.dtype)
        for _ in range(k):
            start = np.concat((start, np.expand_dims(diff[0], axis=0)), axis=0)
            diff = np.diff(diff, axis=0)
        return start, diff

    order = np.asarray(order_of_derivative, dtype=np.int64)
    assert order.ndim == 1 and order.shape[0] == inp.shape[1]
    assert (order >= 0).all()
    max_k = int(order.max())
    assert max_k <= inp.shape[0]
    if use_decimal:
        assert np_is_decimal(inp)
    else:
        assert np_is_float64(inp)
    # Raw prefix of length max_k is the universal start for the array path.
    start = inp[:max_k].copy()
    F = inp.shape[1]
    N = inp.shape[0] - max_k
    if use_decimal:
        diff = np_to_decimal(np.zeros((N, F), dtype=np.float64))
    else:
        diff = np.zeros((N, F), dtype=inp.dtype)
    for c in range(F):
        k_c = int(order[c])
        d = inp[:, c]
        for _ in range(k_c):
            d = np.diff(d)
        # d has length T - k_c ≥ N. Take last N values.
        diff[:, c] = d[-N:] if N > 0 else d[:0]
    return start, diff


def integrate(
    start: np.ndarray,
    diff: np.ndarray,
    order_of_derivative: int | np.ndarray,
    use_decimal: bool,
) -> np.ndarray:
    """Inverse of `differentiate`, supporting scalar or per-column orders."""
    if _is_scalar_order(order_of_derivative):
        k = int(order_of_derivative)
        assert k == start.shape[0]
        if use_decimal:
            assert np_is_decimal(start)
            assert np_is_decimal(diff)
        else:
            assert np_is_float64(start)
            assert np_is_float64(diff)
        result = diff
        for i in range(k):
            _start = start[-i - 1]
            result = np.concat((np.expand_dims(_start, axis=0), result), axis=0)
            result = np.cumsum(result, axis=0)
        return result

    order = np.asarray(order_of_derivative, dtype=np.int64)
    assert order.ndim == 1 and order.shape[0] == diff.shape[1]
    max_k = start.shape[0]
    assert (order <= max_k).all()
    F = diff.shape[1]
    N = diff.shape[0]
    T = N + max_k
    if use_decimal:
        out = np_to_decimal(np.zeros((T, F), dtype=np.float64))
    else:
        out = np.zeros((T, F), dtype=diff.dtype)
    for c in range(F):
        k_c = int(order[c])
        # Derivative starts for column c are derived on the fly from the raw
        # prefix. raw_c[0..k_c] suffices: (i)-th-diff first value =
        # (np.diff^i(raw_c))[0] for i = 0, ..., k_c-1.
        raw_c = start[max_k - k_c : max_k, c] if k_c > 0 else start[:0, c]
        der_starts = []
        d = raw_c
        for _ in range(k_c):
            der_starts.append(d[0])
            d = np.diff(d)
        result_c = diff[:, c]
        for i in range(k_c):
            _start_val = der_starts[k_c - 1 - i]
            _start_arr = np.array([_start_val], dtype=result_c.dtype)
            result_c = np.concat([_start_arr, result_c])
            result_c = np.cumsum(result_c)
        # result_c has length N + k_c; pad to T with the raw prefix head.
        if max_k > k_c:
            result_c = np.concat([start[: max_k - k_c, c], result_c])
        out[:, c] = result_c
    return out


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


def _encode_column_tokens(
    scaled_col: np.ndarray,
    k: int,
    vocab_size: int,
    use_decimal: bool,
) -> np.ndarray:
    """Apply the appropriate Σ-Δ variant to one column's scaled diffs.

    - k = 0: plain quantization (round, shift, clip).
    - k = 1 or insufficient V: legacy carry-based 1st-order scheme.
    - k ≥ 2 with sufficient V: MASH cascade (tight uniform bound).

    Returns shape (N, 1) int64 token array.
    """
    N = scaled_col.shape[0]
    use_mash = k >= 2 and vocab_size > 2 ** k + 2
    if k == 0:
        # No noise-shaping needed (no integration at decode).
        rounded = ufunc_round(scaled_col) if use_decimal else np.rint(scaled_col)
        if use_decimal:
            tokens = np.array(
                [[int(v) for v in row] for row in rounded], dtype=np.int64
            )
        else:
            tokens = rounded.astype(np.int64)
        tokens = tokens + vocab_size // 2
        np.clip(tokens, 0, vocab_size - 1, out=tokens)
        return tokens
    if not use_mash:
        # Legacy carry-based Σ-Δ.
        trunced = np.trunc(scaled_col)
        residual = scaled_col - trunced
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
        encoded = trunced + vocab_size // 2 + residual
        encoded[encoded < 0] = 0
        encoded[encoded >= vocab_size] = vocab_size - 1
        return encoded.astype(np.int64)
    return _sigma_delta_k(scaled_col, k, vocab_size, use_decimal)


def encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int | np.ndarray,
    use_decimal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode multivariate signal to tokens.

    `order_of_derivative` can be scalar (uniform k for all columns) or a 1D
    array (per-column k). In the per-column case, each column's scale is
    sized by its own k (headroom for MASH if applicable).
    """
    assert vocab_size % 2 == 0
    assert vocab_size >= 4
    assert domain_of_definition.shape[0] == inp.shape[1]
    F = inp.shape[1]
    if use_decimal:
        assert np_is_decimal(inp)
        assert np_is_decimal(domain_of_definition)
    else:
        assert np_is_float64(inp)
        assert np_is_float64(domain_of_definition)
    # Normalize order to per-column array for uniform scale computation.
    if _is_scalar_order(order_of_derivative):
        scalar_k = int(order_of_derivative)
        assert scalar_k >= 0
        order_arr = np.full(F, scalar_k, dtype=np.int64)
    else:
        order_arr = np.asarray(order_of_derivative, dtype=np.int64)
        assert order_arr.shape == (F,)
        assert (order_arr >= 0).all()

    scale = np.ones(shape=domain_of_definition.shape, dtype=np.float64)
    if use_decimal:
        scale = np_to_decimal(scale)
    _filter = np.abs(domain_of_definition) > np.finfo(np.float64).eps
    # Per-column scale: for MASH columns include headroom 2^{k_c}, else use
    # the standard (V − 2)/2 formula.
    for c in range(F):
        if not _filter[c]:
            continue
        k_c = int(order_arr[c])
        use_mash_c = k_c >= 2 and vocab_size > 2 ** k_c + 2
        if use_mash_c:
            scale[c] = (vocab_size - 2 - 2 ** k_c) / domain_of_definition[c] / 2
        else:
            scale[c] = (vocab_size - 2) / domain_of_definition[c] / 2

    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    scaled_diff = diff * scale

    if _is_scalar_order(order_of_derivative):
        # Same k for all columns → existing vectorized path.
        scalar_k = int(order_of_derivative)
        use_mash = scalar_k >= 2 and vocab_size > 2 ** scalar_k + 2
        if not use_mash:
            trunced = np.trunc(scaled_diff)
            residual = scaled_diff - trunced
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
            encoded_data = trunced + vocab_size // 2 + residual
            encoded_data[encoded_data < 0] = 0
            encoded_data[encoded_data >= vocab_size] = vocab_size - 1
            encoded_data = encoded_data.astype(np.int64)
        else:
            encoded_data = _sigma_delta_k(
                scaled_diff, scalar_k, vocab_size, use_decimal
            )
        return start, scale, encoded_data

    # Per-column path.
    N = scaled_diff.shape[0]
    encoded_data = np.zeros((N, F), dtype=np.int64)
    for c in range(F):
        k_c = int(order_arr[c])
        col_tokens = _encode_column_tokens(
            scaled_diff[:, c : c + 1], k_c, vocab_size, use_decimal
        )
        encoded_data[:, c] = col_tokens[:, 0]
    return start, scale, encoded_data


def decode(
    start: np.ndarray,
    scale: np.ndarray,
    inp: np.ndarray,
    vocab_size: int,
    order_of_derivative: int | np.ndarray,
    use_decimal: bool,
) -> np.ndarray:
    assert vocab_size % 2 == 0
    assert vocab_size >= 4
    if use_decimal:
        assert np_is_decimal(start)
        assert np_is_decimal(scale)
    else:
        assert np_is_float64(start)
        assert np_is_float64(scale)
    assert inp.dtype == np.int64
    inp_diff = (inp - vocab_size // 2) / scale
    result = integrate(
        start=start,
        diff=inp_diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )
    return result


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: int | np.ndarray,
    use_decimal: bool,
) -> np.ndarray:
    if use_decimal:
        assert np_is_decimal(inp)
    else:
        assert np_is_float64(inp)
    if _is_scalar_order(order_of_derivative):
        k = int(order_of_derivative)
        assert k >= 0
        diff = np.diff(inp, n=k, axis=0)
        return np.max(np.abs(diff), axis=0)
    order = np.asarray(order_of_derivative, dtype=np.int64)
    assert order.ndim == 1 and order.shape[0] == inp.shape[1]
    assert (order >= 0).all()
    F = inp.shape[1]
    if use_decimal:
        result = np_to_decimal(np.zeros(F, dtype=np.float64))
    else:
        result = np.zeros(F, dtype=np.float64)
    for c in range(F):
        k_c = int(order[c])
        diff_c = np.diff(inp[:, c], n=k_c)
        if len(diff_c) > 0:
            result[c] = np.max(np.abs(diff_c))
    return result
