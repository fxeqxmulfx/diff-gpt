import math
import os
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


# Which tokenizer `encode`/`decode` use when the caller does not say:
#   "mash": prefix-anchored Σ-Δ (carry scheme for k<=1, MASH cascade for k>=2).
#   "lgt":  level-grid tokenization, q_t = Δ^k round(s·x_t) + V/2 (see `encode_lgt`).
# Default is "lgt" (since 0.32): same Δ/2 bound and headroom as MASH, but local,
# window-consistent and exactly decodable in integers. Set DIFF_GPT_TOKENIZER=mash
# (or pass tokenizer="mash") to reproduce models trained with the Σ-Δ encoder.
DEFAULT_TOKENIZER = os.environ.get("DIFF_GPT_TOKENIZER", "lgt")


def _order_array(order_of_derivative: int | np.ndarray, n_features: int) -> np.ndarray:
    """Normalise a scalar or per-column order to an int64 array of shape (F,)."""
    if _is_scalar_order(order_of_derivative):
        k = int(order_of_derivative)
        assert k >= 0
        return np.full(n_features, k, dtype=np.int64)
    order = np.asarray(order_of_derivative, dtype=np.int64)
    assert order.shape == (n_features,), (
        f"order_of_derivative shape {order.shape} must be ({n_features},)"
    )
    assert (order >= 0).all()
    return order


# Level-grid scale for columns whose training domain is D = 0 (constant column):
# a fine grid so the prefix level is reproduced to 2^-21 instead of being
# rounded on the unit grid. (The Σ-Δ path keeps scale 1 there and carries the
# level through `start`.)
LGT_CONSTANT_COLUMN_SCALE = float(2 ** 20)


def _compute_scale(
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_arr: np.ndarray,
    use_decimal: bool,
    lgt: bool = False,
) -> np.ndarray:
    """Per-column scale s_c.

    Σ-Δ ("mash"): headroom 2^k when k>=2 and V > 2^k + 2 (MASH cascade), else
    (V-2)/2 (carry scheme, whose noise term needs only ±1).
    Level-grid ("lgt"): the noise term Δ^k ζ always needs headroom 2^(k-1) per
    side, so k>=2 requires V > 2^k + 2 (asserted); k<=1 uses (V-2)/2.
    """
    scale = np.ones(shape=domain_of_definition.shape, dtype=np.float64)
    if use_decimal:
        scale = np_to_decimal(scale)
    _filter = np.abs(domain_of_definition) > np.finfo(np.float64).eps
    for c in range(domain_of_definition.shape[0]):
        k_c = int(order_arr[c])
        if not _filter[c]:
            if lgt:
                scale[c] = (
                    Decimal(LGT_CONSTANT_COLUMN_SCALE) if use_decimal else LGT_CONSTANT_COLUMN_SCALE
                )
            continue
        if lgt and k_c >= 2:
            assert vocab_size > 2 ** k_c + 2, (
                f"level-grid tokenization needs vocab_size > 2^k + 2 = {2 ** k_c + 2} "
                f"for k={k_c} (got {vocab_size}); use tokenizer='mash' for smaller vocabularies"
            )
        use_headroom = k_c >= 2 and vocab_size > 2 ** k_c + 2
        if use_headroom:
            scale[c] = (vocab_size - 2 - 2 ** k_c) / domain_of_definition[c] / 2
        else:
            scale[c] = (vocab_size - 2) / domain_of_definition[c] / 2
    return scale


def _round_levels(x_col: np.ndarray, s, use_decimal: bool) -> np.ndarray:
    """L = round(s·x) as int64 (banker's rounding in both float64 and Decimal)."""
    if use_decimal:
        vals = ufunc_round(x_col * s)
        return np.array([int(v) for v in vals], dtype=np.int64)
    return np.rint(x_col * s).astype(np.int64)


def encode_lgt(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int | np.ndarray,
    use_decimal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Level-grid tokenization (LGT).

    Round the signal onto the absolute grid of step 1/s, L_t = round(s·x_t),
    and emit the k-th differences of the grid index: q_t = Δ^k L_t + V/2.

    Same scale and headroom as the MASH path, same bound ‖x̂ − x‖∞ ≤ 1/(2s),
    because Δ^k L = s·Δ^k x + Δ^k ζ with |ζ| ≤ 1/2 and |Δ^k ζ| ≤ 2^(k−1).
    The headroom is mandatory: k >= 2 requires vocab_size > 2^k + 2 (there is
    no carry-scheme fallback here). Constant columns (D = 0) use a fine level
    grid (`LGT_CONSTANT_COLUMN_SCALE`) so their level survives the round trip.
    Unlike Σ-Δ the token at t depends only on x_{t−k..t}: encoding a window
    equals the slice of the full encoding, identical values give identical
    tokens anywhere in the series, and the decoder is exact integer cumsum.

    `start` is the raw prefix inp[:max_k] for both scalar and per-column
    orders (the decoder re-rounds it). Token row r corresponds to raw time
    r + max_k, as in the Σ-Δ path.
    """
    assert vocab_size % 2 == 0 and vocab_size >= 4
    T, F = inp.shape
    assert domain_of_definition.shape[0] == F
    order_arr = _order_array(order_of_derivative, F)
    max_k = int(order_arr.max()) if F > 0 else 0
    assert max_k <= T
    scale = _compute_scale(vocab_size, domain_of_definition, order_arr, use_decimal, lgt=True)
    start = inp[:max_k].copy()
    N = T - max_k
    tokens = np.zeros((N, F), dtype=np.int64)
    for c in range(F):
        k_c = int(order_arr[c])
        L = _round_levels(inp[:, c], scale[c], use_decimal)
        q = np.diff(L, n=k_c) if k_c > 0 else L
        tokens[:, c] = q[-N:] if N > 0 else q[:0]
    tokens += vocab_size // 2
    np.clip(tokens, 0, vocab_size - 1, out=tokens)
    return start, scale, tokens


def decode_lgt(
    start: np.ndarray,
    scale: np.ndarray,
    inp: np.ndarray,
    vocab_size: int,
    order_of_derivative: int | np.ndarray,
    use_decimal: bool,
) -> np.ndarray:
    """Inverse of `encode_lgt`: exact integer k-fold cumsum, then one division."""
    assert vocab_size % 2 == 0 and vocab_size >= 4
    assert inp.dtype == np.int64
    N, F = inp.shape
    order_arr = _order_array(order_of_derivative, F)
    max_k = start.shape[0]
    assert (order_arr <= max_k).all()
    q = inp - vocab_size // 2
    T = N + max_k
    levels = np.zeros((T, F), dtype=np.int64)
    for c in range(F):
        k_c = int(order_arr[c])
        L_pre = (
            _round_levels(start[:, c], scale[c], use_decimal)
            if max_k > 0
            else np.zeros(0, dtype=np.int64)
        )
        # Integer derivative starts from the last k_c prefix levels.
        d = L_pre[max_k - k_c : max_k]
        der_starts = []
        for _ in range(k_c):
            der_starts.append(int(d[0]))
            d = np.diff(d)
        res = q[:, c]
        for i in range(k_c):
            res = np.cumsum(np.concatenate([[der_starts[k_c - 1 - i]], res]))
        if max_k > k_c:
            res = np.concatenate([L_pre[: max_k - k_c], res])
        levels[:, c] = res
    if use_decimal:
        out = np.empty((T, F), dtype=object)
        for c in range(F):
            out[:, c] = [Decimal(int(v)) / scale[c] for v in levels[:, c]]
        return out
    return levels / scale


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
    tokenizer: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode multivariate signal to tokens.

    `order_of_derivative` can be scalar (uniform k for all columns) or a 1D
    array (per-column k). In the per-column case, each column's scale is
    sized by its own k (headroom for MASH if applicable).

    `tokenizer`: "mash" (prefix-anchored Σ-Δ, this function's body) or
    "lgt" (`encode_lgt`); None picks `DEFAULT_TOKENIZER`.
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
    tok = tokenizer or DEFAULT_TOKENIZER
    if tok == "lgt":
        return encode_lgt(
            inp=inp,
            vocab_size=vocab_size,
            domain_of_definition=domain_of_definition,
            order_of_derivative=order_of_derivative,
            use_decimal=use_decimal,
        )
    if tok != "mash":
        raise ValueError(f"unknown tokenizer {tok!r}; expected 'mash' or 'lgt'")
    order_arr = _order_array(order_of_derivative, F)
    scale = _compute_scale(vocab_size, domain_of_definition, order_arr, use_decimal)

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
    tokenizer: str | None = None,
) -> np.ndarray:
    tok = tokenizer or DEFAULT_TOKENIZER
    if tok == "lgt":
        return decode_lgt(
            start=start,
            scale=scale,
            inp=inp,
            vocab_size=vocab_size,
            order_of_derivative=order_of_derivative,
            use_decimal=use_decimal,
        )
    if tok != "mash":
        raise ValueError(f"unknown tokenizer {tok!r}; expected 'mash' or 'lgt'")
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
    quantile: float | None = None,
) -> np.ndarray:
    """Per-column bound D on |Δ^k x|, which sets the bin width Δ = 2D/(V−2).

    `quantile` (e.g. 0.9999) uses that quantile of |Δ^k x| instead of the max.
    The max is set by the single most extreme step in the whole training set, so
    on heavy-tailed collections it makes the grid useless for typical series: on
    M4 Daily the max is 19.5 while the median step is 0.02, and Δ/2 alone comes
    to 3.8 typical steps. A quantile trades a small fraction of clipped values
    (which the encoder saturates, see the clipping note in the docs) for a grid
    that can actually represent ordinary moves.
    """
    if use_decimal:
        assert np_is_decimal(inp)
        assert quantile is None, "quantile domain is float64-only"
    else:
        assert np_is_float64(inp)
    assert quantile is None or 0.0 < quantile <= 1.0
    reduce = (
        (lambda a: np.max(a, axis=0))
        if quantile is None or quantile == 1.0
        else (lambda a: np.quantile(a, quantile, axis=0))
    )
    if _is_scalar_order(order_of_derivative):
        k = int(order_of_derivative)
        assert k >= 0
        diff = np.diff(inp, n=k, axis=0)
        return reduce(np.abs(diff))
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
            result[c] = reduce(np.abs(diff_c))
    return result
