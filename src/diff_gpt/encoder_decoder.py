# Copyright 2025 Mikhail Borisov
"""
Derivative tokenizers: encode a (T, F) series as integer tokens and decode it back.

Each column is scaled so that its k-th differences fit the vocabulary and is then
tokenized by "lgt" (level-grid, the default) or "mash" (prefix-anchored Σ-Δ).
Values are float64 arrays or, for exact arithmetic, `Decimal` object arrays
(`use_decimal=True`).
"""

import math
import os
import warnings
from decimal import Decimal

import numpy as np

ufunc_decimal = np.frompyfunc(Decimal, 1, 1)


def np_to_decimal(inp: np.ndarray) -> np.ndarray:
    """Convert `inp` element-wise to `Decimal`; the result is always an object array."""
    result = ufunc_decimal(inp)
    if isinstance(result, Decimal):
        result = np.array(result)
    return result


def np_is_decimal(inp: np.ndarray) -> bool:
    """
    Return True if `inp` holds `Decimal` values.

    Only the first element is checked; an empty array counts if its dtype is
    object.
    """
    if inp.size == 0:
        return inp.dtype == object
    return isinstance(inp.flat[0], Decimal)


def np_is_float64(inp: np.ndarray) -> bool:
    """Return True if `inp` has dtype float64."""
    return inp.dtype == np.dtype("float64")


def _check_dtype(name: str, inp: np.ndarray, *, use_decimal: bool) -> None:
    """Check the dtype: Decimal mode takes Decimal object arrays, float mode float64."""
    if use_decimal and not np_is_decimal(inp):
        msg = f"{name} must hold Decimal values when use_decimal=True"
        raise TypeError(msg)
    if not use_decimal and not np_is_float64(inp):
        msg = f"{name} must be float64 when use_decimal=False, got {inp.dtype}"
        raise TypeError(msg)


def _check_finite(name: str, inp: np.ndarray, *, use_decimal: bool) -> None:
    """Reject NaN and infinities: they would turn into arbitrary tokens."""
    if use_decimal:
        finite = all(v.is_finite() for v in inp.flat)
    else:
        finite = bool(np.isfinite(inp).all())
    if not finite:
        msg = f"{name} must be finite, but it holds NaN or infinity"
        raise ValueError(msg)


# Smallest vocabulary the tokenizers accept (V must also be even).
_MIN_VOCAB_SIZE = 4


def _check_vocab_size(vocab_size: int) -> None:
    """Check that V is even and at least 4: tokens are centred on V/2."""
    if vocab_size % 2 != 0 or vocab_size < _MIN_VOCAB_SIZE:
        msg = f"vocab_size must be even and >= 4, got {vocab_size}"
        raise ValueError(msg)


def _check_domain(domain_of_definition: np.ndarray, n_features: int) -> None:
    """Check that the domain holds one bound per column."""
    if domain_of_definition.shape[0] != n_features:
        msg = (
            f"domain_of_definition has {domain_of_definition.shape[0]} entries "
            f"for {n_features} columns"
        )
        raise ValueError(msg)


ufunc_round = np.frompyfunc(round, 1, 1)


def _is_scalar_order(order_of_derivative: int | np.ndarray) -> bool:
    return isinstance(order_of_derivative, (int, np.integer))


def _check_order(order: np.ndarray, n_features: int) -> None:
    """Check that a per-column order is a non-negative array of shape (F,)."""
    if order.shape != (n_features,):
        msg = f"order_of_derivative shape {order.shape} must be ({n_features},)"
        raise ValueError(msg)
    if (order < 0).any():
        msg = f"order_of_derivative must be non-negative, got {order}"
        raise ValueError(msg)


# Which tokenizer `encode`/`decode` use when the caller does not say:
#   "mash": prefix-anchored Σ-Δ (rounding at k=0, a saturating carry scheme at
#           k=1, the MASH cascade for k>=2).
#   "lgt":  level-grid tokenization, q_t = Δ^k round(s·x_t) + V/2 (see `encode_lgt`).
# Default is "lgt" (since 0.32): same Δ/2 bound and headroom as MASH, but local,
# window-consistent and exactly decodable in integers. Set DIFF_GPT_TOKENIZER=mash
# (or pass tokenizer="mash") to reproduce models trained with the Σ-Δ encoder.
DEFAULT_TOKENIZER = os.environ.get("DIFF_GPT_TOKENIZER", "lgt")
_TOKENIZERS = ("lgt", "mash")


def resolve_tokenizer(tokenizer: str | None) -> str:
    """Return `tokenizer`, or `DEFAULT_TOKENIZER` for None; reject unknown names."""
    tok = DEFAULT_TOKENIZER if tokenizer is None else tokenizer
    if tok not in _TOKENIZERS:
        msg = f"unknown tokenizer {tok!r}; expected 'mash' or 'lgt'"
        raise ValueError(msg)
    return tok


class TokenClippingWarning(UserWarning):
    """Some k-th differences fell outside the domain and their tokens were clipped."""


def _clip_tokens(
    tokens: np.ndarray,
    vocab_size: int,
    *,
    n_saturated: int = 0,
) -> np.ndarray:
    """
    Clip `tokens` to [0, V-1] in place and warn with the number clipped.

    A clipped token stands for a step outside `domain_of_definition`; the
    excess is lost, so at k >= 1 every later decoded value is off by it.
    `n_saturated` steps were already cut by a saturating loop (MASH at
    k = 1), which makes their excess up later; the warning counts them too.
    """
    n_out = int(np.count_nonzero((tokens < 0) | (tokens >= vocab_size)))
    n_clipped = n_out + n_saturated
    if n_clipped > 0:
        msg = (
            f"{n_clipped} of {tokens.size} tokens fell outside [0, {vocab_size - 1}] "
            "and were clipped: the series steps outside domain_of_definition, and "
            "at k >= 1 each clipped step offsets the decoded values after it "
            "(MASH at k = 1 makes the offset up over the next steps)"
        )
        warnings.warn(msg, TokenClippingWarning, stacklevel=3)
    np.clip(tokens, 0, vocab_size - 1, out=tokens)
    return tokens


def _to_tokens(values: np.ndarray, vocab_size: int, *, use_decimal: bool) -> np.ndarray:
    """
    Convert integer-valued (N, F) token values to int64, before clipping.

    Values are first limited to [-1, V], so a token out of range stays out of
    range for `_clip_tokens` to count, without overflowing int64.
    """
    if use_decimal:
        return np.array(
            [[min(max(int(v), -1), vocab_size) for v in row] for row in values],
            dtype=np.int64,
        ).reshape(values.shape)
    return np.clip(np.rint(values), -1, vocab_size).astype(np.int64)


def _order_array(order_of_derivative: int | np.ndarray, n_features: int) -> np.ndarray:
    """Normalise a scalar or per-column order to an int64 array of shape (F,)."""
    if _is_scalar_order(order_of_derivative):
        k = int(order_of_derivative)
        if k < 0:
            msg = f"order_of_derivative must be non-negative, got {k}"
            raise ValueError(msg)
        return np.full(n_features, k, dtype=np.int64)
    order = np.asarray(order_of_derivative, dtype=np.int64)
    _check_order(order, n_features)
    return order


# Level-grid scale for columns whose training domain is D = 0 (constant column):
# a fine grid so the prefix level is reproduced to 2^-21 instead of being
# rounded on the unit grid. (The Σ-Δ path keeps scale 1 there and carries the
# level through `start`.)
LGT_CONSTANT_COLUMN_SCALE = float(2**20)

# Lowest derivative order that reserves 2^k headroom in the vocabulary: from
# k = 2 on, Σ-Δ switches to the MASH cascade (when V > 2^k + 2) and the level
# grid requires V > 2^k + 2; lower orders use the (V-2)/2 scale.
_MIN_HEADROOM_ORDER = 2


def _compute_scale(
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_arr: np.ndarray,
    *,
    use_decimal: bool,
    lgt: bool = False,
) -> np.ndarray:
    """
    Per-column scale s_c.

    Σ-Δ ("mash"): headroom 2^k when k>=2 and V > 2^k + 2 (MASH cascade), else
    (V-2)/2 (carry scheme, whose noise term needs only ±1).
    Level-grid ("lgt"): the noise term Δ^k ζ always needs headroom 2^(k-1) per
    side, so k>=2 requires V > 2^k + 2 (checked); k<=1 uses (V-2)/2.
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
                    Decimal(LGT_CONSTANT_COLUMN_SCALE)
                    if use_decimal
                    else LGT_CONSTANT_COLUMN_SCALE
                )
            continue
        if lgt and k_c >= _MIN_HEADROOM_ORDER and vocab_size <= 2**k_c + 2:
            msg = (
                f"level-grid tokenization needs vocab_size > 2^k + 2 = {2**k_c + 2} "
                f"for k={k_c} (got {vocab_size}); use tokenizer='mash' for smaller "
                "vocabularies"
            )
            raise ValueError(msg)
        use_headroom = k_c >= _MIN_HEADROOM_ORDER and vocab_size > 2**k_c + 2
        if use_headroom:
            scale[c] = (vocab_size - 2 - 2**k_c) / domain_of_definition[c] / 2
        else:
            scale[c] = (vocab_size - 2) / domain_of_definition[c] / 2
    return scale


# Levels must stay below 2^(62 - k) in magnitude: then the int64 levels and
# their k-th differences (at most 2^k times larger) cannot overflow.
_LEVEL_BITS = 62
# Fractional part from which `_round_levels` rounds up.
_ROUND_HALF = 0.5


def _round_levels(
    x_col: np.ndarray,
    s: float | Decimal,
    *,
    use_decimal: bool,
    order: int,
) -> np.ndarray:
    """
    L = floor(s·x + 1/2) as int64: round half up, in float64 and in Decimal.

    Half up rather than half to even keeps ζ = L - s·x in (-1/2, 1/2]. At
    k = 1 the scale bounds |s·Δx| by m = (V-2)/2, so the integer
    ΔL = s·Δx + Δζ lies strictly between -(m+1) and m+1, that is within ±m,
    and the token stays in [1, V-1]. With ties to even ζ reaches -1/2 as
    well, and a step of exactly D with a tie at both ends gives ΔL = m + 1:
    token V, which clips.

    Raises ValueError when |s·x| reaches 2^(62 - `order`), where the levels
    or their `order`-th differences would overflow int64.
    """
    u = x_col * s
    limit = 2 ** (_LEVEL_BITS - order)
    if use_decimal:
        peak = max((abs(v) for v in u), default=Decimal(0))
    else:
        peak = float(np.max(np.abs(u), initial=0.0))
    if peak >= limit:
        msg = (
            f"the level grid needs |s·x| < 2^{_LEVEL_BITS - order} at "
            f"order_of_derivative {order}, but scale s = {float(s):.3g} gives "
            f"{float(peak):.3g}: subtract the series level, or widen a "
            "near-zero domain_of_definition"
        )
        raise ValueError(msg)
    if use_decimal:
        half = Decimal("0.5")
        return np.array([math.floor(v + half) for v in u], dtype=np.int64)
    f = np.floor(u)
    # The test compares the true fractional part with 1/2: u - f is exact
    # (Sterbenz) except for -1/2 < u < 0, where it exceeds 1/2 either way.
    # floor(u + 0.5) would round the sum first, e.g. to 1 at u = 0.5 - 2^-54.
    return (f + (u - f >= _ROUND_HALF)).astype(np.int64)


def encode_lgt(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int | np.ndarray,
    *,
    use_decimal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Level-grid tokenization (LGT).

    Round the signal onto the absolute grid of step 1/s, L_t = round(s·x_t),
    and emit the k-th differences of the grid index: q_t = Δ^k L_t + V/2.

    Same scale and headroom as the MASH path, same bound ‖x̂ - x‖∞ ≤ 1/(2s),
    because Δ^k L = s·Δ^k x + Δ^k ζ with ζ = L - s·x. Levels round half up,
    so ζ lies in (-1/2, 1/2]: at k = 1 the integer ΔL stays within ±(V-2)/2
    even when a step of exactly D meets rounding ties, and at k >= 2
    |Δ^k ζ| < 2^(k-1) fits the headroom (see `_round_levels`). The headroom
    is mandatory: k >= 2 requires vocab_size > 2^k + 2 (there is no
    carry-scheme fallback here). Constant columns (D = 0) use a fine level
    grid (`LGT_CONSTANT_COLUMN_SCALE`) so their level survives the round trip.
    Unlike Σ-Δ the token at t depends only on x_{t-k..t}: encoding a window
    equals the slice of the full encoding, identical values give identical
    tokens anywhere in the series, and the decoder is exact integer cumsum.

    `start` is the raw prefix inp[:max_k] (the decoder re-rounds it). Token
    row r corresponds to raw time r + max_k, as in the Σ-Δ path. Tokens out of
    range are clipped with a `TokenClippingWarning`.
    """
    _check_vocab_size(vocab_size)
    T, F = inp.shape
    _check_domain(domain_of_definition, F)
    order_arr = _order_array(order_of_derivative, F)
    max_k = int(order_arr.max()) if F > 0 else 0
    if max_k > T:
        msg = f"order_of_derivative {max_k} exceeds the series length {T}"
        raise ValueError(msg)
    scale = _compute_scale(
        vocab_size,
        domain_of_definition,
        order_arr,
        use_decimal=use_decimal,
        lgt=True,
    )
    start = inp[:max_k].copy()
    N = T - max_k
    tokens = np.zeros((N, F), dtype=np.int64)
    for c in range(F):
        k_c = int(order_arr[c])
        L = _round_levels(inp[:, c], scale[c], use_decimal=use_decimal, order=k_c)
        q = np.diff(L, n=k_c)
        tokens[:, c] = q[len(q) - N :]
    tokens += vocab_size // 2
    return start, scale, _clip_tokens(tokens, vocab_size)


def decode_lgt(
    start: np.ndarray,
    scale: np.ndarray,
    inp: np.ndarray,
    vocab_size: int,
    order_of_derivative: int | np.ndarray,
    *,
    use_decimal: bool,
) -> np.ndarray:
    """Inverse of `encode_lgt`: exact integer k-fold cumsum, then one division."""
    _check_vocab_size(vocab_size)
    if inp.dtype != np.int64:
        msg = f"tokens must be int64, got {inp.dtype}"
        raise TypeError(msg)
    N, F = inp.shape
    order_arr = _order_array(order_of_derivative, F)
    max_k = start.shape[0]
    if (order_arr > max_k).any():
        msg = f"order_of_derivative {order_arr} exceeds the {max_k}-row start prefix"
        raise ValueError(msg)
    q = inp - vocab_size // 2
    T = N + max_k
    levels = np.zeros((T, F), dtype=np.int64)
    for c in range(F):
        k_c = int(order_arr[c])
        L_pre = (
            _round_levels(start[:, c], scale[c], use_decimal=use_decimal, order=k_c)
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
    *,
    use_decimal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-column k-th finite differences.

    `order_of_derivative` is either a scalar (applied to every column) or a
    1D array of shape (F,) giving per-column orders.

    Returns `start`, the raw prefix inp[:max_k], and `diff` of shape
    (T - max_k, F): column c holds the last T - max_k of its k_c-th
    differences, so row r of every column belongs to time r + max_k.
    `integrate` inverts this.
    """
    T, F = inp.shape
    order = _order_array(order_of_derivative, F)
    max_k = int(order.max()) if F > 0 else 0
    if max_k > T:
        msg = f"order_of_derivative {max_k} exceeds the series length {T}"
        raise ValueError(msg)
    _check_dtype("inp", inp, use_decimal=use_decimal)
    start = inp[:max_k].copy()
    N = T - max_k
    if use_decimal:
        diff = np_to_decimal(np.zeros((N, F), dtype=np.float64))
    else:
        diff = np.zeros((N, F), dtype=inp.dtype)
    for c in range(F):
        d = np.diff(inp[:, c], n=int(order[c]))
        # d has length T - k_c >= N; keep its last N values.
        diff[:, c] = d[len(d) - N :]
    return start, diff


def integrate(
    start: np.ndarray,
    diff: np.ndarray,
    order_of_derivative: int | np.ndarray,
    *,
    use_decimal: bool,
) -> np.ndarray:
    """
    Inverse of `differentiate`, for scalar or per-column orders.

    `start` is the raw prefix of the series, at least max_k rows; the result
    has `len(start) + len(diff)` rows.
    """
    N, F = diff.shape
    order = _order_array(order_of_derivative, F)
    max_k = start.shape[0]
    if (order > max_k).any():
        msg = f"order_of_derivative {order} exceeds the {max_k}-row start prefix"
        raise ValueError(msg)
    if start.shape[1:] != (F,):
        msg = f"start has shape {start.shape}, but diff has {F} columns"
        raise ValueError(msg)
    _check_dtype("start", start, use_decimal=use_decimal)
    _check_dtype("diff", diff, use_decimal=use_decimal)
    T = N + max_k
    if use_decimal:
        out = np_to_decimal(np.zeros((T, F), dtype=np.float64))
    else:
        out = np.zeros((T, F), dtype=diff.dtype)
    for c in range(F):
        k_c = int(order[c])
        # Derivative starts for column c come from its last k_c prefix values:
        # the i-th difference starts at (np.diff^i(raw_c))[0], i = 0..k_c-1.
        raw_c = start[max_k - k_c : max_k, c]
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
    *,
    use_decimal: bool,
) -> np.ndarray:
    """
    k-th-order MASH (cascade) Σ-Δ modulator.

    Each of the k stages is a stable 1st-order Σ-Δ loop; stages ≥ 2 operate
    on the previous stage's quantization error (|ε| ≤ 1/2). Outputs are
    combined with a digital differentiator network:

        q_t = q1_t + Δ q2_t + Δ² q3_t + ... + Δ^{k-1} qk_t,  Δ := (1 - z^{-1}).

    Algebra gives q_t - y_t = Δ^k ε_k_t with |ε_k| ≤ 1/2, so after k decoder
    integrations the residual is bounded uniformly in T. The cascade avoids
    the unconditional instability of direct high-order EFM at k ≥ 3.

    Vectorized over columns only: the feedback loop runs in Python over T·k
    (time, stage) steps, in float64 (`use_decimal=False`) and in Decimal
    (`use_decimal=True`, exact). Float64 does not hold the bound for every
    order and length: rounding in the decoder's k-fold cumsum grows roughly
    as T^(k-1)·eps, and at k = 4 it already shows from T of about 10^4. Use
    Decimal there.

    Returns unclipped tokens; `encode` clips them.
    """
    if k < 1:
        msg = f"the MASH cascade needs k >= 1, got {k}"
        raise ValueError(msg)
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
    # stage 0 receives y, stage i (>0) receives -ε of stage (i-1).
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

    # Combine stages: combined_t = Σ_{i=0}^{k-1} (1 - z^{-1})^i q_stage[i].
    combined = np.full((T, F), zero, dtype=dtype)
    for i in range(k):
        for j in range(i + 1):
            raw = ((-1) ** j) * math.comb(i, j)
            coef = Decimal(raw) if use_decimal else float(raw)
            if j == 0:
                combined = combined + coef * q_stage[i]
            else:
                combined[j:] = combined[j:] + coef * q_stage[i, : T - j]

    # Decimal values are already integer-valued (sums of ints).
    return _to_tokens(combined + half, vocab_size, use_decimal=use_decimal)


def _saturate_steps(steps: np.ndarray, vocab_size: int) -> tuple[np.ndarray, int]:
    """
    Fit the integer first-order `steps` (N, F') into the token range.

    A step past the range is cut to its edge, and the cut part is added to
    the next steps until their running sum is back at that of `steps`: the
    loop saturates inside, so a step outside the domain costs a few steps of
    lag instead of an offset in every later decoded value. Up to the first
    step out of range, and after each catch-up, the steps stay as they are.
    Returns the steps and the number of them that were out of range.
    """
    half = vocab_size // 2
    lo, hi = -half, vocab_size - 1 - half
    out_of_range = (steps < lo) | (steps > hi)
    n_out = int(np.count_nonzero(out_of_range))
    if n_out == 0:
        return steps, 0
    rows = np.flatnonzero(out_of_range.any(axis=1))
    fitted = steps.copy()
    lag = steps[0] * 0  # the part of the running sum still to make up
    t = int(rows[0])
    while t < len(steps):
        want = steps[t] + lag
        fitted[t] = np.minimum(np.maximum(want, lo), hi)
        lag = want - fitted[t]
        t += 1
        if not (lag != 0).any():
            # Caught up: the steps stay as they are until the next one out
            # of range.
            later = rows[rows >= t]
            if later.size == 0:
                break
            t = int(later[0])
    return fitted, n_out


def _encode_column_tokens(
    scaled_col: np.ndarray,
    k: int,
    vocab_size: int,
    *,
    use_decimal: bool,
) -> tuple[np.ndarray, int]:
    """
    Apply the appropriate Σ-Δ variant to the scaled diffs of columns of order `k`.

    - k = 0: plain quantization (round, shift).
    - k = 1: legacy carry-based 1st-order scheme, saturating inside the loop
      (`_saturate_steps`).
    - k ≥ 2 with insufficient V: the same scheme without saturation, since
      after k integrations a made-up step still leaves an offset.
    - k ≥ 2 with sufficient V: MASH cascade (tight uniform bound).

    Takes (N, F') and returns the (N, F') int64 tokens, unclipped, with the
    number of steps that the saturation cut (0 unless k = 1).
    """
    use_mash = k >= _MIN_HEADROOM_ORDER and vocab_size > 2**k + 2
    if k == 0:
        # No noise-shaping needed (no integration at decode).
        rounded = ufunc_round(scaled_col) if use_decimal else np.rint(scaled_col)
        tokens = _to_tokens(
            rounded + vocab_size // 2, vocab_size, use_decimal=use_decimal
        )
        return tokens, 0
    if use_mash:
        return _sigma_delta_k(scaled_col, k, vocab_size, use_decimal=use_decimal), 0
    # Legacy carry-based Σ-Δ: trunc(y) plus the change of the rounded running
    # remainder.
    trunced = np.trunc(scaled_col)
    residual = np.cumsum(scaled_col - trunced, axis=0)
    residual = ufunc_round(residual) if use_decimal else np.rint(residual)
    residual = np.concat(
        (np.zeros(shape=(1, residual.shape[1]), dtype=residual.dtype), residual),
        axis=0,
    )
    steps = trunced + np.diff(residual, axis=0)
    n_saturated = 0
    if k == 1:
        steps, n_saturated = _saturate_steps(steps, vocab_size)
    tokens = _to_tokens(steps + vocab_size // 2, vocab_size, use_decimal=use_decimal)
    return tokens, n_saturated


def encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int | np.ndarray,
    *,
    use_decimal: bool,
    tokenizer: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode multivariate signal to tokens.

    `order_of_derivative` can be scalar (uniform k for all columns) or a 1D
    array (per-column k). In the per-column case, each column's scale is
    sized by its own k (headroom for MASH if applicable).

    `tokenizer`: "mash" (prefix-anchored Σ-Δ, this function's body) or
    "lgt" (`encode_lgt`); None picks `DEFAULT_TOKENIZER`.

    Returns `start` (the raw prefix inp[:max_k]), the per-column `scale` and
    the (T - max_k, F) int64 tokens. NaN and infinities raise ValueError.
    Tokens of steps outside `domain_of_definition` are clipped to [0, V-1],
    with a `TokenClippingWarning` that gives their number. At k = 1 the MASH
    loop saturates instead and makes the cut part up over the next steps.
    """
    _check_vocab_size(vocab_size)
    F = inp.shape[1]
    _check_domain(domain_of_definition, F)
    _check_dtype("inp", inp, use_decimal=use_decimal)
    _check_dtype("domain_of_definition", domain_of_definition, use_decimal=use_decimal)
    _check_finite("inp", inp, use_decimal=use_decimal)
    _check_finite(
        "domain_of_definition",
        domain_of_definition,
        use_decimal=use_decimal,
    )
    if resolve_tokenizer(tokenizer) == "lgt":
        return encode_lgt(
            inp=inp,
            vocab_size=vocab_size,
            domain_of_definition=domain_of_definition,
            order_of_derivative=order_of_derivative,
            use_decimal=use_decimal,
        )
    order_arr = _order_array(order_of_derivative, F)
    scale = _compute_scale(
        vocab_size,
        domain_of_definition,
        order_arr,
        use_decimal=use_decimal,
    )

    start, diff = differentiate(
        inp=inp,
        order_of_derivative=order_arr,
        use_decimal=use_decimal,
    )
    scaled_diff = diff * scale

    # Columns of the same order go through the modulator together.
    tokens = np.zeros(scaled_diff.shape, dtype=np.int64)
    n_saturated = 0
    for k in np.unique(order_arr):
        cols = np.flatnonzero(order_arr == k)
        col_tokens, n_cut = _encode_column_tokens(
            scaled_diff[:, cols],
            int(k),
            vocab_size,
            use_decimal=use_decimal,
        )
        tokens[:, cols] = col_tokens
        n_saturated += n_cut
    return start, scale, _clip_tokens(tokens, vocab_size, n_saturated=n_saturated)


def decode(
    start: np.ndarray,
    scale: np.ndarray,
    inp: np.ndarray,
    vocab_size: int,
    order_of_derivative: int | np.ndarray,
    *,
    use_decimal: bool,
    tokenizer: str | None = None,
) -> np.ndarray:
    """
    Decode tokens back into the signal: the inverse of `encode`.

    `start` and `scale` are the values `encode` returned along with the tokens,
    and `tokenizer` must be the one they were encoded with ("mash" or "lgt");
    None picks `DEFAULT_TOKENIZER`. Any max_k consecutive raw rows can serve
    as `start`: the tokens then continue from them.
    """
    if resolve_tokenizer(tokenizer) == "lgt":
        return decode_lgt(
            start=start,
            scale=scale,
            inp=inp,
            vocab_size=vocab_size,
            order_of_derivative=order_of_derivative,
            use_decimal=use_decimal,
        )
    _check_vocab_size(vocab_size)
    _check_dtype("start", start, use_decimal=use_decimal)
    _check_dtype("scale", scale, use_decimal=use_decimal)
    if inp.dtype != np.int64:
        msg = f"tokens must be int64, got {inp.dtype}"
        raise TypeError(msg)
    inp_diff = (inp - vocab_size // 2) / scale
    return integrate(
        start=start,
        diff=inp_diff,
        order_of_derivative=order_of_derivative,
        use_decimal=use_decimal,
    )


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: int | np.ndarray,
    *,
    use_decimal: bool,
    quantile: float | None = None,
) -> np.ndarray:
    """
    Per-column bound D on |Δ^k x|, which sets the bin width Δ = 2D/(V-2).

    `quantile` (e.g. 0.9999) uses that quantile of |Δ^k x| instead of the max.
    The max is set by the single most extreme step in the whole training set, so
    on heavy-tailed collections it makes the grid useless for typical series: on
    M4 Daily the max is 19.5 while the median step is 0.02, and Δ/2 alone comes
    to 3.8 typical steps. A quantile trades a small fraction of clipped values
    (which the encoder clips with a `TokenClippingWarning`) for a grid that
    can actually represent ordinary moves.

    Raises ValueError when the series has no k-th difference (T <= max k) or
    holds NaN or infinities.
    """
    _check_dtype("inp", inp, use_decimal=use_decimal)
    _check_finite("inp", inp, use_decimal=use_decimal)
    if quantile is not None and use_decimal:
        msg = "quantile domain is float64-only"
        raise ValueError(msg)
    if quantile is not None and not 0.0 < quantile <= 1.0:
        msg = f"quantile {quantile} must be in (0, 1]"
        raise ValueError(msg)
    reduce = (
        (lambda a: np.max(a, axis=0))
        if quantile is None or quantile == 1.0
        else (lambda a: np.quantile(a, quantile, axis=0))
    )
    T, F = inp.shape
    order = _order_array(order_of_derivative, F)
    max_k = int(order.max()) if F > 0 else 0
    if max_k >= T:
        msg = (
            f"order_of_derivative {max_k} needs a series longer than {max_k} rows "
            f"to have a difference to bound, got {T}"
        )
        raise ValueError(msg)
    if use_decimal:
        result = np_to_decimal(np.zeros(F, dtype=np.float64))
    else:
        result = np.zeros(F, dtype=np.float64)
    for c in range(F):
        result[c] = reduce(np.abs(np.diff(inp[:, c], n=int(order[c]))))
    return result
