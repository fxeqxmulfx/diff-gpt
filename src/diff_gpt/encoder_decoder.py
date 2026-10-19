"""
Derivative tokenizers: encode a (T, F) series as integer tokens and decode it back.

Both tokenizers put each column on a grid of step 1/s, measured in levels
from an anchor B, and emit the k-th differences of the levels, in the same
way for every order k:

    L_t = floor(s·x_t - B_t + 1/2),    q_t = Δ^k L_t + V/2,
    s = (V - max(2, 2^k)) / (2D),      D >= max |Δ^k x_t|.

The anchor is a polynomial of degree < k, which the k-th difference removes
(Δ^k B = 0), and it is all that differs between them:

- "lgt" (level grid, the default): B = 0, the absolute grid. The token at t
  depends only on x_{t-k..t}, so encoding a window gives the matching slice
  of the full encoding.
- "mash" (prefix-anchored): B passes through the k prefix values of s·x, so
  the grid moves with the prefix, as in a MASH Σ-Δ cascade started there.

Every token lies in [1, V-1], and the decoder, an exact k-fold cumulative
sum of the levels followed by x̂ = (B + L)/s, stays within
1/(2s) = D/(V - max(2, 2^k)) of the series, uniformly in T (README
Theorem 1, proved in lean/DiffGPT/LevelGrid.lean).

The arithmetic is float64 and int64. B enters only through its fractional
part F_t, a multiple of 2^-G with G <= 53 that float64 holds exactly: with
R_t = floor(s·x_t - F_t + 1/2) the levels are L_t = R_t - floor(B_t), so
the tokens are Δ^k R + Δ^k F + V/2 (Δ^k F = -Δ^k floor(B) is an integer)
and x̂ = (R + F)/s. B itself, which grows like t^(k-1), is never formed.

The levels are exact: R is read off the float64 rounding of s·x and its
exact error (Dekker's product), without rounding s·x itself
(lean/DiffGPT/ExactLevels.lean). The scale is rounded down, so that
s·D <= V/2 - 2^(k-1) holds exactly, and the max of
`get_domain_of_definition` is rounded up, so that |Δ^k x| <= D does: every
token then lies in [1, V-1] as in exact arithmetic, the levels stay within
2^53, where float64 holds every one of them, and no int64 sum overflows
(lean/DiffGPT/Machine.lean). Only the decode is rounded, by about 2^-53 of
|x̂| per operation on top of 1/(2s).
"""

import math
import os
import warnings
from fractions import Fraction

import numpy as np


def _check_dtype(name: str, inp: np.ndarray) -> None:
    """Check that `inp` is float64."""
    if inp.dtype != np.float64:
        msg = f"{name} must be float64, got {inp.dtype}"
        raise TypeError(msg)


def _check_finite(name: str, inp: np.ndarray) -> None:
    """Reject NaN and infinities: they would turn into arbitrary tokens."""
    if not np.isfinite(inp).all():
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


# Which tokenizer `encode`/`decode` use when the caller does not say: "lgt"
# (the absolute grid) or "mash" (the grid anchored at the prefix), see the
# module docstring. Set DIFF_GPT_TOKENIZER=mash, or pass tokenizer="mash", for
# the latter.
DEFAULT_TOKENIZER = os.environ.get("DIFF_GPT_TOKENIZER", "lgt")
_TOKENIZERS = ("lgt", "mash")

# Version of the token scheme. DiffGPT checkpoints record it, so that weights
# trained on other tokens are refused. Version 3 computes the levels of s·x
# exactly and rounds the scale down; version 2 rounded s·x first, which moved
# levels at rounding ties. Version 2 put both tokenizers on one level grid for
# every k; before it "mash" ran a Σ-Δ cascade (a carry scheme at k <= 1) and
# both used the scale (V - 2 - 2^k) / (2D) from k = 2 on.
ENCODER_VERSION = 3


def resolve_tokenizer(tokenizer: str | None) -> str:
    """Return `tokenizer`, or `DEFAULT_TOKENIZER` for None; reject unknown names."""
    tok = DEFAULT_TOKENIZER if tokenizer is None else tokenizer
    if tok not in _TOKENIZERS:
        msg = f"unknown tokenizer {tok!r}; expected 'mash' or 'lgt'"
        raise ValueError(msg)
    return tok


class TokenClippingWarning(UserWarning):
    """Some k-th differences fell outside the domain and their tokens were clipped."""


def _clip_tokens(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Clip `tokens` to [0, V-1] in place and warn with the number clipped.

    A clipped token stands for a step outside `domain_of_definition`; the
    excess is lost, so at k >= 1 every later decoded value is off by it.
    """
    n_clipped = int(np.count_nonzero((tokens < 0) | (tokens >= vocab_size)))
    if n_clipped > 0:
        msg = (
            f"{n_clipped} of {tokens.size} tokens fell outside [0, {vocab_size - 1}] "
            "and were clipped: the series steps outside domain_of_definition, and "
            "at k >= 1 each clipped step offsets the decoded values after it"
        )
        warnings.warn(msg, TokenClippingWarning, stacklevel=3)
    np.clip(tokens, 0, vocab_size - 1, out=tokens)
    return tokens


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


# Scale of a column whose domain is D = 0, one that is a polynomial of degree
# < k: a fine grid, so that "lgt" reproduces its level to 2^-21 instead of
# rounding it on the unit grid.
CONSTANT_COLUMN_SCALE = float(2**20)


def _headroom(k: int) -> int:
    """Levels the scale leaves for the rounding noise Δ^k ζ: max(2, 2^k)."""
    return max(2, 2**k)


def _compute_scale(
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_arr: np.ndarray,
) -> np.ndarray:
    """
    Per-column scale s = (V - max(2, 2^k)) / (2D), the same for both tokenizers.

    s is rounded down, so that s·D <= V/2 - 2^(k-1), or V/2 - 1 at k = 0,
    holds exactly. Levels round half up, so ζ = L - s·(x - B) lies in
    (-1/2, 1/2] and |Δ^k ζ| < 2^(k-1) for k >= 1: the integer
    Δ^k L = s·Δ^k x + Δ^k ζ stays within V/2 - 1, and every token lies in
    [1, V-1] (`abs_fwdDiff_iter_levels_le` in lean/DiffGPT/LevelGrid.lean).
    Columns with D = 0 get `CONSTANT_COLUMN_SCALE`. Raises ValueError when
    V <= max(2, 2^k), where no scale leaves room for the noise.
    """
    scale = np.empty(domain_of_definition.shape, dtype=np.float64)
    for c in range(domain_of_definition.shape[0]):
        k = int(order_arr[c])
        headroom = _headroom(k)
        if vocab_size <= headroom:
            msg = (
                f"order_of_derivative {k} needs vocab_size > max(2, 2^k) = "
                f"{headroom}, got {vocab_size}"
            )
            raise ValueError(msg)
        domain = domain_of_definition[c]
        if abs(domain) > np.finfo(np.float64).eps:
            s = (vocab_size - headroom) / domain / 2
            # The quotient rounds to nearest, so it may land above (V - h)/D.
            while Fraction(s) * Fraction(domain) > Fraction(vocab_size - headroom, 2):
                s = np.nextafter(s, 0.0)
            scale[c] = s
        else:
            scale[c] = CONSTANT_COLUMN_SCALE
    return scale


# Values put on the levels at order k must stay below 2^(62 - k) in
# magnitude: then the int64 levels and their differences up to the k-th, at
# most 2^k times larger, cannot overflow.
_LEVEL_BITS = 62
# float64 holds the integers below 2^53, and so the fractions n / 2^G,
# 0 <= n < 2^G, exactly for G <= 53. s·x must stay below 2^53 as well: past
# it float64 no longer holds every level, so neither `_levels` nor the
# decoder can.
_FLOAT64_EXACT_BITS = 53
# `_levels` raises the exponent of s·x to at least this. A product below 2^-53
# in magnitude stays there, where its level depends on its sign alone (F - 1/2
# is 0 or at least 2^-53 away from it), and the parts of the product stay
# clear of float64's subnormals, where np.ldexp would round them.
_MIN_EXPONENT = -_FLOAT64_EXACT_BITS
# Levels round half up.
_HALF = 0.5
# `_split` cuts a float64 mantissa after this many bits.
_SPLIT_BITS = 26


def _split(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the mantissas 1/2 <= |m| < 1, or 0, exactly into m = hi + lo.

    hi = rint(m·2^26)/2^26 is H·2^-26 and lo = m - hi is L·2^-53, with
    integers |H|, |L| <= 2^26, so the product of two halves is an integer of
    at most 2^52 times a power of two: float64 holds it exactly.
    """
    hi = np.ldexp(np.rint(np.ldexp(m, _SPLIT_BITS)), -_SPLIT_BITS)
    return hi, m - hi


def _two_product(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return p = fl(a·b) and its error e = a·b - p, exactly (Dekker's product).

    For mantissas 1/2 <= |a|, |b| < 1, or 0, as np.frexp returns them. The
    products of their halves (`_split`) are exact, and so is each sum below:
    every partial sum is a multiple of the finer of its terms' grids, small
    enough for float64 to hold.
    """
    p = a * b
    a_hi, a_lo = _split(a)
    b_hi, b_lo = _split(b)
    e = (((a_hi * b_hi - p) + a_hi * b_lo) + a_lo * b_hi) + a_lo * b_lo
    return p, e


def _levels(
    x: np.ndarray,
    scale: float,
    fraction: np.ndarray,
    *,
    order: int,
) -> np.ndarray:
    """
    Return the int64 levels R = floor(s·x - F + 1/2) of the floats `x`, exactly.

    s = `scale`, and F = `fraction` holds multiples of 2^-53 in [0, 1) (see
    `_anchor_fraction`), so c = F - 1/2 is exact. With np.frexp,
    s·x = 2^E·m_s·m_x, and `_two_product` splits m_s·m_x into its rounding
    p0 and the exact error e0; then p = 2^E·p0 is the float64 rounding of
    s·x, e = 2^E·e0 its error, and both are exact (see `_MIN_EXPONENT`):

    - |p| >= 1/2: n = rint(p) and w = (p - n) - c are exact, w lies in
      (-1, 1] and |e| <= 1/2, so R = n + floor(w + e) is n - 2 plus the
      number of j in {-1, 0, 1} with (w - j) + e >= 0. That sum of two
      floats rounds to its sign: w - j is exact where |w - j| < 1, and
      decides the sign alone elsewhere.
    - |p| < 1/2: R is 0 where s·x >= c and -1 elsewhere. A float c lies
      between s·x and its rounding p only if p = c, so p decides, or the
      sign of e where p = c.

    Raises ValueError when |p| reaches 2^min(53, 62 - `order`): past 2^53
    float64 no longer holds every level, and past 2^(62 - `order`) the levels
    or their `order`-th differences would overflow int64. Below it the
    levels lie in [-2^b - 1, 2^b] for b = min(53, 62 - `order`)
    (`levels_mem` in lean/DiffGPT/ExactLevels.lean).
    """
    bits = min(_FLOAT64_EXACT_BITS, _LEVEL_BITS - order)
    s_mantissa, s_exponent = np.frexp(scale)
    x_mantissa, x_exponent = np.frexp(x)
    p, e = _two_product(s_mantissa, x_mantissa)
    exponent = np.maximum(s_exponent + x_exponent, _MIN_EXPONENT)
    with np.errstate(over="ignore"):
        p = np.ldexp(p, exponent)
    peak = float(np.max(np.abs(p), initial=0.0))
    if peak >= 2.0**bits:
        msg = (
            f"the level grid needs |s·x| < 2^{bits} at order_of_derivative "
            f"{order}, but it reaches {peak:.3g}: subtract the series level or "
            "widen a near-zero domain_of_definition"
        )
        raise ValueError(msg)
    e = np.ldexp(e, exponent)
    offset = fraction - _HALF
    n = np.rint(p)
    w = (p - n) - offset
    above = sum(((w - j) + e >= 0).astype(np.int64) for j in (-1, 0, 1))
    small = ((p > offset) | ((p == offset) & (e >= 0))).astype(np.int64) - 1
    return np.where(np.abs(p) >= _HALF, n.astype(np.int64) - 2 + above, small)


# Bits of int64 and uint64 values.
_WORD_BITS = 64


def _fraction_bits(k: int) -> int:
    """
    Fraction bits G of the "mash" anchor at order k: min(53, 64 - k).

    Its fractional parts n / 2^G are exact in float64, and the k-th
    differences of their numerators, below 2^(k+G-1) in magnitude, fit int64.
    """
    return min(_FLOAT64_EXACT_BITS, _WORD_BITS - k)


def _anchor_fraction(
    scaled_prefix: np.ndarray,
    n_rows: int,
    *,
    tokenizer: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the fractional part F of the anchor B and its k-th differences.

    Returns F on the k rows of `scaled_prefix` and the `n_rows` rows after
    them, and the int64 Δ^k F on the latter. "lgt": B = 0. "mash": B is the
    polynomial of degree < k through the prefix values rounded to G fraction
    bits (`_fraction_bits`). Then F_t = (P_t mod 2^G) / 2^G for the integers
    P = 2^G·B, and P mod 2^G depends only on the prefix numerators mod 2^G:
    P is continued with Δ^k P = 0 in uint64, whose sums wrap modulo 2^64, a
    multiple of 2^G. Δ^k F = -Δ^k floor(B) is an integer.
    """
    k = scaled_prefix.shape[0]
    if tokenizer == "lgt" or k == 0:
        return np.zeros(k + n_rows), np.zeros(n_rows, dtype=np.int64)
    bits = _fraction_bits(k)
    fraction = scaled_prefix - np.floor(scaled_prefix)
    seeds = np.rint(np.ldexp(fraction, bits)).astype(np.uint64)
    anchor = _integrate_levels(seeds, np.zeros(n_rows, dtype=np.uint64))
    numerators = (anchor & np.uint64(2**bits - 1)).astype(np.int64)
    fraction_steps = np.diff(numerators, n=k) >> bits
    return np.ldexp(numerators.astype(np.float64), -bits), fraction_steps


def _cumsum(values: np.ndarray) -> np.ndarray:
    """
    np.cumsum: modulo 2^64 for uint64, exact for int64 or ValueError.

    An int64 partial sum overflows exactly when its two terms share a sign
    that the wrapped sum lacks; the first overflow is caught so, before any
    wrapped value enters a later test.
    """
    total = np.cumsum(values)
    if values.dtype == np.int64 and total.size > 1:
        before, terms, after = total[:-1], values[1:], total[1:]
        if (((before ^ terms) >= 0) & ((before ^ after) < 0)).any():
            msg = (
                "the tokens integrate to levels outside int64: the decoded "
                "series leaves the grid's range"
            )
            raise ValueError(msg)
    return total


def _integrate_levels(prefix_levels: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """
    Invert the k-th difference in integers, exactly.

    Returns the levels that start with the k `prefix_levels` and have the
    k-th differences `steps`; no other levels do (`eq_of_fwdDiff_iter_eq`).
    """
    firsts = []
    d = prefix_levels
    for _ in range(prefix_levels.shape[0]):
        firsts.append(d[:1])
        d = np.diff(d)
    levels = steps
    for first in reversed(firsts):
        levels = _cumsum(np.concatenate([first, levels]))
    return levels


def encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: int | np.ndarray,
    *,
    tokenizer: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode a float64 (T, F) series as the tokens q_t = Δ^k L_t + V/2.

    `order_of_derivative` is one k for every column or an array of shape
    (F,) with one per column; each column gets the scale of its own k (see
    `_compute_scale`). `tokenizer` is "lgt" or "mash", the anchor of the
    grid (see the module docstring); None picks `DEFAULT_TOKENIZER`.

    Returns `start`, the raw prefix inp[:max_k], the per-column `scale` and
    the (T - max_k, F) int64 tokens, whose row r belongs to time r + max_k.
    NaN and infinities raise ValueError. Tokens of steps outside
    `domain_of_definition` are clipped to [0, V-1], with a
    `TokenClippingWarning` that gives their number.
    """
    tok = resolve_tokenizer(tokenizer)
    _check_vocab_size(vocab_size)
    T, F = inp.shape
    _check_domain(domain_of_definition, F)
    _check_dtype("inp", inp)
    _check_dtype("domain_of_definition", domain_of_definition)
    _check_finite("inp", inp)
    _check_finite("domain_of_definition", domain_of_definition)
    order_arr = _order_array(order_of_derivative, F)
    max_k = int(order_arr.max()) if F > 0 else 0
    if max_k > T:
        msg = f"order_of_derivative {max_k} exceeds the series length {T}"
        raise ValueError(msg)
    scale = _compute_scale(vocab_size, domain_of_definition, order_arr)
    N = T - max_k
    tokens = np.zeros((N, F), dtype=np.int64)
    for c in range(F):
        k = int(order_arr[c])
        # The column's k prefix rows, then its N token rows.
        column = inp[max_k - k :, c]
        fraction, fraction_steps = _anchor_fraction(
            column[:k] * scale[c],
            N,
            tokenizer=tok,
        )
        levels = _levels(column, scale[c], fraction, order=k)
        # Unclipped, but limited to [-1, V] so that one out of range stays out
        # of range for `_clip_tokens` to count.
        tokens[:, c] = np.clip(
            np.diff(levels, n=k) + fraction_steps + vocab_size // 2,
            -1,
            vocab_size,
        )
    return inp[:max_k].copy(), scale, _clip_tokens(tokens, vocab_size)


def decode(
    start: np.ndarray,
    scale: np.ndarray,
    inp: np.ndarray,
    vocab_size: int,
    order_of_derivative: int | np.ndarray,
    *,
    tokenizer: str | None = None,
) -> np.ndarray:
    """
    Decode tokens back into the series: the inverse of `encode`.

    `start` and `scale` are the values `encode` returned along with the
    tokens, and `tokenizer` must be the one they were encoded with; None picks
    `DEFAULT_TOKENIZER`. Any max_k consecutive raw rows can serve as `start`:
    the tokens then continue from them.

    The `start` rows come back as they are. The levels of the others come
    back exactly, by an int64 k-fold cumulative sum from those of the
    prefix, and x̂ = (B + L)/s lies within 1/(2s) of the series where no
    token was clipped. Tokens whose levels leave int64, which `encode`
    never emits, raise ValueError.
    """
    tok = resolve_tokenizer(tokenizer)
    _check_vocab_size(vocab_size)
    _check_dtype("start", start)
    _check_dtype("scale", scale)
    if inp.dtype != np.int64:
        msg = f"tokens must be int64, got {inp.dtype}"
        raise TypeError(msg)
    N, F = inp.shape
    order_arr = _order_array(order_of_derivative, F)
    max_k = start.shape[0]
    if (order_arr > max_k).any():
        msg = f"order_of_derivative {order_arr} exceeds the {max_k}-row start prefix"
        raise ValueError(msg)
    steps = inp - vocab_size // 2
    out = np.empty((max_k + N, F), dtype=np.float64)
    out[:max_k] = start
    for c in range(F):
        k = int(order_arr[c])
        prefix = start[max_k - k :, c]
        fraction, fraction_steps = _anchor_fraction(
            prefix * scale[c],
            N,
            tokenizer=tok,
        )
        levels = _integrate_levels(
            _levels(prefix, scale[c], fraction[:k], order=k),
            steps[:, c] - fraction_steps,
        )
        out[max_k:, c] = (levels[k:] + fraction[k:]) / scale[c]
    return out


def differentiate(
    inp: np.ndarray,
    order_of_derivative: int | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-column k-th finite differences of a float64 (T, F) series.

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
    _check_dtype("inp", inp)
    start = inp[:max_k].copy()
    N = T - max_k
    diff = np.zeros((N, F), dtype=np.float64)
    for c in range(F):
        d = np.diff(inp[:, c], n=int(order[c]))
        # d has length T - k_c >= N; keep its last N values.
        diff[:, c] = d[len(d) - N :]
    return start, diff


def integrate(
    start: np.ndarray,
    diff: np.ndarray,
    order_of_derivative: int | np.ndarray,
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
    _check_dtype("start", start)
    _check_dtype("diff", diff)
    out = np.zeros((N + max_k, F), dtype=np.float64)
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
            result_c = np.cumsum(np.concat([[der_starts[k_c - 1 - i]], result_c]))
        # result_c has length N + k_c; pad to T with the raw prefix head.
        if max_k > k_c:
            result_c = np.concat([start[: max_k - k_c, c], result_c])
        out[:, c] = result_c
    return out


def _float_at_or_above(q: Fraction) -> float:
    """Return the least float64 at or above the rational `q` (inf past the range)."""
    try:
        f = float(q)
    except OverflowError:
        return math.inf
    if Fraction(f) < q:
        f = math.nextafter(f, math.inf)
    return f


def _max_abs_difference(column: np.ndarray, k: int) -> float:
    """
    Return the least float64 at or above max |Δ^k x| over `column`, exactly.

    np.diff rounds every difference, so its max can fall short of the exact
    one, and a step of the exact max would then reach a token past V - 1.
    The rounding error of each difference is exact (Knuth's two-sum), so
    their sums bound how far each k-th difference is off: where the bound
    is 0 the difference is exact, and only the rows whose bound reaches the
    largest value known to be attained are summed in rationals.
    """
    diff = column
    slack = np.zeros_like(column)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(k):
            a, b = diff[1:], -diff[:-1]
            diff = a + b
            b_rounded = diff - a
            error = (a - (diff - b_rounded)) + (b - b_rounded)
            slack = slack[1:] + slack[:-1] + np.abs(error)
        size = np.abs(diff)
        exact = slack == 0
        # The float sums of the slack fall short of the exact sums by a factor
        # (1 - 2^-53)^(2k) at most, which 2·slack covers; np.nextafter covers
        # the rounding of the bounds themselves.
        lower = np.nextafter(size - 2 * slack, -np.inf)
        upper = np.nextafter(size + 2 * slack, np.inf)
    best = Fraction(float(np.max(size[exact], initial=0.0)))
    attained = np.where(exact | np.isnan(lower), -np.inf, lower)
    threshold = max(float(best), float(np.max(attained, initial=-np.inf)))
    # A NaN bound, from an overflow, never compares below the threshold.
    rows = np.flatnonzero(~exact & ~(upper < threshold))
    weights = [(-1) ** (k - j) * math.comb(k, j) for j in range(k + 1)]
    for t in rows:
        value = abs(sum(w * Fraction(column[t + j]) for j, w in enumerate(weights)))
        best = max(best, value)
    return _float_at_or_above(best)


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: int | np.ndarray,
    *,
    quantile: float | None = None,
) -> np.ndarray:
    """
    Per-column bound D on |Δ^k x|, which sets the bin width Δ = 2D/(V - max(2, 2^k)).

    The max is exact: the least float64 at or above max |Δ^k x|, with the
    differences of the floats in `inp` taken without rounding, so that
    `encode` clips none of these steps.

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
    _check_dtype("inp", inp)
    _check_finite("inp", inp)
    if quantile is not None and not 0.0 < quantile <= 1.0:
        msg = f"quantile {quantile} must be in (0, 1]"
        raise ValueError(msg)
    T, F = inp.shape
    order = _order_array(order_of_derivative, F)
    max_k = int(order.max()) if F > 0 else 0
    if max_k >= T:
        msg = (
            f"order_of_derivative {max_k} needs a series longer than {max_k} rows "
            f"to have a difference to bound, got {T}"
        )
        raise ValueError(msg)
    result = np.zeros(F, dtype=np.float64)
    for c in range(F):
        k = int(order[c])
        if quantile is None or quantile == 1.0:
            result[c] = _max_abs_difference(inp[:, c], k)
        else:
            result[c] = np.quantile(np.abs(np.diff(inp[:, c], n=k)), quantile)
    return result
