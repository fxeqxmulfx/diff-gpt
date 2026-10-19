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
The rounding to levels is exact, the levels stay below 2^53, where float64
holds every one of them, and no int64 sum overflows. Only s·x and the
decode are rounded: about 2^-53·|s·x| levels each, which add to the bounds
above, so a token can reach V and clip only where a step of nearly D has
all its k + 1 levels at rounding ties (lean/DiffGPT/Machine.lean).
"""

import os
import warnings

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
# trained on other tokens are refused. Version 2 puts both tokenizers on one
# level grid for every k; before it "mash" ran a Σ-Δ cascade (a carry scheme
# at k <= 1) and both used the scale (V - 2 - 2^k) / (2D) from k = 2 on.
ENCODER_VERSION = 2


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

    Then s·D = V/2 - 2^(k-1), or V/2 - 1 at k = 0. Levels round half up, so
    ζ = L - s·(x - B) lies in (-1/2, 1/2] and |Δ^k ζ| < 2^(k-1) for k >= 1:
    the integer Δ^k L = s·Δ^k x + Δ^k ζ stays within V/2 - 1, and every
    token lies in [1, V-1] (`abs_fwdDiff_iter_levels_le` in
    lean/DiffGPT/LevelGrid.lean). Columns with D = 0 get
    `CONSTANT_COLUMN_SCALE`. Raises ValueError when V <= max(2, 2^k), where
    no scale leaves room for the noise.
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
        if abs(domain_of_definition[c]) > np.finfo(np.float64).eps:
            scale[c] = (vocab_size - headroom) / domain_of_definition[c] / 2
        else:
            scale[c] = CONSTANT_COLUMN_SCALE
    return scale


# Values rounded onto the levels at order k must stay below 2^(62 - k) in
# magnitude: then the int64 levels and their differences up to the k-th, at
# most 2^k times larger, cannot overflow.
_LEVEL_BITS = 62
# float64 holds the integers below 2^53, and so the fractions n / 2^G,
# 0 <= n < 2^G, exactly for G <= 53. The values rounded onto the levels must
# stay below 2^53 as well: past it fl(s·x) misses whole levels, and that
# error alone, 2^k times larger in the k-th differences, pushes tokens out
# of the vocabulary.
_FLOAT64_EXACT_BITS = 53
# Fractional part from which `_round_levels` rounds up.
_ROUND_HALF = 0.5


def _round_levels(u: np.ndarray, *, order: int) -> np.ndarray:
    """
    Round the scaled values `u` half up onto the int64 levels: floor(u + 1/2).

    Half up rather than half to even keeps ζ = L - u in (-1/2, 1/2],
    half-open, which is what gives every token room in [1, V-1] (see
    `_compute_scale`). With ties to even ζ reaches -1/2 as well, and at
    k = 1 a step of exactly D with a tie at both ends gives the token V,
    which clips.

    Raises ValueError when |u| reaches 2^min(53, 62 - `order`): past 2^53
    float64 no longer holds every level, and past 2^(62 - `order`) the levels
    or their `order`-th differences would overflow int64.
    """
    bits = min(_FLOAT64_EXACT_BITS, _LEVEL_BITS - order)
    peak = float(np.max(np.abs(u), initial=0.0))
    if peak >= 2.0**bits:
        msg = (
            f"the level grid needs |s·x| < 2^{bits} at order_of_derivative "
            f"{order}, but it reaches {peak:.3g}: subtract the series level or "
            "widen a near-zero domain_of_definition"
        )
        raise ValueError(msg)
    f = np.floor(u)
    # The test compares the true fractional part with 1/2: u - f is exact
    # (Sterbenz) except for -1/2 < u < 0, where it exceeds 1/2 either way.
    # floor(u + 0.5) would round the sum first, e.g. to 1 at u = 0.5 - 2^-54.
    return (f + (u - f >= _ROUND_HALF)).astype(np.int64)


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
        # The column's k prefix rows, then its N token rows, on the level scale.
        scaled = inp[max_k - k :, c] * scale[c]
        fraction, fraction_steps = _anchor_fraction(scaled[:k], N, tokenizer=tok)
        levels = _round_levels(scaled - fraction, order=k)
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
        scaled_prefix = start[max_k - k :, c] * scale[c]
        fraction, fraction_steps = _anchor_fraction(scaled_prefix, N, tokenizer=tok)
        levels = _integrate_levels(
            _round_levels(scaled_prefix - fraction[:k], order=k),
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


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: int | np.ndarray,
    *,
    quantile: float | None = None,
) -> np.ndarray:
    """
    Per-column bound D on |Δ^k x|, which sets the bin width Δ = 2D/(V - max(2, 2^k)).

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
    result = np.zeros(F, dtype=np.float64)
    for c in range(F):
        result[c] = reduce(np.abs(np.diff(inp[:, c], n=int(order[c]))))
    return result
