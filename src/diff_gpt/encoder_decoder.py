"""
Derivative tokenizers: encode a (T, F) series as integer tokens and decode it back.

Both tokenizers put each column on a grid of step 1/s, measured in levels
from an anchor A, and emit the levels differenced by an integer operator
c(B) = 1 + c_1 B + ... + c_p B^p, where B is the backshift, B x_t = x_{t-1}:

    L_t = floor(s·x_t - A_t + 1/2),    q_t = c(B) L_t + V/2,
    s = (V - ‖c‖₁) / (2D),             D >= max |c(B) x_t|.

`order_of_derivative` sets c per column, as a product of differences
(1 - B^S): the order k is Δ^k = (1 - B)^k, `Difference(24)` the seasonal
difference x_t - x_{t-24} of hourly data with a daily period, and
`Difference(1, 24)` both, (1 - B)(1 - B^24), as in SARIMA. The scale leaves
‖c‖₁ = Σ|c_j| levels, 2^k for Δ^k, for the rounding noise c(B)ζ.

The anchor lies in the kernel of c (c(B) A = 0), and it is all that differs
between the tokenizers:

- "lgt" (level grid): A = 0, the absolute grid. The token at t depends only
  on x_{t-p..t}, so encoding a window gives the matching slice of the full
  encoding.
- "mash" (prefix-anchored, the default): A passes through the p prefix
  values of s·x, so the grid moves with the prefix, as in a MASH Σ-Δ
  cascade started there.

Every token lies in [1, V-1], and the decoder, which inverts c on the levels
exactly and returns x̂ = (A + L)/s, stays within 1/(2s) = D/(V - ‖c‖₁) of
the series, uniformly in T (README Theorem 1, proved in
lean/DiffGPT/LevelGrid.lean).

The order 0, c = 1, needs no room for the noise, and both tokenizers take
A = 1/2 there instead: with s·D < V/2 the levels L = floor(s·x) take all V
tokens [0, V-1], and x̂ = (L + 1/2)/s, the middle of the cell
[L, L + 1)/s, stays within 1/(2s), a hair above D/V, the least that V
levels allow on [-D, D].

The arithmetic is float64 and int64. A enters only through its fractional
part F_t, a multiple of 2^-G with G <= 53 that float64 holds exactly: with
R_t = floor(s·x_t - F_t + 1/2) the levels are L_t = R_t - floor(A_t), so
the tokens are c(B) R - c(B) floor(A) + V/2 and x̂ = (R + F)/s. As
c(B) A = 0, the integer c(B) floor(A) is -c(B) F, which F gives; at the
order 0 floor(A) = 0. A itself, which grows like t^(k-1) at Δ^k, is never
formed.

The levels are exact: R is read off the float64 rounding of s·x and its
exact error (Dekker's product), without rounding s·x itself
(lean/DiffGPT/ExactLevels.lean). The scale is rounded down, so that
s·D <= (V - ‖c‖₁)/2 holds exactly, and the max of
`get_domain_of_definition` is rounded up, so that |c(B) x| <= D does: every
token then lies in its range as in exact arithmetic, the levels stay within
2^53, where float64 holds every one of them, and `decode` gets them back
without an int64 overflow (`encode_decode` and `encode_decode_zero` in
lean/DiffGPT/Machine.lean). Only the decode is rounded, by about 2^-53 of
|x̂| per operation on top of 1/(2s).
"""

import math
import os
import warnings
from collections.abc import Sequence
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


class Difference:
    """
    The difference operator c(B) = (1 - B^S_1)···(1 - B^S_m) at the lags S_i >= 1.

    B is the backshift, so (1 - B^S) x_t = x_t - x_{t-S}. The order k is
    Difference(1, ..., 1) with k lags of 1, Δ^k = (1 - B)^k, and
    Difference() the order 0. Difference(24) is the seasonal difference of
    hourly data with a daily period, and Difference(1, 24) the product
    (1 - B)(1 - B^24), as in SARIMA. The factors commute, so the lags are
    kept sorted.
    """

    __slots__ = ("lags",)

    def __init__(self, *lags: int) -> None:
        """Take the lags S_i >= 1 of the factors 1 - B^S_i."""
        if not all(isinstance(lag, (int, np.integer)) and lag >= 1 for lag in lags):
            msg = f"Difference takes integer lags >= 1, got {lags}"
            raise ValueError(msg)
        self.lags = tuple(sorted(int(lag) for lag in lags))

    def __eq__(self, other: object) -> bool:
        """Compare the lags."""
        if not isinstance(other, Difference):
            return NotImplemented
        return self.lags == other.lags

    def __hash__(self) -> int:
        """Hash the lags."""
        return hash(self.lags)

    def __repr__(self) -> str:
        """Return the call that builds the operator."""
        return f"Difference({', '.join(map(str, self.lags))})"


# One operator for every column, or one per column: an order k stands for
# Difference(1, ..., 1) with k lags of 1.
type Order = int | Difference | Sequence[int | Difference] | np.ndarray


def _is_single_order(order_of_derivative: Order) -> bool:
    return isinstance(order_of_derivative, (int, np.integer, Difference))


def _lags(order: int | Difference) -> tuple[int, ...]:
    """Return the lags of one column's operator: k lags of 1 for the order k."""
    if isinstance(order, Difference):
        return order.lags
    if isinstance(order, (int, np.integer)):
        if order < 0:
            msg = f"order_of_derivative must be non-negative, got {order}"
            raise ValueError(msg)
        return (1,) * int(order)
    msg = f"order_of_derivative takes ints and Difference, got {order!r}"
    raise TypeError(msg)


def column_lags(order_of_derivative: Order, n_features: int) -> list[tuple[int, ...]]:
    """
    Return the lags of every column's operator, sorted, as `encode` reads them.

    `order_of_derivative` is one order for every column, or a sequence or
    array of shape (F,) with one per column; an order is an int k >= 0 or a
    `Difference`.
    """
    if _is_single_order(order_of_derivative):
        return [_lags(order_of_derivative)] * n_features
    order = np.asarray(order_of_derivative)
    if order.dtype != object:
        order = order.astype(np.int64)
    if order.shape != (n_features,):
        msg = f"order_of_derivative shape {order.shape} must be ({n_features},)"
        raise ValueError(msg)
    return [_lags(k) for k in order]


def prefix_rows(order_of_derivative: Order, n_features: int | None = None) -> int:
    """
    Return the raw rows `encode` keeps in front of the tokens: the largest degree.

    The operator of degree p = S_1 + ... + S_m reaches p rows back: the
    order k reaches k rows, Difference(1, 24) 25. A per-column order must
    hold `n_features` entries; None takes as many as it holds.
    """
    if n_features is None and _is_single_order(order_of_derivative):
        n_features = 1
    elif n_features is None:
        n_features = len(order_of_derivative)
    lags = column_lags(order_of_derivative, n_features)
    return max((sum(column) for column in lags), default=0)


def _coefficients(lags: tuple[int, ...]) -> list[int]:
    """Return c_0, ..., c_p of c(B) = Π (1 - B^S), exactly: c_0 = 1, c(1) = 0."""
    coefficients = [1]
    for lag in lags:
        # (1 - B^S) c(B) has the coefficients c_j - c_{j-S}.
        padded = [*coefficients, *([0] * lag)]
        shifted = [*([0] * lag), *coefficients]
        coefficients = [a - b for a, b in zip(padded, shifted, strict=True)]
    return coefficients


def _headroom(lags: tuple[int, ...]) -> int:
    """
    Return the levels the scale leaves for the rounding noise c(B)ζ.

    ‖c‖₁ = Σ|c_j|, at most 2^m for m factors and less where their terms
    cancel: (1 - B)(1 - B^2)(1 - B^3) has ‖c‖₁ = 6. None at the order 0.
    """
    return sum(abs(c) for c in _coefficients(lags)) if lags else 0


# Which tokenizer `encode`/`decode` use when the caller does not say: "mash"
# (the grid anchored at the prefix) or "lgt" (the absolute grid), see the
# module docstring. Set DIFF_GPT_TOKENIZER=lgt, or pass tokenizer="lgt", for
# the latter.
DEFAULT_TOKENIZER = os.environ.get("DIFF_GPT_TOKENIZER", "mash")
_TOKENIZERS = ("lgt", "mash")

# Version of the token scheme. DiffGPT checkpoints record it, so that weights
# trained on other tokens are refused. Version 4 takes the order-0 levels
# floor(s·x) on a grid shifted by half a step, which uses all V tokens, and
# adds the operators `Difference`. Version 3 computes the levels of s·x
# exactly and rounds the scale down; version 2 rounded s·x first, which
# moved levels at rounding ties. Version 2 put both tokenizers on one level
# grid for every k; before it "mash" ran a Σ-Δ cascade (a carry scheme at
# k <= 1) and both used the scale (V - 2 - 2^k) / (2D) from k = 2 on.
ENCODER_VERSION = 4


def resolve_tokenizer(tokenizer: str | None) -> str:
    """Return `tokenizer`, or `DEFAULT_TOKENIZER` for None; reject unknown names."""
    tok = DEFAULT_TOKENIZER if tokenizer is None else tokenizer
    if tok not in _TOKENIZERS:
        msg = f"unknown tokenizer {tok!r}; expected 'mash' or 'lgt'"
        raise ValueError(msg)
    return tok


class TokenClippingWarning(UserWarning):
    """Some differences fell outside the domain and their tokens were clipped."""


def _clip_tokens(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Clip `tokens` to [0, V-1] in place and warn with the number clipped.

    A clipped token stands for a step outside `domain_of_definition`; the
    excess is lost, so past the order 0 every later decoded value is off by
    it.
    """
    n_clipped = int(np.count_nonzero((tokens < 0) | (tokens >= vocab_size)))
    if n_clipped > 0:
        msg = (
            f"{n_clipped} of {tokens.size} tokens fell outside [0, {vocab_size - 1}] "
            "and were clipped: the series steps outside domain_of_definition, and "
            "past the order 0 each clipped step offsets the decoded values after it"
        )
        warnings.warn(msg, TokenClippingWarning, stacklevel=3)
    np.clip(tokens, 0, vocab_size - 1, out=tokens)
    return tokens


# Scale of a column whose domain is D = 0, one in the kernel of c, such as a
# polynomial of degree < k at Δ^k: a fine grid, so that "lgt" reproduces its
# level to 2^-21 instead of rounding it on the unit grid.
CONSTANT_COLUMN_SCALE = float(2**20)


def _compute_scale(
    vocab_size: int,
    domain_of_definition: np.ndarray,
    lags: list[tuple[int, ...]],
) -> np.ndarray:
    """
    Per-column scale s = (V - ‖c‖₁) / (2D), the same for both tokenizers.

    s is rounded down, so that s·D <= (V - ‖c‖₁)/2 holds exactly. Levels
    round half up, so ζ = L - s·(x - A) lies in (-1/2, 1/2], and c, with
    c(1) = 0, has coefficients of both signs, so |c(B) ζ| < ‖c‖₁/2: the
    integer c(B) L = s·c(B) x + c(B) ζ stays within V/2 - 1, and every token
    lies in [1, V-1] (`abs_diffs_levels_le` in lean/DiffGPT/LevelGrid.lean).
    No larger scale does: errors with the signs of the coefficients push a
    token out (`exists_levels_ge`). At the order 0, s = V/(2D) rounded down
    below it: s·D < V/2, so the levels floor(s·x) lie in [-V/2, V/2 - 1] and
    the tokens in [0, V-1] (`floor_mem_of_domain`). Columns with D = 0 get
    `CONSTANT_COLUMN_SCALE`. Raises ValueError when V <= ‖c‖₁, where no
    scale leaves room for the noise.
    """
    scale = np.empty(domain_of_definition.shape, dtype=np.float64)
    for c in range(domain_of_definition.shape[0]):
        headroom = _headroom(lags[c])
        if vocab_size <= headroom:
            msg = (
                f"order_of_derivative {Difference(*lags[c])!r} needs vocab_size > "
                f"‖c‖₁ = {headroom}, the sum of its |coefficients|, got {vocab_size}"
            )
            raise ValueError(msg)
        domain = domain_of_definition[c]
        if abs(domain) > np.finfo(np.float64).eps:
            room = Fraction(vocab_size - headroom, 2)
            s = (vocab_size - headroom) / domain / 2
            # The quotient rounds to nearest, so it may land past the room;
            # at the order 0 s·D must stay below it, where a step of D would
            # floor to the level V/2, the token V.
            while not _within(Fraction(s) * Fraction(domain), room, strict=not lags[c]):
                s = np.nextafter(s, 0.0)
            scale[c] = s
        else:
            scale[c] = CONSTANT_COLUMN_SCALE
    return scale


def _within(product: Fraction, room: Fraction, *, strict: bool) -> bool:
    return product < room if strict else product <= room


# Values put on the levels with m factors in c must stay below 2^(62 - m) in
# magnitude, so that their differences by the factors in turn, each at most
# twice the one before, stay in int64: `decode`'s cumulative sums pass
# through every one of them, and the token sum of a step outside the domain
# stays exact, so it clips toward its own side (`toInt_tokenSum` in
# lean/DiffGPT/Machine.lean). A token within the domain is right either way,
# since int64 wraps modulo 2^64.
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
    factors: int,
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

    Raises ValueError when |p| reaches 2^min(53, 62 - `factors`): past 2^53
    float64 no longer holds every level, and past 2^(62 - m) their
    differences by the m factors of c, which `decode` integrates through,
    could leave int64 (see `_LEVEL_BITS`). Below it the levels lie in
    [-2^b - 1, 2^b] for b = min(53, 62 - m) (`levels_mem` in
    lean/DiffGPT/ExactLevels.lean).
    """
    bits = min(_FLOAT64_EXACT_BITS, _LEVEL_BITS - factors)
    s_mantissa, s_exponent = np.frexp(scale)
    x_mantissa, x_exponent = np.frexp(x)
    p, e = _two_product(s_mantissa, x_mantissa)
    exponent = np.maximum(s_exponent + x_exponent, _MIN_EXPONENT)
    with np.errstate(over="ignore"):
        p = np.ldexp(p, exponent)
    peak = float(np.max(np.abs(p), initial=0.0))
    if peak >= 2.0**bits:
        msg = (
            f"the level grid needs |s·x| < 2^{bits} with {factors} factors in "
            f"order_of_derivative, but it reaches {peak:.3g}: subtract the series "
            "level or widen a near-zero domain_of_definition"
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


def _fraction_bits(factors: int) -> int:
    """
    Fraction bits G of the "mash" anchor of an operator of m factors: min(53, 64 - m).

    Its fractional parts n / 2^G are exact in float64, and the differences
    of their numerators by the m factors in turn, below 2^(m+G-1) in
    magnitude, fit int64.
    """
    return min(_FLOAT64_EXACT_BITS, _WORD_BITS - factors)


def _difference(values: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """Return c(B) applied to `values`, the factors in turn: p fewer entries."""
    for lag in lags:
        values = values[lag:] - values[:-lag]
    return values


def _cumsum(values: np.ndarray, lag: int) -> np.ndarray:
    """
    Sum along every `lag`-th value: total_t = values_t + total_{t-lag}.

    np.cumsum down the columns of `values` laid out in rows of `lag`: modulo
    2^64 for uint64, exact for int64 or ValueError. An int64 sum overflows
    exactly when its two terms share a sign that the wrapped sum lacks; the
    first overflow in a column is caught so, before any wrapped value enters
    a later test, and the zeros that pad the last row never trip it.
    """
    n = values.shape[0]
    rows = -(-n // lag)
    grid = np.pad(values, (0, rows * lag - n)).reshape(rows, lag)
    total = np.cumsum(grid, axis=0)
    if values.dtype == np.int64 and rows > 1:
        before, terms, after = total[:-1], grid[1:], total[1:]
        if (((before ^ terms) >= 0) & ((before ^ after) < 0)).any():
            msg = (
                "the tokens integrate to levels outside int64: the decoded "
                "series leaves the grid's range"
            )
            raise ValueError(msg)
    return total.reshape(-1)[:n]


def _invert_difference(
    prefix: np.ndarray,
    steps: np.ndarray,
    lags: tuple[int, ...],
) -> np.ndarray:
    """
    Invert c(B): the values that start with `prefix` and have the differences `steps`.

    The p = S_1 + ... + S_m values of `prefix` fix them, since c_0 = 1: no
    other values do (`eq_of_diffs_eq_of_lt` in lean/DiffGPT/LevelGrid.lean).
    Each factor 1 - B^S is undone by `_cumsum` along every S-th value,
    started from the first S values of the prefix differenced by the factors
    before it. Exact for integers (`toInt_invertI64` in
    lean/DiffGPT/Int64.lean).
    """
    firsts = []
    d = prefix
    for lag in lags:
        firsts.append(d[:lag])
        d = d[lag:] - d[:-lag]
    values = steps
    for lag, first in zip(reversed(lags), reversed(firsts), strict=True):
        values = _cumsum(np.concatenate([first, values]), lag)
    return values


def _anchor_fraction(
    scaled_prefix: np.ndarray,
    n_rows: int,
    *,
    lags: tuple[int, ...],
    tokenizer: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the fractional part F of the anchor A and the integers -c(B) floor(A).

    Returns F on the p rows of `scaled_prefix` and the `n_rows` rows after
    them, and the int64 -c(B) floor(A) on the latter. The order 0 (no lags)
    takes A = 1/2: F = 1/2 and floor(A) = 0. Otherwise "lgt" takes A = 0,
    and "mash" the A with c(B) A = 0 through the prefix values rounded to G
    fraction bits (`_fraction_bits`). Then F_t = (P_t mod 2^G) / 2^G for the
    integers P = 2^G·A, and P mod 2^G depends only on the prefix numerators
    mod 2^G: P is continued with c(B) P = 0 in uint64, whose sums wrap
    modulo 2^64, a multiple of 2^G. As c(B) A = 0, -c(B) floor(A) is the
    integer c(B) F, which the numerators give exactly.
    """
    p = scaled_prefix.shape[0]
    if not lags:
        return np.full(n_rows, _HALF), np.zeros(n_rows, dtype=np.int64)
    if tokenizer == "lgt":
        return np.zeros(p + n_rows), np.zeros(n_rows, dtype=np.int64)
    bits = _fraction_bits(len(lags))
    fraction = scaled_prefix - np.floor(scaled_prefix)
    seeds = np.rint(np.ldexp(fraction, bits)).astype(np.uint64)
    anchor = _invert_difference(seeds, np.zeros(n_rows, dtype=np.uint64), lags)
    numerators = (anchor & np.uint64(2**bits - 1)).astype(np.int64)
    fraction_steps = _difference(numerators, lags) >> bits
    return np.ldexp(numerators.astype(np.float64), -bits), fraction_steps


def encode(
    inp: np.ndarray,
    vocab_size: int,
    domain_of_definition: np.ndarray,
    order_of_derivative: Order,
    *,
    tokenizer: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode a float64 (T, F) series as the tokens q_t = c(B) L_t + V/2.

    `order_of_derivative` is one order for every column or one per column,
    in a sequence or array of shape (F,); an order is an int k, for Δ^k, or
    a `Difference`. Each column gets the scale of its own operator (see
    `_compute_scale`). `tokenizer` is "lgt" or "mash", the anchor of the
    grid (see the module docstring); None picks `DEFAULT_TOKENIZER`.

    Returns `start`, the raw prefix inp[:max_p] for the largest degree
    max_p (`prefix_rows`), the per-column `scale` and the (T - max_p, F)
    int64 tokens, whose row r belongs to time r + max_p. NaN and
    infinities raise ValueError. Tokens of steps outside
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
    lags = column_lags(order_of_derivative, F)
    max_p = max((sum(column) for column in lags), default=0)
    if max_p > T:
        msg = f"order_of_derivative {max_p} exceeds the series length {T}"
        raise ValueError(msg)
    scale = _compute_scale(vocab_size, domain_of_definition, lags)
    N = T - max_p
    tokens = np.zeros((N, F), dtype=np.int64)
    for c in range(F):
        p = sum(lags[c])
        # The column's p prefix rows, then its N token rows.
        column = inp[max_p - p :, c]
        fraction, fraction_steps = _anchor_fraction(
            column[:p] * scale[c],
            N,
            lags=lags[c],
            tokenizer=tok,
        )
        levels = _levels(column, scale[c], fraction, factors=len(lags[c]))
        # Unclipped, but limited to [-1, V] so that one out of range stays out
        # of range for `_clip_tokens` to count.
        tokens[:, c] = np.clip(
            _difference(levels, lags[c]) + fraction_steps + vocab_size // 2,
            -1,
            vocab_size,
        )
    return inp[:max_p].copy(), scale, _clip_tokens(tokens, vocab_size)


def decode(
    start: np.ndarray,
    scale: np.ndarray,
    inp: np.ndarray,
    vocab_size: int,
    order_of_derivative: Order,
    *,
    tokenizer: str | None = None,
) -> np.ndarray:
    """
    Decode tokens back into the series: the inverse of `encode`.

    `start` and `scale` are the values `encode` returned along with the
    tokens, and `tokenizer` must be the one they were encoded with; None picks
    `DEFAULT_TOKENIZER`. Any max_p consecutive raw rows can serve as `start`:
    the tokens then continue from them.

    The `start` rows come back as they are. The levels of the others come
    back exactly, by inverting c in int64 from those of the prefix, and
    x̂ = (A + L)/s lies within 1/(2s) of the series where no token was
    clipped. Tokens whose levels leave int64, which `encode` never emits,
    raise ValueError.
    """
    tok = resolve_tokenizer(tokenizer)
    _check_vocab_size(vocab_size)
    _check_dtype("start", start)
    _check_dtype("scale", scale)
    if inp.dtype != np.int64:
        msg = f"tokens must be int64, got {inp.dtype}"
        raise TypeError(msg)
    N, F = inp.shape
    lags = column_lags(order_of_derivative, F)
    max_p = start.shape[0]
    degrees = [sum(column) for column in lags]
    if any(p > max_p for p in degrees):
        msg = f"order_of_derivative {degrees} exceeds the {max_p}-row start prefix"
        raise ValueError(msg)
    steps = inp - vocab_size // 2
    out = np.empty((max_p + N, F), dtype=np.float64)
    out[:max_p] = start
    for c in range(F):
        p = degrees[c]
        prefix = start[max_p - p :, c]
        fraction, fraction_steps = _anchor_fraction(
            prefix * scale[c],
            N,
            lags=lags[c],
            tokenizer=tok,
        )
        levels = _invert_difference(
            _levels(prefix, scale[c], fraction[:p], factors=len(lags[c])),
            steps[:, c] - fraction_steps,
            lags[c],
        )
        out[max_p:, c] = (levels[p:] + fraction[p:]) / scale[c]
    return out


def differentiate(
    inp: np.ndarray,
    order_of_derivative: Order,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-column differences c(B) x of a float64 (T, F) series.

    `order_of_derivative` is one order for every column or one per column,
    as for `encode`.

    Returns `start`, the raw prefix inp[:max_p], and `diff` of shape
    (T - max_p, F): column c holds the last T - max_p of its differences,
    so row r of every column belongs to time r + max_p. `integrate`
    inverts this.
    """
    T, F = inp.shape
    lags = column_lags(order_of_derivative, F)
    max_p = max((sum(column) for column in lags), default=0)
    if max_p > T:
        msg = f"order_of_derivative {max_p} exceeds the series length {T}"
        raise ValueError(msg)
    _check_dtype("inp", inp)
    start = inp[:max_p].copy()
    N = T - max_p
    diff = np.zeros((N, F), dtype=np.float64)
    for c in range(F):
        d = _difference(inp[:, c], lags[c])
        # d has length T - p_c >= N; keep its last N values.
        diff[:, c] = d[len(d) - N :]
    return start, diff


def integrate(
    start: np.ndarray,
    diff: np.ndarray,
    order_of_derivative: Order,
) -> np.ndarray:
    """
    Inverse of `differentiate`, for one order or one per column.

    `start` is the raw prefix of the series, at least max_p rows; the result
    has `len(start) + len(diff)` rows.
    """
    N, F = diff.shape
    lags = column_lags(order_of_derivative, F)
    max_p = start.shape[0]
    degrees = [sum(column) for column in lags]
    if any(p > max_p for p in degrees):
        msg = f"order_of_derivative {degrees} exceeds the {max_p}-row start prefix"
        raise ValueError(msg)
    if start.shape[1:] != (F,):
        msg = f"start has shape {start.shape}, but diff has {F} columns"
        raise ValueError(msg)
    _check_dtype("start", start)
    _check_dtype("diff", diff)
    out = np.zeros((N + max_p, F), dtype=np.float64)
    for c in range(F):
        p = degrees[c]
        # The column's last p prefix values continue into the differences;
        # the raw rows before them pad it to the full length.
        prefix = start[max_p - p :, c]
        out[: max_p - p, c] = start[: max_p - p, c]
        out[max_p - p :, c] = _invert_difference(prefix, diff[:, c], lags[c])
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


def _max_abs_difference(column: np.ndarray, lags: tuple[int, ...]) -> float:
    """
    Return the least float64 at or above max |c(B) x| over `column`, exactly.

    np.diff rounds every difference, so its max can fall short of the exact
    one, and a step of the exact max would then reach a token past V - 1.
    The rounding error of each difference is exact (Knuth's two-sum), so
    their sums bound how far each difference by c is off: where the bound is
    0 the difference is exact, and only the rows whose bound reaches the
    largest value known to be attained are summed in rationals.
    """
    diff = column
    slack = np.zeros_like(column)
    with np.errstate(over="ignore", invalid="ignore"):
        for lag in lags:
            a, b = diff[lag:], -diff[:-lag]
            diff = a + b
            b_rounded = diff - a
            error = (a - (diff - b_rounded)) + (b - b_rounded)
            slack = slack[lag:] + slack[:-lag] + np.abs(error)
        size = np.abs(diff)
        exact = slack == 0
        # The float sums of the slack fall short of the exact sums by a factor
        # (1 - 2^-53)^(2m) at most, which 2·slack covers; np.nextafter covers
        # the rounding of the bounds themselves.
        lower = np.nextafter(size - 2 * slack, -np.inf)
        upper = np.nextafter(size + 2 * slack, np.inf)
    best = Fraction(float(np.max(size[exact], initial=0.0)))
    attained = np.where(exact | np.isnan(lower), -np.inf, lower)
    threshold = max(float(best), float(np.max(attained, initial=-np.inf)))
    # A NaN bound, from an overflow, never compares below the threshold.
    rows = np.flatnonzero(~exact & ~(upper < threshold))
    # c(B) x at t + p is Σ_j c_{p-j} x_{t+j}.
    weights = _coefficients(lags)[::-1]
    for t in rows:
        value = abs(sum(w * Fraction(column[t + j]) for j, w in enumerate(weights)))
        best = max(best, value)
    return _float_at_or_above(best)


def get_domain_of_definition(
    inp: np.ndarray,
    order_of_derivative: Order,
    *,
    quantile: float | None = None,
) -> np.ndarray:
    """
    Per-column bound D on |c(B) x|, which sets the bin width 1/s = 2D/(V - ‖c‖₁).

    The max is exact: the least float64 at or above max |c(B) x|, with the
    differences of the floats in `inp` taken without rounding, so that
    `encode` clips none of these steps.

    `quantile` (e.g. 0.9999) uses that quantile of |c(B) x| instead of the
    max. The max is set by the single most extreme step in the whole
    training set, so on heavy-tailed collections it makes the grid useless
    for typical series: on M4 Daily the max is 19.5 while the median step is
    0.02, and Δ/2 alone comes to 3.8 typical steps. A quantile trades a
    small fraction of clipped values (which the encoder clips with a
    `TokenClippingWarning`) for a grid that can actually represent ordinary
    moves.

    Raises ValueError when the series has no difference to bound (T <= the
    largest degree) or holds NaN or infinities.
    """
    _check_dtype("inp", inp)
    _check_finite("inp", inp)
    if quantile is not None and not 0.0 < quantile <= 1.0:
        msg = f"quantile {quantile} must be in (0, 1]"
        raise ValueError(msg)
    T, F = inp.shape
    lags = column_lags(order_of_derivative, F)
    max_p = max((sum(column) for column in lags), default=0)
    if max_p >= T:
        msg = (
            f"order_of_derivative {max_p} needs a series longer than {max_p} rows "
            f"to have a difference to bound, got {T}"
        )
        raise ValueError(msg)
    result = np.zeros(F, dtype=np.float64)
    for c in range(F):
        if quantile is None or quantile == 1.0:
            result[c] = _max_abs_difference(inp[:, c], lags[c])
        else:
            result[c] = np.quantile(np.abs(_difference(inp[:, c], lags[c])), quantile)
    return result
