# Issues and observations in `src/diff_gpt/encoder_decoder.py`

Original audit: 2026-09-05, package v0.31.1 (commit 1097617).
All 26 original `tests/diff_gpt/encoder_decoder` tests passed (4 min 17 s).
The measurements below used white noise, random walks, smooth sinusoids,
and half-integer rounding ties, with T = 1200–20000, V in {8, 16, 64, 256},
k = 0–5, in float64 and Decimal. References to legacy behavior describe
the audited MASH tokenizer; later fixes and LGT findings are recorded below.
Fixes from the review of 2026-10-19 are marked in the overview and in each
section. Version 2 of the token scheme (2026-10-20, §12) replaced the MASH
cascade with the level grid and removed the Decimal path.

## Overview

| # | Severity | Location | Finding | Status |
|---|---|---|---|---|
| 1 | High | `_sigma_delta_k`, `encode`, `_encode_column_tokens` | Silent out-of-domain clipping leaves a persistent level error | Reported by a warning; both tokenizers clip alike since §12; `predict` no longer affected |
| 2 | Medium | Scalar `encode` at k=0 | Carry scheme instead of rounding: error up to one bin, history-dependent tokens | Fixed; the carry scheme is gone (§12) |
| 3 | Medium | `integrate`, `differentiate` docstring | The `start` representation is not detected; mismatched representations silently produce incorrect output | Fixed |
| 4 | Medium | `_sigma_delta_k` docstring, `integrate` | The float64 precision limit is understated: k=4 already degrades at T around 10000 | Moot: the decoder sums int64 levels (§12) |
| 5 | Observation | Tokenization as a whole | Modulator state makes separately encoded windows differ from slices of the full encoding | LGT, the default, has no modulator state; MASH anchors at the window's prefix (§12) |
| 6 | Low | `get_domain_of_definition` | At T=k, the scalar path raises ValueError while the per-column path returns zero | Fixed |
| 7 | Low | `encode` | Token zero is never emitted for in-domain inputs | No action planned |
| 8 | Low | `_sigma_delta_k` docstring | Described as "fast vectorized", but loops over T in Python | Moot: `_sigma_delta_k` is gone (§12) |
| 11 | High | `_round_levels` (LGT) | At k=1, rounding ties overflow the vocabulary and lower the decoded level for good | Fixed |
| 12 | Change | Both tokenizers | Token scheme version 2: one level grid for every k, float64 and int64 only | Done, proved in Lean |

## 1. Silent out-of-domain clipping and persistent level error

**Location.** The `np.clip` operations following the modulator and legacy
carry schemes in [encoder_decoder.py](src/diff_gpt/encoder_decoder.py).

**Problem.** Clipping happens after the feedback loop. The modulator state
(`eps_prev` or the cumulative remainder) assumes the unclipped token was
emitted. The lost amount is never compensated and remains after decoder
integration. The caller receives neither a clipping flag nor a warning.

**Measurements.**

| Scenario | Reconstruction error | Bound Δ/2 |
|---|---|---|
| k=1, V=64, one level jump of 40·D halfway through the series | 38.5 at the jump, after 100 steps, and at the end | 0.016 |
| k=1, V=64, domain underestimated by a factor of three (D/3) throughout | 8.1 | 0.019 |
| k=2 (MASH), V=64, domain underestimated by a factor of three | 938; pre-clipping tokens range from −54 to 112 | 0.023 |

In MASH, the differentiating network `Σ (1−z⁻¹)ⁱ qᵢ` spreads a stage error
over several adjacent tokens and can amplify the effect.

**Effect outside this module.** `DiffGPT.predict` decodes context and forecast
as one token stream. A level offset caused by a context outlier therefore
carries into the forecast.

**Possible changes.**

1. Return or log the number of clipped tokens from `encode`, so `DiffGPT`
   can report clipping.
2. For first order, saturate inside the feedback loop and account for the
   token actually emitted. Prototype:

```python
def encode_saturating(x, V, D):
    scale = (V - 2) / D / 2
    y = np.diff(x, axis=0) * scale
    half = V // 2
    e = np.zeros(x.shape[1])
    out = np.zeros(y.shape, dtype=np.int64)
    for t in range(y.shape[0]):
        q = np.rint(y[t] - e)
        q = np.clip(q, -half, V - 1 - half)  # saturate inside the loop
        e = q - (y[t] - e)  # error of the emitted token
        out[t] = q + half
    return x[:1].copy(), scale, out
```

On the same 40·D jump, in-domain tokens match the existing code bit for bit,
and the error returns to Δ/2 after 39 steps while sufficient amplitude
headroom remains. Saturation inside a k≥2 MASH cascade needs a separate
stability analysis; a clipping flag is the minimum change there.

**Fixed in part (2026-10-19).** `encode` emits a `TokenClippingWarning` with the
number of clipped tokens (change 1). `predict` decodes only the generated
tokens, anchored at the last `max_k` context rows, so a clipped context step no
longer shifts the forecast. MASH at k=1 saturates inside its carry loop (change
2): a series inside its domain keeps its tokens bit for bit, and a cut step is
made up over the next ones. On a 40·D jump at V=64 the error exceeds Δ/2 on the
38 values from the jump on, then stays within it. Teacher-forced `predict`
encodes the future from the same anchor rows, so a catch-up still running at the
end of the context does not offset the covariates it returns. The k≥2 cascade
and LGT still only warn, so a clipped step leaves a level error in `decode`
there: the cascade needs the stability analysis above, and LGT has no state to
carry the cut part, which keeps its windows consistent (issue 5).

## 2. Scalar k=0 uses the carry scheme

**Location.** The scalar `encode` path in
[encoder_decoder.py](src/diff_gpt/encoder_decoder.py). When `use_mash` is false,
every order, including zero, enters the truncation-plus-carry scheme.
The per-column order-zero path instead uses `rint`.

**Problem.** At k=0 the decoder does not integrate. Noise shaping can increase
the per-sample error to a full bin, and equal values can produce different
tokens depending on the carry state. Scalar `0` and array `[0]` differ.

Uniform noise, T=5000, V=256:

| Order | Maximum error / bin width | Different tokens |
|---|---|---|
| Scalar `0` | 0.98 | 1220 of 5000 |
| Array `[0]` | 0.50 | |

A random walk with F=3, T=3000 gives 2258 differing tokens out of 9000 (25%).

**Reproduction.**

```python
x = np.random.default_rng(0).uniform(-1, 1, size=(5000, 1))
dod = get_domain_of_definition(x, 0)
for order in (0, np.array([0])):
    st, sc, enc = encode(x, 256, dod, order, tokenizer="mash")
    dec = decode(st, sc, enc, 256, order, tokenizer="mash")
    print(order, np.abs(dec - x).max() * sc[0])  # 0.98 versus 0.50
```

**Possible change.** Use the same order-zero branch as `_encode_column_tokens`.
The original `test_diff_encoder_decoder_sin_cos` threshold is about one bin
(0.007873 for a 0.007874 bin) and could then be halved.

**Fixed (2026-10-19).** A scalar order 0 takes the order-zero branch, so
scalar `0` and `[0]` give the same tokens. The sin/cos threshold is halved to
0.003937, just below Δ/2.

## 3. `integrate` does not detect the representation of `start`

**Location.** Dispatch on the type of `order_of_derivative` in `integrate`.
The `differentiate` docstring says `integrate` "detects which form it received".

**Problem.** There are two representations: the scalar path returns initial
derivative values, while the per-column path returns the raw prefix
`inp[:max_k]`. Dispatch uses the type of `order`, not the representation of
`start`. A scalar-path `start` paired with an array-valued order passes the
assertions and can produce incorrect values.

With integer data and k=2, calling `differentiate(x, 2)` followed by
`integrate(start, diff, np.array([2, 2, 2]))` raises no exception but gives a
maximum error of 18962. `DiffGPT` consistently passes the same order object
to both functions, so this mismatch does not occur on its normal path.

**Possible change.** Validate the representation explicitly, or use a raw
prefix for both paths. The per-column integrator already computes initial
derivative values from that prefix.

**Fixed (2026-10-19).** Both paths return the raw prefix `inp[:max_k]` as
`start`, and `integrate` raises `ValueError` on a `start` of the wrong shape.

## 4. Float64 decoder precision is overstated

**Location.** The `_sigma_delta_k` docstring says Decimal is "needed for k ≥ 5".
Repeated cumulative sums in `integrate` can degrade earlier.

Roundoff from an inner `cumsum` is integrated another k−1 times and grows
roughly as T^(k−1)·eps. A `differentiate → integrate` round trip without any
quantization, using white noise and V=256, gives errors in units of Δ/2:

| k | T=1000 | T=10000 | T=100000 |
|---|---|---|---|
| 1 | 0 | 0 | 0 |
| 2 | 6e-11 | 4e-9 | 6e-8 |
| 3 | 3e-8 | 8e-6 | 1.4e-3 |
| 4 | 8e-6 | 1.0e-2 | 41 |
| 5 | 6e-4 | 18 | 1.3e6 |

A full encode/decode round trip at k=4, V=256, T=4000 reaches 0.5013 bins
in float64 and 0.49995 in Decimal. The encoder identity
`combined = y + Δᵏ·ε_{k−1}` has residual 7e-15 with error magnitude ≤0.5.

**Possible changes.** Clarify the order-and-length dependence in the docstring.
The audit found k=4 safe only to roughly T=5000 at the desired precision.
Using `np.cumsum(..., dtype=np.longdouble)` may help on platforms with a wider
mantissa; on the audited x86 system it offered roughly 2000 times less
roundoff. `test_kth_order_mash_bound_is_tight_and_uniform_in_T` already allows
a factor-of-two margin for this effect.

**Fixed (2026-10-19).** The `_sigma_delta_k` docstring states the dependence:
float64 degrades at k = 4 from T of about 10^4; use Decimal there.

## 5. Modulator state makes tokenization window-dependent

A separately encoded window differs from the corresponding slice of the
full-series encoding. For a random walk, T=20000, V=256, with interior
windows of length 200:

| k | Different tokens | Maximum absolute token difference |
|---|---|---|
| 1 | 38.6% | 1 |
| 2 | 60.2% | 3 |
| 3 | 79.0% | 6 |

Both streams reconstruct the signal within Δ/2. However,
`DiffDataFrameDataLoader` encodes full series and slices windows, whereas
`predict` encodes its context with a fresh modulator state. Training and
inference therefore see different token representations of the same window.

Even an exactly periodic signal (period 50, 40 periods, V=256) can have
nonperiodic tokens. At k=1, periodicity requires the scaled increments to sum
to an integer over a period. With linear drift, adjacent periods differ in
27.7% of tokens and produce 28 distinct encodings over 38 periods. At k=2
and k=3, the cascade state can fail to return even without drift: 58% and
57% of tokens differ, with maximum differences 2 and 4, and 30 and 38
distinct encodings over 38 periods. The first and eleventh periods differ
in 62–64% of tokens.

`DiffSQLiteDataLoader.get_data` also re-encodes each window with fresh state,
while `DiffDataFrameDataLoader` slices the full encoding. The loaders can
therefore emit different token streams for identical data.

## 6. `get_domain_of_definition` at T=k

The scalar path reduces an empty array with `np.max` and raises
`ValueError: zero-size array to reduction operation maximum`.
The per-column path returns zero. The behavior should be consistent,
preferably with an explicit validation message.

**Fixed (2026-10-19).** Both paths raise `ValueError`: the series needs more
than k rows.

## 7. Token zero is unused

In 80000 tokens at V=16, k=0–3, token zero never appeared. At k≥2, tokens 1
and V−1 were also absent. Scaling by `(V−2)/(2D)` and rounding/truncation
keeps signed values within `(V−2)/2`, hence tokens within [1, V−1]. This
wastes one or more vocabulary slots and motivates revisiting the scale.

## 8. The "fast vectorized" docstring

The MASH float path loops over T·k steps in Python and vectorizes only over
channels. A first-order stage can be expressed through `S = cumsum(u)`,
`Q = round(S)`, `q = diff(Q)` with a leading zero, `e = Q − S`, and
`u_next = −e`. This resembles the legacy carry scheme. Equivalence at exact
half-integer ties and under floating-point rounding was not verified;
this remains a proposal.

**Fixed (2026-10-19).** The docstring now says the loop runs in Python over
T·k steps and vectorizes over columns only.

## 9. Findings while making LGT the default (2026-09-05)

Both findings appeared when running the Sigma-Delta encoder tests with LGT.

**9a. LGT requires V > 2ᵏ+2 at k≥2; there is no fallback.** The compatibility
encoder can fall back to a first-order carry scheme with ±1 headroom.
For LGT, the term Δᵏζ always needs ±2^(k−1) bins. At V=4, k=2, x=t²,
overflowing tokens were clipped and the decoder returned x/2.
**Fixed:** `_compute_scale(..., lgt=True)` reserves the required headroom
and raises a ValueError suggesting `tokenizer='mash'` when V is too small.

**9b. Constant columns (D=0).** Sigma-Delta uses scale 1 and preserves the
level through `start`. LGT initially rounded a constant such as 0.01 to the
unit grid and lost it. **Fixed:** at D=0, LGT uses
`LGT_CONSTANT_COLUMN_SCALE = 2^20`, giving error ≤2^-21.

**Tests.** `test_encoder_decoder.py` explicitly selects `tokenizer="mash"`.
Its thresholds are empirical Sigma-Delta maxima just below Δ/2, such as
0.0037750 against a 0.0037760 bound. LGT can approach the theoretical bound
more closely and exceed those empirical thresholds by 0.03%. Sigma-Delta
is exact when the k-th difference is on its grid, while LGT is exact when
the signal level is on its grid. `test_lgt.py` checks the LGT Δ/2 bound.

## 10. Choosing k chooses the signal assumption (user observation, 2026-09-05)

The domain is estimated from training data and tokens are clipped to it.
The derivative order determines which signals remain representable:

- k=0 requires the level itself to remain within the observed range.
  A sustained trend eventually reaches the boundary and produces a plateau.
- k=1 requires bounded first differences. A linear trend has constant
  increments, and integration can continue the growth without a level bound.
- k=2 can represent quadratic trends through bounded second differences.

For polynomial trends, **k should be at least the trend degree**. This is
the practical form of the README assumption that some derivative is bounded.
Measured fractions outside the training-derived domain:

| Data | k=0 | k=1 |
|---|---|---|
| Synthetic 0.01·t + sin(t/7), domain from first 80% | 15.4% boundary tokens; continuation error 3.96 versus 0.017 in-domain | 2.7%; error 0.0006 throughout |
| M4 Hourly, per-series z-normalization | 1.86% of points; 25.4% of series affected | 0.22%; 7.7% of series affected |
| exchange_rate with drift | 30.2% | 0.03% |

Clipping at k≥1 leaves an integrated level error; at k=0 it causes local
saturation and prevents continuation of a trend. Order zero is therefore
an option for stationary-level series rather than a general default.

## 11. LGT at k=1 overflows the vocabulary on rounding ties (2026-10-19)

**Location.** `_round_levels` rounded half to even, and `_compute_scale`
gives k ≤ 1 the scale (V−2)/(2D).

**Problem.** When s·x is exactly n + 1/2, a step of exactly D gives
ΔL = V/2, that is token V, which the clip turns into V−1. Every such clip
lowers the decoded level by one bin for the rest of the series. This broke
the README's Δ/2 bound for the default tokenizer at k=1, the order every
benchmark uses, on integer or grid-valued data, where ties are exact.

**Reproduction.** x = [3, 5], D = 2, V = 256 give s = 63.5 and levels 190
and 318 (190.5 and 317.5 round to even), so ΔL = 128 and token 256 clips to
255. On an integer walk with steps −2..2 and T = 5000, 261 of 4999 steps
clipped and the decode error reached 4.12, that is 523·Δ/2. MASH on the same
data stays within Δ/2, and Decimal mode behaved like float64. k = 0 and k ≥ 2
stayed within Δ/2 on tie-heavy data.

**Fixed (2026-10-19).** Levels round half up, L = floor(s·x + 1/2), in float
and in Decimal. Then |ΔL| ≤ (V−2)/2 at k=1, and the k=0 and k≥2 bounds still
hold because |ζ| ≤ 1/2. Tokens change only at exact ties. `test_lgt.py`
checks the integer walk in both modes.

## 12. Token scheme version 2 (2026-10-20)

**Change.** Both tokenizers put each column on one level grid, for every k:
L = ⌊s·x − B + 1/2⌋, q = Δ^k L + V/2, s = (V − max(2, 2^k))/(2D). The anchor B
is a polynomial of degree < k, which Δ^k removes: B = 0 for LGT, and for MASH
the polynomial through the k prefix values. The MASH Σ-Δ cascade, its carry
scheme at k ≤ 1 and the saturation at k = 1 (§1) are gone, as is the Decimal
path. Every token lies in [1, V−1] and the error stays within
D/(V − max(2, 2^k)) for both tokenizers (`lean/DiffGPT/LevelGrid.lean`).
Checkpoints record `ENCODER_VERSION` = 2, and `DiffGPT.load` refuses others.

**float64 and int64.** MASH needs B only through its fractional part F, a
multiple of 2^−G with G = min(53, 64 − k), which float64 holds exactly. The
numerators of F are continued modulo 2^64 in uint64; with R = ⌊s·x − F + 1/2⌋
the tokens are Δ^k R + Δ^k F + V/2, where Δ^k F is an integer, and
x̂ = (R + F)/s. Levels round half up exactly (`roundLevels_eq` in
`Float64.lean`), `np.diff` and the decoder's cumulative sums are exact in
int64 below the guard, and `_cumsum` detects the first overflow
(`Int64.lean`). `Machine.lean` puts these together for the code as written.

**Guard.** `encode` raises where |s·x| reaches 2^min(53, 62−k). Past 2^53
float64 misses whole levels: one sin/cos period sampled every 10^−6 has a
k = 3 domain of float64 noise, s·x of 7·10^16, and with only the int64 guard
over a thousand tokens clipped and LGT decoded 0.03 off, against Δ/2 of
10^−17. Past 2^(62−k) the k-th differences would overflow int64.

**Remaining float64 caveat.** The token range is exact only where s·x is
computed exactly. Otherwise a token can reach V, and clip, where a step of
nearly D has all its k + 1 levels within about 2^−53·|s·x| of rounding ties
(`token_mem_of_domain`). No series in the tests has hit this.

## Checks that passed in the original audit

| Check | Result |
|---|---|
| Float64 bound ‖x̂−x‖∞ ≤ Δ/2, k=0–4, V in {8,16,64,256}, four signal types including half-integer ties | Holds apart from the 0.1–0.3% k=4, V=256 excess explained in Section 4 |
| Same bound in Decimal, k=1–5 | Holds; 0.49995 bins at k=5 |
| Decimal versus float64 tokens, k=1–5, T=1200 | Bitwise equal; at k=4, T=4000, 5 of 8000 differ |
| Scalar k versus array [k,…], k=1,2,3 | Identical tokens |
| Causality: `encode(x[:n])` matches the prefix of `encode(x)`, including mixed orders | Holds for start, scale, and tokens with a shared domain |
| `differentiate → integrate` on integer data, matching representations | Exact inversion |
| MASH identity `combined = y + Δᵏ ε_{k−1}` | Residual 7e-15; maximum error magnitude 0.49998 |
| In-domain clipping, intercepted `np.clip`, adversarial ties | No clipping observed |
| Mixed orders [0,1,2,3], V=256 | Each column satisfies its own bound |
| T=max_k (empty tokens) and T<k | Correct output and assertion, respectively |
| Decimal legacy path (`np.trunc` on object arrays) | Works; tokens match float64 |

## Appendix: confirmed issues outside `encoder_decoder.py`

These were found in the wider review and checked with scripts. Item B, on
the attention residuals, went away with the move to the plain GPT.

**A. `train()` returned the schedule-free y weights instead of x.**
After the last `optimizer.step()`, no `optimizer.eval()` was called. Without
best-snapshot restoration, the model remained at the extrapolated point.
The optimizer returned with `train_mode=True`, and weights differed from x
by 4e-3 after six iterations. Snapshots were correctly taken in eval mode.
The change adds `optimizer.eval()` after training and before snapshot restore.

**C. `DiffDataFrameDataLoader.target_columns` ignored window phase.**
**Fixed on 2026-09-05:** `_mask_targets` receives a phase per batch row;
`align_windows` and `test_data_loader_phase.py` were added.
The original mask assumed position j belonged to channel `j % n_features`.
For a window starting at flat offset `local`, the actual target channel is
`(local + 1 + j) % n_features`. The original mask was aligned in only 1/F
of windows. Three distinguishable channels (constant → 32, slope +1 → 33,
slope −1 → 31; V=64, D=31), 50 batches of 64 windows, produced:

| target_columns | Unmasked y from c0 | From c1 | From c2 |
|---|---|---|---|
| [0] | 0.32 | 0.34 | 0.33 |
| [1] | 0.33 | 0.32 | 0.34 |
| [2] | 0.34 | 0.33 | 0.32 |

Thus, loss used covariates in two thirds of windows despite selecting one
target channel. The original test checked window-relative periodicity and
missed this. Passing `(local + 1) % n_features` per batch row fixes the
mask. Aligning window starts to channel boundaries also matches inference,
whose context starts at channel zero. `_predict_teacher_forced` already
used absolute token positions and was unaffected.

**Completed on 2026-10-19.** The phase is relative to the val slice, whose
start `n` lay on a column boundary only with `align_windows=True`. With the
default `align_windows=False` and n % F ≠ 0 (3 columns, 335 rows, k=1,
train_part=0.7 give n = 701), the val loss scored the neighbouring column.
The split point is now always rounded down to a whole row.

**D. Smaller issues.** `train_time_s` used `timedelta.seconds` and wrapped
after a day. Sampling entry points did not forward a seed, so Engine's
fixed seed 42 produced identical repeated samples. The M4 benchmark
docstring referred to the nonexistent `DiffGPT.train_multi` method.
