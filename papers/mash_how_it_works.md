# How MASH (Sigma-Delta quantization) works in diff-gpt

Based on `paper.tex` (Section 2) and
[`src/diff_gpt/encoder_decoder.py`](../src/diff_gpt/encoder_decoder.py).
This note describes the compatibility tokenizer selected with `tokenizer="mash"`;
the package now defaults to level-grid tokenization (LGT).

MASH stands for **multi-stage noise shaping**: a cascade of Sigma-Delta
($\Sigma\Delta$) modulators. In diff-gpt it runs before the language model and
converts a continuous signal into integer tokens with a small reconstruction
error bounded uniformly in the sequence length $T$, under the stated
domain, vocabulary, and arithmetic assumptions.

## 0. Why use Sigma-Delta instead of simply rounding?

The decoder reconstructs the signal by integrating the dequantized differences
$k$ times. At order one:

$$\hat x_{t+1}=\hat x_t+(q_t-V/2)\,\Delta.$$

Integration accumulates errors. Independently rounding every difference can
produce a reconstruction drift that grows with $T$; for example, independent
zero-mean rounding errors give a typical $O(\sqrt T)$ accumulated error.

**Noise shaping** feeds the accumulated quantization error back into the next
input. The resulting error has a difference structure that integration cancels.

## 1. The building block: a first-order error-feedback modulator

For an input $u_t$, one stage computes

$$
\begin{aligned}
y^{\mathrm{eff}}_t&=u_t-e_{t-1},\\
q_t&=\operatorname{round}(y^{\mathrm{eff}}_t),\\
e_t&=q_t-y^{\mathrm{eff}}_t,\qquad |e_t|\le\tfrac12.
\end{aligned}
$$

Substitution gives the first-order identity

$$q_t=u_t+(e_t-e_{t-1})=u_t+(1-z^{-1})e_t,$$

where $z^{-1}$ delays a sequence by one step. The quantization error appears
as a first difference. In the implementation, this is the inner loop over `i`:

```python
y_eff = stage_input - eps_prev[i]
q_rounded = round_op(y_eff)
eps_cur = q_rounded - y_eff
eps_prev[i] = eps_cur
```

## 2. Higher orders and the cascade

A direct higher-order feedback loop requires a separate stability argument.
MASH instead builds a $k$-th-order noise-shaping operator from $k$ stable
first-order stages:

- Stage 1 receives $y_t=s\,\Delta^k x_t$, the scaled $k$-th difference.
- Each later stage receives the negative quantization error of the preceding
  stage, $-\varepsilon^{(i-1)}_t$, bounded in magnitude by $1/2$.

The implementation uses an outer loop over stages:

```python
stage_input = y[t]
for i in range(k):
    y_eff = stage_input - eps_prev[i]
    q_rounded = round_op(y_eff)
    eps_cur = q_rounded - y_eff
    q_stage[i, t] = q_rounded
    eps_prev[i] = eps_cur
    stage_input = -eps_cur
```

## 3. Combining stages with digital differentiators

Apply $(1-z^{-1})^i$ to the output of stage $i+1$, then sum:

$$
q_t=\sum_{i=0}^{k-1}(1-z^{-1})^i q^{(i+1)}_t
   =q^{(1)}_t+\Delta q^{(2)}_t+\cdots+\Delta^{k-1}q^{(k)}_t.
$$

Here $\Delta=(1-z^{-1})$ denotes the difference operator, distinct from the
quantization step also conventionally written $\Delta$ in the paper.
The binomial expansion gives the implementation:

```python
for i in range(k):
    for j in range(i + 1):
        coef = (-1)**j * comb(i, j)
        combined[j:] += coef * q_stage[i, :T-j]
```

Finally, shift the signed output into the vocabulary and clip:

```python
tokens = np.rint(combined).astype(np.int64) + half
np.clip(tokens, 0, vocab_size - 1, out=tokens)
```

## 4. The noise-shaping identity

The intermediate errors telescope, leaving only the final stage's error:

$$
\boxed{q_t-y_t=(1-z^{-1})^k\varepsilon_{k,t},
       \qquad |\varepsilon_{k,t}|\le\tfrac12.}
$$

The decoder integrates $k$ times. With the matching initial conditions, the
inverse difference operators cancel, leaving

$$
\hat x_T-x_T=\varepsilon_{k,T-k}\,\Delta,
\qquad \|\hat x-x\|_\infty\le\frac{\Delta}{2}
$$

uniformly in $T$. This identity assumes the encoded tokens are not clipped.

## 5. Why the scale reserves headroom

The combined output can exceed the input by

$$
|q_t-y_t|=|\Delta^k\varepsilon_k|
\le\|(1-z^{-1})^k\|_1\max|\varepsilon_k|
=2^k\cdot\tfrac12=2^{k-1}.
$$

The binomial coefficients have total absolute weight $2^k$. To keep this
overshoot inside the vocabulary, the scale reserves $2^k$ bins overall:

$$
s_c=\frac{V-2-2^k}{2D_c},
\qquad \Delta_c=\frac1{s_c}=\frac{2D_c}{V-2-2^k}.
$$

The corresponding scale calculation is

```python
if use_mash_c:  # k >= 2 and V > 2**k + 2
    scale[c] = (vocab_size - 2 - 2**k_c) / domain[c] / 2
else:
    scale[c] = (vocab_size - 2) / domain[c] / 2
```

The higher-order MASH branch requires `k >= 2 and vocab_size > 2**k + 2`.

## 6. Which scheme runs at each order?

The per-column compatibility encoder has three branches:

| Order | Scheme | Reason |
|---|---|---|
| $k=0$ | Round, shift, clip | No integration during decoding |
| $k=1$, or insufficient vocabulary headroom | Legacy carry-based first-order Sigma-Delta | Simpler compatibility branch |
| $k\ge2$, $V>2^k+2$ | MASH cascade (`_sigma_delta_k`) | Higher-order noise shaping with a uniform $\Delta/2$ bound |

Each channel may have its own derivative order $k_c$. The scalar legacy
order-zero path has a known discrepancy from the per-column path; see
[the encoder/decoder issue report](../encoder_decoder_issues.md).

## 7. Float64 and Decimal

`_sigma_delta_k` supports two arithmetic modes:

- `use_decimal=False`: float64, vectorized over channels, with Python loops
  over time and stages.
- `use_decimal=True`: `Decimal` arithmetic with configurable precision.
  Repeated integration can amplify floating-point roundoff at high orders
  or long horizons. The issue report records visible float64 error already
  at $k=4$ for sufficiently long signals; $k\ge5$ needs particular care.

Decimal is slower and still requires sufficient precision for the desired
sequence length and order.

## 8. Reconstruction error and forecast error

The $\Delta/2$ bound concerns an ideal round trip using the true encoded
tokens. During forecasting, the decoder integrates predicted tokens, whose
errors need not have the noise-shaping structure.

An isolated token prediction error of magnitude $\eta$ contributes

$$O\!\left(\eta\,\Delta\,H^{k-1}\right)$$

after $H$ steps. At $k=1$ that isolated contribution persists; at $k=2$ it
grows linearly. Errors at multiple forecast steps also accumulate, so this
isolated-error expression is not a bound for an entire erroneous rollout.

The recorded benchmark preferred $k=1$: increasing to $k=2$ worsened sMAPE
from 17.49 to 39.27 despite lower token-level cross-entropy. Noise shaping
controls quantization error under exact encoding; forecast errors require
separate analysis.

## Main points

1. Independent rounding of differences can produce growing reconstruction drift.
2. MASH cascades stable first-order stages, each quantizing the preceding error.
3. Differentiated stage outputs telescope to $(1-z^{-1})^k\varepsilon_k$.
4. Matching decoder integration cancels that operator, giving a uniform
   $\Delta/2$ reconstruction bound under the stated assumptions.
5. Vocabulary headroom and adequate arithmetic precision are necessary.
6. The empirical choice of derivative order must also account for forecast
   error amplification.
