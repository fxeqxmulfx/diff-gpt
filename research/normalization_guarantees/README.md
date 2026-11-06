# Improved finite-epsilon RMS bound

The unchanged parameter-free RMS function has a checked global Lipschitz bound
of **`1/sqrt(epsilon)`**, improving the previous checked estimate
`2/sqrt(epsilon)`. Applying a fixed residual coefficient after normalization gives
**`abs(alpha)/sqrt(epsilon)`**. This improves the mathematical estimate; no model,
optimizer, benchmark result or frozen selection is changed.

For positive integer width `d` and `epsilon > 0`, the existing Lean definition is

```text
N_epsilon(x) = sqrt(d) * x / sqrt(||x||² + d*epsilon)
            = x / sqrt(mean(x²) + epsilon).

||N_epsilon(x) - N_epsilon(y)||
    <= ||x-y|| / sqrt(epsilon)

||alpha*N_epsilon(x) - alpha*N_epsilon(y)||
    <= abs(alpha)*||x-y|| / sqrt(epsilon).
```

The final statements in [RMSLipschitz.lean](RMSLipschitz.lean) use
`Transformer.GPTMini.rmsNormEps` directly. They do not introduce a substitute
normalizer or an assumption that already contains the desired bound.
The equation matches the real operation evaluated by
[rms_norm.py](../../src/diff_gpt/domain/model/rms_norm.py); the benchmark's FP32
epsilon is `2^-23 = 1/8388608`. The resulting bound is about **2896.31**, so even
this improved bound permits substantial sensitivity near zero.

The proof treats radial and angular changes together. For `c > 0`, the scalar
map `sqrt(c)*t/sqrt(t²+c)` has derivative
`sqrt(c)*c/(t²+c)^(3/2)` between zero and one. This bounds changes in radius.
The associated vector map also reduces angular distances, using Cauchy–Schwarz
and normalization coefficients at most one. Expanding squared distances avoids
the factor of two lost by the previous triangle-inequality estimate. The exact
relation `N_epsilon = (1/sqrt(epsilon))*radialMap(d*epsilon)` then gives the result.

All six statements compile with **zero warnings and only the standard Lean
axioms** `propext`, `Classical.choice`, and `Quot.sound`. Each statement with
hypotheses has a concrete positive-parameter example; the RMS examples use the
actual FP32 epsilon. [The check record](results/guarantees.json) includes source
hashes, the pinned external project and Mathlib revisions, and the compiler
output. Reproduce from `/home/misha/lean_projects/transformer` with:

```bash
lake env lean /home/misha/python_projects/gpt/research/normalization_guarantees/RMSLipschitz.lean
```

This is a real-arithmetic guarantee for one normalization operation and a fixed
post-normalization scalar. A raw attention/FFN map contributes its own
sensitivity; an input-dependent residual gate requires additional terms.
Unconstrained tied readout weights are not bounded by this result. It does not
prove floating-point accuracy, a whole-network Jacobian bound, convergence,
grokking or successful stress-test learning. Numerical repairs and their tested
limits remain in [the separate stress studies](../transformer_stress_fixes/README.md).

This supplemental proof was developed after the short selection was frozen.
The original [round-two guarantee record](../early99/results/round2/guarantees.json)
and its hashes remain intact; it correctly records the earlier, looser bound.
