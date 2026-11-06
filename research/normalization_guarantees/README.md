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

## Normalized attention sensitivity

[AttentionSensitivityImproved.lean](AttentionSensitivityImproved.lean) bounds
the same real RMS, split-half rotary and softmax equations. For a fixed
nonempty visible row and fixed rotary phases, let

```text
score_j(q) = gamma * <RMS-and-rotate(q), RMS-and-rotate(key_j)>
p(q) = softmax(score(q))
C = abs(gamma) * sqrt(head_width) / sqrt(epsilon).

|score_j(q) - score_j(q')| <= C * ||q-q'||              (fixed keys)
sum_j |p_j(q,K) - p_j(q',K)| <= C * ||q-q'||           (fixed keys)
sum_j |p_j(q,K) - p_j(q',K')| <= C * (||q-q'|| + E)    (each key changes <= E).
```

Here `q` and the keys are **projected vectors before their RMS**. The joint
bound permits every key to change by Euclidean norm at most `E`. Neither
bound contains an extra factor equal to the number of keys. For the
implemented SSMax gain `gamma = m*log(N)*g`, it grows with `log(N)` and scales
linearly with `m`. Half-scale and quarter-scale reduce these displayed constants
by factors of two and four relative to the same parent. With ideal original
gain `g=1/sqrt(head_width)`, the width factor cancels.

The proof establishes that arbitrary fixed rotations preserve differences,
then combines the improved RMS Lipschitz estimate, the normalized key's norm
bound and Cauchy–Schwarz. [SoftmaxLipschitz.lean](SoftmaxLipschitz.lean) proves
the stronger infinity-to-one estimate
`sum(abs(softmax(u)-softmax(v))) <= max(abs(u-v))`, improving the previously
checked constant four to one. It bounds each subset's probability-mass
derivative by half the maximum logit speed, applies the mean value theorem,
then uses the subset where the endpoint difference is positive. No optimality
theorem is claimed.

All **seven new statements** compile with **zero warnings and only standard
axioms**. Every statement has an example; attention examples use the actual
FP32 epsilon and quarter-scale ideal gain, with positive width, nonempty rows
and varying phases. The [new proof record](results/improved-sensitivity.json)
contains the exact commands, dependencies and compiler-output hashes. Builds
use temporary artifacts and do not edit the external Lean project.

The earlier four-statement [attention proof](AttentionSensitivity.lean), its
[record](results/attention-sensitivity.json) and compiler output remain intact.
Its looser `4*C` query bound is still true. The new proof improves that bound
to `C` and adds simultaneous query/key control; it changes no model behavior.

These results control one probability row under projected query/key
perturbations. They do not control the value path, learned angles, XSA,
unbounded projection or readout matrices, or the whole network. Small epsilon still
allows large constants. It adds no finite-precision or learning guarantee and
does not remove the weaker concentration bound at smaller score coefficients.
The short-suite selection and its frozen proof records remain unchanged.

The [supplemental audit](../early99_score_scale_reporting/results/supplemental-proof-audit.json)
verifies all **21** statements across the RMS proof, initial attention proof,
strengthened sensitivity results and selectivity bounds below. It checks the
actual printed axiom lists, fixture
and compiler-output hashes, implementation references, external dependency
revisions and Lean toolchain. It does not recompile the proofs or claim a
verified PyTorch translation. Reproduce from the GPT repository root with
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99_score_scale_reporting.verify_supplemental`.

## Attention selectivity ceiling

[AttentionSelectivity.lean](AttentionSelectivity.lean) supplies the complementary
**upper** bound on any one key's probability. For a nonempty row of `N` visible
keys, if every score satisfies `abs(score_j) <= B`, then

```text
p_i <= 1 / (1 + (N-1)*exp(-2*B)).

For Q/K RMS + arbitrary rotations + fixed SSMax coefficient m >= 0:
B = m*log(N)*abs(g)*d.
```

The proof retains the selected key in the denominator, tightening the older
`exp(2*B)/N` estimate. It applies to every projected query and key with
`epsilon > 0`, positive head width `d`, and arbitrary rotary phases. It needs
no assumed winning margin. The general theorem accepts any fixed real gain
`g`, including the actual rounded buffer value; it does not silently replace
that value by an exact `1/sqrt(d)`.

With ideal gain `g=1/sqrt(d)`, algebraically rewriting the exponential gives
`p_i <= 1/(1+(N-1)*N^(-2*m*sqrt(d)))`. Softer scores tighten sensitivity bounds
while limiting the concentration that a single row can achieve. The checked
specialization **`d=4, m=1/4, g=1/2`** gives **`p_i <= N/(2*N-1)`** for every
positive epsilon. Its concrete 64-key example gives **`p_i <= 64/127`**, about
**0.50394**, regardless of the projected vectors and phases. Finite epsilon
can make the actual attainable maximum lower; this is an upper bound.

This width-four example is **outside the fixed benchmark**. Its 35 task
configurations use head widths 16 (10 tasks), 32 (24 tasks), and 128 (one task),
as verified directly from the frozen suite. The result is a constraint on
reusing the same gain at other widths, not an explanation inferred from long
outcomes. A ceiling on one probability row does not prove a full transformer
cannot solve a task: values, XSA, multiple layers and readout remain outside
this statement.

All **four statements** and their concrete examples compile with zero warnings
and only `propext`, `Classical.choice`, and `Quot.sound`. The
[record](results/attention-selectivity.json) includes source/output hashes,
the dependency build command, exact hypotheses and implementation mapping.
No model, scientific plan, training cell, or frozen selection changed.
