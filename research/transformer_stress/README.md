# Transformer component and residual stress

This independent suite tests the user's complete Peri-RMS / XSA / SwiGLU /
SSMax / QK-RMS / scaled residual / tied-weight / parameter-free RMS /
embedding RMS / Log-Scaled LeRoPE candidate. It imports the actual frozen GPT
operations. It does not edit or replace the running early99 benchmark.
The [plan](plan.json) fixes every magnitude, depth, seed, budget and ranking
before scientific execution. The separate [Q/K retrieval stress](../qk_scale_stress/README.md)
already provides paired training evidence for that component.

Run from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress.run --check-only
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress.run --components --learning --report
```

The default archive is `runs/transformer-stress/v1/`. It retains source hashes
and copies, data tensors and fingerprints, preflight equations, every numerical
case, all training histories, model/optimizer checkpoints and a complete CSV.
Changed scientific sources are rejected on reuse. Completed training cells may
be skipped, but an interrupted incomplete cell is not silently resumed.

## What is measured

| Component | Stress and measurements |
|---|---|
| Embedding / parameter-free RMS | Near zero to huge input scales; output bound, error against FP64 with the same epsilon, radial and tangent gradients. Finite zero after overflow is an accuracy failure. |
| Q/K RMS | Prior 24-cell retrieval experiment, plus the complete GPT's QKV weight-scale probes. |
| SSMax | Singleton through 65536 keys; fixed zero/positive margins, probability, entropy and the exact conditional concentration equation. |
| Centered XSA | Constant, nearly parallel and common-offset values around the clipping threshold and at large magnitudes; original clipped projection in FP64. References use the same quantized values, not lost pre-rounding information. |
| SwiGLU | Increasing input-projection weight magnitude; intermediate products, output RMS and input gradients. A bound after RMS does not bound earlier products. |
| Residual | Identity-skip counterexamples, normalized aligned/contracting branches across depth, Jacobian singular values, and the paired learning experiment below. |
| Tied readout | Equality of the combined gradient to both parameter-use paths; embedding norm and unbounded readout-logit scale are measured separately. |
| LeRoPE and log-angle variant | Position/frequency sweeps compare FP32 phases with an FP64 angle formula; near-isometry and phase accuracy are different properties. FP64 is not arbitrary-precision ground truth at extreme angles. |
| Complete GPT | 65 probes over depth, context and individual weight/frequency interventions; all hidden RMS values, logits, gradients, tied identity and FP64 discrepancies. |

## Residual alternatives

There are 84 paired learning cells: seven residual choices × two depths
(4 or 16 branches) × two teacher strengths × three seeds. Inputs have RMS one;
the fixed teacher is `y = x + a*tanh(xR)` for one orthogonal `R`. Every arm
learns the same SwiGLU matrices from the same draws, with pre-RMS input and
no learned readout. The teacher gives a measurable nonzero residual to learn.
All runs receive 512 native AdamW updates, including the zero-rate first
warmup update; the common threshold is 5% of the identity predictor's MSE.
This is residual-function learning, not exact-sequence classification.

Write `D` for branch count and `N` for parameter-free RMS. The tested branch
contributions are raw `F`, `N(F)`, `N(F)/sqrt(D)` (current), `N(F)/D`,
`2*sigmoid(s)*N(F)/sqrt(D)`, `N(2*sigmoid(s)*F)/sqrt(D)`, and
`s*N(F)/sqrt(D)`. All gates start at zero; positive sigmoid gates therefore
start at one, while the ReZero-style scalar starts at zero. Gates add `D`
parameters. The post-gate and pre-gate arms have identical initial forwards to
the current scaled arm. The pre-gate arm is a control for RMS cancelling a
scalar's effect. The zero-gate arm begins as identity but still must learn.

The plan ranks coverage first. Only complete threshold coverage permits a
geometric mean of first-crossing updates. Otherwise the worst final relative
MSE over **all** cells is used; numerical failures rank last. Every seed,
miss and regression is retained. The best result here is a residual-test
candidate, not a universal transformer recommendation.

## Guarantees and related alternatives

The local transformer project at `8890a985` supplies checked contracts in
`Transformer.Residual.GradientPaths` and `Transformer.MHC.Section4_*`.
For normalized input, the current fixed scaled residual has real-arithmetic
RMS bound `1 + sqrt(D)`, while `1/D` has bound `2`. A bounded sigmoid
post-gate has bound `1 + 2*sqrt(D)`. None is a lower bound on the network's
Jacobian singular values. An unconstrained learned zero-initialized scalar
needs a bound on its later value before claiming a uniform state bound.

[ReZero](https://arxiv.org/abs/2003.04887) motivates the zero-initialized scalar
gate only. This experiment retains Peri-RMS and depth scaling and does not
claim to reproduce the paper's full model or training recipe. The bounded
post-gate idea also appears in the local `adaptive_residual.py`; here each gate
is a scalar, not its input-dependent linear row.

[mHC, Sections 4.1–4.2](https://arxiv.org/html/2512.24880v2) motivates a separate
skip-mixing diagnostic. Exact doubly stochastic mixing is nonexpansive and
preserves the stream mean, but can erase differences between streams. Twenty
finite Sinkhorn iterations need not make both sums exactly one. Literal
exponentiation and a log-domain evaluation are both recorded. These tests do
not implement or rank the complete input-dependent mHC architecture.

The earlier local `residual_generalization` experiment also records poor
held-out learning for depth-attention alternatives despite fitting training
data. Its tasks, optimizer and budgets differ, so its results are not merged
into this suite or early99. DeepNorm and ResiDual remain related candidates,
not tested reproductions in this small residual comparison.

## Complete first results

All **322 numerical probes and 84 learning cells** completed. The
[audit](results/audit.json) verifies every source archive, paired initialization,
optimizer/sampler state, history and final full-split evaluation. Run it with
`PYTHONPATH=. .venv/bin/python research/stress_audits/transformer_v1.py`.
The [complete numerical records](results/components.json),
[learning summary](results/learning-summary.json) and
[all-cell CSV](results/learning-cells.csv) retain every failure.

The five ordinary full-stack controls remain finite and agree closely with
FP64. Extreme interventions expose genuine limitations of the implementation:
native FP32 RMS can return an incorrect finite zero after its squared norm
overflows; the standalone SwiGLU probe does this at projection scale `1e10`
and becomes nonfinite at `1e20`. The tied-readout probe similarly returns zero
at weight scale `1e20`, despite a nonzero finite FP64 reference. Ten of the 65
full-stack probes have nonfinite values/gradients, at the largest SwiGLU and
frequency interventions. Other probes remain finite with large accuracy errors.

Log-Scaled LeRoPE's rotation norm is still accurate to about `1e-7` when its
phase vector is badly wrong. At position 1048576 and raw log frequency 4,
the maximum phase-vector error is **1.999** versus the FP64 formula. Even
at the initial log frequency zero it is **0.03994** there. Norm preservation
does not establish intended positional accuracy.

All residual learning runs stayed finite, but **0/84 reached the fixed 5%
relative-MSE threshold**. No filtered first-crossing mean is reported.

| Residual choice | Passed / 12 | Worst final relative MSE | Median final relative MSE |
|---|---:|---:|---:|
| Raw pre-normalized branch | 0 | 0.1670 | 0.1349 |
| Unit Peri-RMS | 0 | 29.2124 | 0.9371 |
| Current `1/sqrt(D)` Peri-RMS | 0 | 3.8927 | 0.2768 |
| `1/D` Peri-RMS | 0 | 0.3968 | 0.2375 |
| Bounded post-RMS gate | 0 | 2.8241 | 0.2615 |
| Bounded pre-RMS gate | 0 | 3.8926 | 0.2768 |
| Zero-initialized scalar gate | 0 | 0.2667 | 0.1844 |

The raw pre-normalized branch ranks first under the declared worst-error
tie-break, but it **does not solve the test**. The zero-initialized gate is
more robust to depth here than the current scaled Peri-RMS. The post-gate
does not remove the learning failure, and the pre-gate closely reproduces the
fixed arm. These are evidence against declaring the complete recipe ideal.

The user's subsequent instruction is to fix failures. A separate
`research/transformer_stress_fixes/` experiment preserves these original
results and tests numerical repairs separately from a learning change.
The original data, thresholds and budgets remain unchanged.
