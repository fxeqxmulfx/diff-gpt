# Fixing the transformer stress failures

The [original stress suite](../transformer_stress/README.md) found numerical
failures and missed its learning threshold in all 84 residual cells. This
follow-up preserves those failures and tests fixes under the
[separately frozen plan](plan.json). Nothing changes in the running early99
benchmark or its source snapshots.

The numerical experiment repeats all 65 full-stack interventions with the
unchanged parent and six candidates: safer RMS, wider rotary angles, wider
SwiGLU, wider XSA/output projection, all four repairs, and all repairs plus
an explicitly bounded learned frequency. Model parameters and residual states
stay FP32. The reference computes the intended angle map in FP64, so the
parent is remeasured under that same reference; the original v1 measurements
against a promoted rounded table are preserved separately.

RMS uses a scaled sum of squares and a pairwise FP64 backward calculation
that avoids subtracting nearly equal radial projections. SwiGLU and XSA
retain wider intermediate values through their output normalization. Rotary
angles are constructed in FP64 before sine/cosine. These repairs preserve
the real equations. The pairwise RMS backward costs quadratic work in width;
this is a correctness prototype and runtime optimization remains deferred.

Wider arithmetic alone cannot bound `exp(alpha)` or its gradient as the
learned parameter grows. The last candidate therefore changes the effective
log frequency to `4*tanh(alpha/4)`. It retains the initial frequency and
initial derivative, limits its multiplier to `[exp(-4), exp(4)]`, and bounds
the multiplier's derivative by `exp(4)`. It can saturate at large raw values;
this does not establish successful learning or a global Jacobian bound.
Its numerical reference uses this explicitly changed equation.

The learning experiment changes only SwiGLU's product from
`SiLU(gate)*value` to `(1+SiLU(gate))*value`. The added linear path uses the
same matrices, initialization and parameter count. All 84 cells reuse the
original data, seeds, optimizer, budget and 5% relative-MSE threshold.
Numerical repairs are not mixed into this ablation. A train-only linear
least-squares diagnostic has held-out relative MSE 0.06227, so a linear
path by itself is insufficient: the nonlinear part must still learn.

The fix validation has **12 passing checks**, including finite differences,
radial derivatives at extreme scales, cached decoding, causality, parameter
pairing and an AST check that the learning loop changes only its model
constructor. The original suite has five further passing checks.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_stress_fixes.run --numerical --learning
```

Sources, data and complete outputs are retained under
`runs/transformer-stress/fixes-v1/`. Scientific results are pending; no claim
that these candidates solve the failures is made before measuring them.
