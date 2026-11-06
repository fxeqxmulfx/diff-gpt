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
`runs/transformer-stress/fixes-v1/`. The [audit](results/audit.json) verifies
all 455 numerical cases and 84 learning checkpoints. Reproduce it with
`PYTHONPATH=. .venv/bin/python research/stress_audits/transformer_fixes_v1.py`.

## Numerical results

The accuracy threshold is relative logit error below `0.001`, together with
finite outputs, loss and parameter gradients. The new reference uses the
intended FP64 angle formula for every arm; the bounded-frequency arm uses its
explicitly changed formula. All parameters remain FP32 and tied weights remain
the same parameter object. The [full records](results/numerical.json) retain
every case, including the remeasured original controls.

| Candidate | Finite and accurate / 65 |
|---|---:|
| Original | 44 |
| Safer RMS only | 44 |
| Wider rotary angles only | 53 |
| Wider SwiGLU only | 49 |
| Wider XSA only | 44 |
| All four numerical repairs | 63 |
| All repairs + bounded learned frequency | **65** |

The combined numerical repairs make every forward accurate, but two
`alpha=90` cases still overflow the FP32 frequency gradient: 2 layers/context
1024 and 8 layers/context 32. The bounded-frequency candidate removes these
remaining failures. Its maximum relative logit error is **4.2793e-6**. This
solves the declared finite numerical grid, not every possible finite-precision
failure. In particular, a bound on normalized hidden states is not a bound on
unconstrained tied readout weights or every learned Jacobian.

The [Lean fixture](StressFixes.lean) checks seven new statements: bounded
frequency range and derivative, preserved initial value/derivative, the
pairwise tangent identity, its division form for nonzero energy, and the
zero tangent numerator for a parallel upstream. The
[proof record](results/guarantees.json) and [checker output](results/lean-check.txt)
have zero warnings and only `propext`, `Classical.choice`, `Quot.sound`.
These are real-arithmetic statements; finite-precision behavior is measured
separately above.

## Learning results

The [complete learning comparison](results/learning-summary.json) has **7/84
threshold passes**: five with zero-initialized scalar gates and two with `1/D`
scaling. All other arms still miss. With the zero-initialized gate, the linear
path improves coverage from **0/12 to 5/12**, worst final relative MSE from
**0.26673 to 0.07459**, and median from **0.18443 to 0.06050**. It is the best
of this declared comparison, but seven cases remain unsolved. No aggregate
first-crossing mean is reported for incomplete coverage.

The deeper 16-branch model fits the teacher better; all six 4-branch cases
still miss. A separately frozen rate study in `../transformer_stress_rates/`
tests whether this is a fixed-budget optimization issue. These results do not
yet establish a full-GPT improvement or justify calling the architecture ideal.

The [frequency-gradient follow-up](../transformer_gradient_stress/README.md)
repairs a subtler failure that finite-gradient checks missed: native tanh
backward rounds tiny true derivatives to zero. All 18 scalar and 65 full-stack
checks pass with identical forward logits and unchanged other gradients.
The [activation follow-up](../transformer_stress_parity/README.md) improves the
original learning test to 10/12 passes but introduces mixed-teacher regressions.
These results preserve the distinction between numerical correctness and
successful learning across tasks.
