# Preserving bounded-frequency gradients

The earlier [numerical fixes](../transformer_stress_fixes/README.md) passed all
65 finite-output/gradient checks. A closer derivative check found a limitation:
at raw frequency parameter 90, native FP64 `tanh` backward gives zero for
`exp(4*tanh(alpha/4))`, although its real derivative is about `6.25153e-18`.
A finite gradient alone did not establish an accurate gradient.

The [separately frozen plan](plan.json), source commit `ed3c65f`, keeps the
forward expression unchanged and evaluates its derivative as

```
r = exp(-2*abs(alpha/4))
derivative = exp(4*tanh(alpha/4)) * 4*r/(1+r)^2
```

This avoids subtracting nearly equal numbers in `1-tanh(alpha/4)^2`. It adds
no parameters and changes neither the frequency cap nor other model operations.

The complete [results](results/summary.json) pass the declared checks:

- All **18 scalar probes** agree with an 80-digit Decimal reference to relative
  derivative error at most **5.30271e-8**, versus up to 1 for the parent.
- All **65 full-stack cases** have identical forward logits and identical
  gradients for every other parameter. All corrected gradients are finite.
- All five `alpha=90` full-stack cases recover their frequency gradients;
  their norms range from `7.82623e-20` to `4.27795e-17`.

The [audit](results/audit.json) verifies the complete case grid, source snapshots
and summaries, and reproduces all scalar cases. It does not claim a second
execution of the full-stack grid. Raw records and source copies remain under
`runs/transformer-stress/gradient-v1/`. Six tests check the scalar calculus and
first/second finite-difference derivatives around ordinary inputs.

The [Lean fixture](FrequencyGradient.lean) proves five real-arithmetic facts:
the exponential tanh identity, both signed and absolute-value expressions for
`1-tanh²`, the exact bounded-frequency derivative, and strict positivity of that
real derivative. Its [proof record](results/guarantees.json) and
[checker output](results/lean-check.txt) have zero warnings and only the standard
`propext`, `Classical.choice`, and `Quot.sound` axioms.

This is a numerical repair, not a new learning result. Extremely small correct
gradients can still produce updates below the parameter's representable spacing.
The real positivity theorem does not guarantee nonzero floating-point gradients
at every possible input. The separate residual learning tests still have six
shallow failures; those failures are preserved.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.transformer_gradient_stress.run
PYTHONPATH=. .venv/bin/python research/stress_audits/gradient.py
.venv/bin/pytest -q tests/diff_gpt/application/test_transformer_gradient_stress.py
```

The experiment refuses to overwrite an existing output directory. Earlier
scientific sources and measurements remain unchanged.
