# Two adjacent RMS normalizations

This supplement studies the actual embedding-RMS followed by first-block input
RMS, with the same positive epsilon and channel width. The [plan](plan.json)
keeps the running architecture study unchanged.

For `R_e(x) = x / sqrt(mean(x^2) + e)`, the existing checked one-operation
Lipschitz bound is `1/sqrt(e)`. Five checked statements establish the two-operation
bound `1/e`, its exact derivative at zero along unit-RMS directions, and the
finite-epsilon scalar composition identity. These are sensitivity statements
about adjacent normalizers, not about the full attention/FFN or residual network.

The source definitions are `Transformer.GPTMini.rmsNormEps` and the existing
`Transformer.Residual.radialRMS`, whose unit-direction mapping is already proved.
The real-arithmetic identity keeps epsilon; no ideal scale invariance is assumed.
Arithmetic rounding, optimization and explanations of observed failures remain
outside this proof. No learned checkpoint or held-out outcome determines a bound.

For a unit-RMS direction `v` with `||v||^2 = d`, put
`phi_e(a) = a/sqrt(a^2+e)`. The exact statements include

```text
phi_e(phi_e(a)) = a / sqrt((1+e)*a^2 + e^2)
R_e(R_e(a*v)) = phi_e(phi_e(a))*v
||R_e(R_e(x)) - R_e(R_e(y))|| <= ||x-y||/e
d/da R_e(R_e(a*v)) at a=0 = v/e.
```

At the actual FP32 epsilon `e = 1/8388608`, one map's zero-input sensitivity is
approximately **2896.3094**; the adjacent pair's is exactly **8388608**. The
directional derivative reaches the displayed global-bound coefficient at zero.
Thus adding another normalization can increase sensitivity near zero even while
its outputs remain bounded. This does not show that the measured training runs
operate near zero or that the composition caused their learning failures.

Only the first branch's input follows this adjacent composition. The embedding
skip and tied readout have other paths; their derivatives are outside this result.
The initial [scale audit](../core_initial_scale/README.md) measured embedding RMS
around 0.02 before normalization, so the zero-input calculation must not be
presented as a measurement of those initialized embeddings.

The [Lean record](results/guarantees.json) pins the checked source and imported
library. All five statements and concrete positive-parameter examples compile
without warnings and use only standard axioms. The initial vector-scalar type
inference error, its source and both compiler outputs are retained in
[attempt notes](results/attempt-notes.json).

The [predeclared numerical grid](runtime-plan.json) checked the existing RMS
implementation at widths 1/64/128/256, amplitudes 0/2^-20/0.02/1, and CPU FP32/FP64.
Both precisions use the benchmark's FP32 epsilon. Independent binary64 formulas
check each single/double forward value and directional gradient at the actual
represented input. **32/32 cases pass**, with maximum relative error
**2.38419e-7**, below the fixed 0.001 threshold; exact-zero references also match
exactly. [All observations](results/runtime.json) and [output](results/runtime-output.txt)
are retained. This is a local numerical check; the ten unresolved full-stack
stress failures remain unchanged.

```bash
# From /home/misha/lean_projects/transformer:
lake env python3 /home/misha/python_projects/gpt/research/rms_composition_guarantees/check.py
# From the GPT repository:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.rms_composition_guarantees.check_runtime
```
