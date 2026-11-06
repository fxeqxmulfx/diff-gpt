# AdamW response to gradient scaling

The initialization audit measures raw gradients. Native AdamW also uses their
retained first and second moments, so a smaller raw gradient need not produce
a proportionally smaller update. This [plan](plan.json) checks the missing link.

Extend the existing first-step scaling identity to full supplied histories and
the native parameter recurrence, preserving bias correction, finite optimizer
epsilon, the zero-rate first update and decoupled weight decay. The entire
input history must be multiplied by the same positive factor. A measured ratio
of gradient RMS values between two model blocks does not establish that premise.

The complete 18-case synthetic CPU grid has finished, with two native traces
of 64 updates per case. Every parameter and moment vector is compared to
80-digit Decimal arithmetic and the scaling relation is checked directly.
The declared 0.001 relative-error or exact-zero criterion remains unchanged.

The model studies remain frozen. This supplement neither selects an initializer
nor predicts convergence; a model intervention can change gradient directions
and future gradients. Optimizer epsilon `1e-8` is separate from RMS epsilon
`2^-23`. CPU arithmetic checks do not certify fused or captured CUDA execution.

The [three statements](Bounds.lean) compile with no warnings or extra axioms;
the failed record-equality elaboration attempt is preserved. For a supplied
history `g`, a constant `c>0` and optimizer epsilon `eps>0`, the exact result is
`AdamW(c*g, eps) = AdamW(g, eps/c)` for the parameter path when initial weights,
rate schedule and decay agree. The corresponding moment paths are multiplied
by `c` and `c²`. This includes the zero-rate first update's moment insertion.

The [reference checks](results/reference-validation.json) pass both independent
cases: a constant-gradient closed-form trajectory with varying warmup rates,
and an exact rational signed-input example with zero betas. Commit `e707f30`
froze all validated sources and evidence before native execution.

The [complete audit](results/complete-audit.json) reconstructs every represented
input, all Decimal states and all **10,368 vector comparisons**. All comparisons
and **18/18 cases pass**, with no execution errors or nonfinite outputs. The
direct scaling identities agree exactly on this power-of-two scale grid.

| CPU dtype | Cases passed | Maximum relative error against Decimal |
| --- | ---: | ---: |
| FP32 | 9/9 | 1.6375021e-6 |
| FP64 | 9/9 | 1.8832000e-15 |

All [raw vectors](results/measurements.json), [case rows](results/cells.csv) and
[individual comparisons](results/comparisons.csv) are retained. There are 2,304
synthetic native optimizer updates, zero model training updates and zero task
scores. The source archive and individual raw cases are under
`runs/adamw-gradient-scaling/optimizer-traces-v1`.

For interpretation, the existing first-step formula is
`u(g) = g/(|g|+eps)`. With the same epsilon, the direction ratio at nonzero `g`
is `u(c*g)/u(g) = c*(|g|+eps)/(c*|g|+eps)`. The following are analytic
illustrations for `c=1/8`, `eps=1e-8`, before multiplication by the learning rate:

| Original gradient magnitude | Adaptive direction ratio after dividing gradient by eight |
| --- | ---: |
| 1e-12 | 0.12501094 |
| 1e-8 | 0.22222222 |
| 1e-4 | 0.99930056 |
| 1 | 0.99999993 |

Thus an eightfold gradient reduction at magnitude `1e-4` changes this initial
adaptive direction by only about **0.070%**. Near optimizer epsilon, suppression
can remain substantial. These are nominal scalar calculations, not new model
measurements; the frozen benchmark's first learning rate is zero even though
the moments are updated. [All 12 illustrations](results/first-direction-illustrations.csv)
are saved separately from the native measurements.

The earlier model Q/K gradient gap therefore cannot by itself quantify an
AdamW update gap or establish an initialization defect. Direction, moment
history, epsilon and parameter-relative updates also matter. The usual-gain
experiment remains informative because it changes the attention function and
the proven score envelope as well as raw gradients. Finish the capacity/gain
comparisons before selecting an isolated initialization intervention.

Reconstruct the report without executing another optimizer trace:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.adamw_gradient_scaling_reporting.audit
```
