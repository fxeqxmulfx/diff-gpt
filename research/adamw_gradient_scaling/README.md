# AdamW response to gradient scaling

The initialization audit measures raw gradients. Native AdamW also uses their
retained first and second moments, so a smaller raw gradient need not produce
a proportionally smaller update. This [plan](plan.json) checks the missing link.

Extend the existing first-step scaling identity to full supplied histories and
the native parameter recurrence, preserving bias correction, finite optimizer
epsilon, the zero-rate first update and decoupled weight decay. The entire
input history must be multiplied by the same positive factor. A measured ratio
of gradient RMS values between two model blocks does not establish that premise.

After proof and independent-reference validation, run all 18 declared synthetic
CPU traces, with 64 updates each. Compare every parameter and moment vector to
80-digit Decimal arithmetic and check the scaling relation directly. Keep every
failure under the declared 0.001 relative-error or exact-zero criterion.

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
and an exact rational signed-input example with zero betas. All synthetic sources
and this evidence must be committed before the 18 native traces execute.
