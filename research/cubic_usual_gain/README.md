# Cubic-sum test of the usual attention gain

The user requested a retest with `1/sqrt(d)`. The [plan](plan.json) declares
three seeds of the full normalized SSMax model at the original one-layer,
width-128 size. Only the attention gain changes from `1/d` to the parent's
existing `exp(log_alpha)/d`, which represents `1/sqrt(d)`. SSMax's `log(N)`,
Q/K RMS, fixed RoPE and every other block remain intact. The original LR1,
original LR2 and inverse-width controls are reused, including all failures.

Run this after the complete [depth/width study](../cubic_capacity/README.md)
has been audited and committed. The capacity grid remains unchanged. Validate
and freeze the exact implementation before these three training runs.

This tests a mechanism suggested by the weaker Q/K gradients and nearly uniform
attention observed in the existing-model diagnostic. It does not yet establish
that scaling causes the generalization failure. The one-task result cannot
select a new winner of the fixed 75-cell benchmark.

Status: declared; implementation and validation pending. No new training started.
