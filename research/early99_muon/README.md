# Muon on the unchanged inverse-width model

The [frozen plan](plan.json) compares four optimizer configurations. Hybrid Muon
updates hidden matrices while native AdamW updates the tied embedding/readout
once. Pure Muon updates every unique matrix, including that tied weight, without
an AdamW fallback. The existing inventory confirms that all35 case configurations
have only matrix parameters.

Each partition is tested at matrix LR×2 and LR×4. Hybrid auxiliary AdamW stays at
LR×2, betas(.9,.98) and epsilon1e-8. Every group keeps the original50-update warmup
from zero, then a constant rate and the original case weight decay. Matrix
direction scaling is `0.2*sqrt(max(rows, columns))`; decay uses the unadjusted rate.
These are optimizer-only experiments on the original fixed-RoPE, Q/K-normalized
model. Cosine and model numerical repairs are separate studies.

The matrix rule uses momentum.95, Nesterov and five quintic Newton–Schulz steps,
with coefficients(3.4445,-4.775,2.0315). Momentum and iterations are FP32; the
initial Frobenius normalization uses FP64 and denominator `max(norm,1e-7)` before
casting to FP32. This explicitly declared precision variant avoids overflow in
the initial norm of finite FP32 gradients. It is not native Torch's BF16 iteration
and does not compute an exact polar factor. Matrix momentum uses the actual
native EMA convention; the complete equations and operation order are in the plan.

The [upstream recipe](https://github.com/KellerJordan/Muon) recommends Muon for
hidden weights and AdamW for other parameters. Applying Muon to the tied
embedding/readout is the user's requested ablation. Neither partition is assumed
better before measurement. The matrix learning-rate grid uses the RMS-matching
shape convention and a doubling around the incumbent rate.

Validate matrix updates against an independent FP64 SVD/scalar-polynomial
reference, partition/RNG/state behavior, CPU/CUDA restoration and the unchanged
trainer protocol before any benchmark update. Compile the precisely scoped real
proofs and pin all source and evidence. Then complete all four18-cell screens;
each requires18/18 and at least3improved parent task medians to advance. Every
advancing arm must complete75short cells. Only a newly frozen short winner gets
one reused30-cell long diagnostic. Replay work is counted by the existing durable
journal. Retain every miss, error, prior negative result and numerical limitation.

The inverse-width complete-model stress result is still40/50. Its five extreme
QKV inaccuracies and five nonfinite SwiGLU cases are deferred until the requested
optimizer comparisons finish. Wider optimizer normalization does not repair those
model failures or prove transformer convergence.

Implementation validation passes54checks:39Muon checks,6unchanged execution-journal
checks and9frozen benchmark-integrity checks. The two real CUDA configurations
cover hybrid/head-width8 and pure/head-width32, compiled versus eager losses,
tail batches, all optimizer state, full-split scoring and nonempty-state restore.
The independent matrix reference uses FP64 SVD and a scalar polynomial recurrence,
covering zero through1e30 gradients and rank-deficient/tall/wide matrices. Complete
CPU trainer resumes reproduce weights, moments, sampler and RNG exactly.

Initial validation failures are retained: importing the lab sampler before the
existing bootstrap; a journal callback with the wrong signature; and the tensor
test helper refusing string partition metadata. The final tests compare all
nonnumeric metadata exactly and every numerical state, without weakening the
matrix or update tolerances. No benchmark updates occurred during these fixes.

[Checked Lean statements](Guarantees.lean) prove that the exact quintic preserves
the interval[0,5/4], using explicit Bernstein certificates, and that every finite
scalar iteration from[0,1] stays there. The same coefficients map1to0.701 in one
step. Three existing matrix statements were separately checked: the conditional
SVD polynomial identity, zero preservation, and the1×1non-polar counterexample.
All six printed theorem dependencies use only standard axioms; final compilation
has no warnings. The old zero-norm normalization theorem is not used to claim
scale invariance below this implementation's epsilon floor. These are real scalar
bounds and conditional matrix algebra, distinct from the finite-precision checks
and the forthcoming learning measurements.
