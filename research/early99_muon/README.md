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
