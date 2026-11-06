# Finite-epsilon LayerNorm and affine normalization bounds

The user keeps Peri-normalization, embedding normalization and tied weights, while
leaving RMSNorm, LayerNorm and learned parameters open. This [plan](plan.json)
examines those choices locally without selecting or training a new architecture.

Use the actual population-variance LayerNorm equation: subtract the coordinate
mean, then apply finite-epsilon RMS. Prove its radius, input-sensitivity and
constant-shift properties. For either normalizer, make the learned diagonal
scale and bias explicit in the output and input-sensitivity bounds.

After the proofs, a declared 160-case CPU comparison will check existing RMS and
native LayerNorm outputs and input/affine gradients against independent 80-digit
Decimal formulas. It includes singleton and constant inputs, and retains every
numerical failure under the fixed 0.001 relative-error threshold. No model change,
optimizer update or task evaluation is part of this supplement.

Status: plan declared; proofs and local numerical comparison pending. The running
capacity grid, queued usual-gain test and deferred full-stack repairs stay intact.
