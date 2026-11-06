# Existing-model gradient paths on cubic-sum

The user requested a diagnosis before considering larger models. The
[plan](plan.json) freezes arithmetic-sum-cube, all three seeds, four fixed
64-example training batches, and the existing initial/final model states.
The original baseline passes 3/3, original LR×2 passes 1/3, and the normalized
inverse-width LR×2 model passes 0/3. Both original rates are retained to expose
the learning-rate confound. All models have the existing one layer, width 128,
four heads, fixed RoPE, Q/K RMS and tied embedding/readout weights.

Observe every executed module and the functional attention/SwiGLU operations.
Measure actual task-loss gradients and a shared unit logit-probe VJP separately:
small training-loss gradients alone cannot diagnose a blocked gradient path.
Reconstruct residual identity/branch gradient contributions and the tied
lookup/readout gradient contributions. Instrumentation must preserve each
forward and every parameter gradient against an uninstrumented execution.

This is a retrospective description on training examples, with no optimizer
updates, new held-out scores, benchmark promotion, or larger-model training.
The saved final checkpoints have different training ages and confidence levels.
The observations cannot establish a causal explanation of their trajectories,
a worst-case Jacobian bound, or suitability at larger scale. Historical failures
and the ten unresolved numerical stress failures remain recorded.

Status: inputs, checkpoint hashes, scope and metrics declared before measurement.
