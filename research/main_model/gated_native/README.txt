Native attention-gate experiment, currently in pre-training validation

Parent: selected complex-peri-rms-ordinary-mhc4-fp16tf32-v1, independently
audited75short/30long passes. Model/current.sh still select that ungated parent.
Plan: ../GATED_ATTENTION_PLAN.txt. Queue: usual real sigmoid gate, complex-linear
predictor with real sigmoid, complex-valued multiplicative gate, then the two
rotary retests. This crate currently implements only the usual real gate.

Build the declared scientific candidate:
  cargo build --offline --release --manifest-path research/main_model/gated_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose,fp32-gate

The added per-layer D-to-D real bias-free gate predictor reads the normalized
mHC attention input and supplies elementwise sigmoid gates before the original
output projection and branch RMS. Zero-initialized gate weights add L*D*D real
parameters without changing the original initialization. Every new parameter
participates in the existing independent-coordinate clip/AdamW recipe.

Gate affine forward/backward operations use fixed FP32. All original affine
operations retain FP16 forward/TF32 backward and FP32 accumulators. Sigmoid
products and adjoints use exp(-abs(logit)/2) factors to avoid amplifying the
rounding of a subnormal stored probability. No runtime dtype fallback/FP64.

Numerical command (fresh output directory required; GPU access required):
  BINARY gate-validation FRESH_DIRECTORY
It archives compiled sources and reports primitive checks, every gate matrix
and input-adjoint coordinate, full-model scalar AD directions, gradient resets
and causal/prefix invariance. Results retain every failed strict comparison.
Other inherited commands include ptx-audit, stress, preflight, archive and cell.
Never start cell/suite training without the separate frozen experiment/auditor.

The fp32-affine-reference feature is a diagnostic of ALL affine operations,
including the original model. It is not the scientific candidate; its CLI
rejects cell/resume/suite. The pure FP32 diagnostic passes189/189 AD directions.
The chosen mixed model with FP32 gates retains9strict large-gate-weight misses;
see ../results/gated-validation-03 and04-reference. Full-suite learning has not
started, so no improvement or retained105/105 claim is yet available.

Rust/Kernite/native tools only. No Python or subagents. Keep500-epoch buffered
saves and original data/seeds/budgets/observations. Preserve failed results.
