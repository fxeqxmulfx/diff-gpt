Complex multiplicative attention gate: validated; benchmark stopped by user

Plan: ../COMPLEX_MULTIPLICATIVE_GATE_PLAN.txt. Parent remains selected ungated
complex-peri-rms-ordinary-mhc4-fp16tf32-v1,with105independently audited passes.
The user stopped both earlier gates after negative short epoch results; their
partial long cohorts remain separate. This is the third requested gate variant.

The complex-linear predictor produces u+i*v. The applied gate is
sigmoid(u)*(1+i*v)/sqrt(1+v*v), multiplying the complex attention-value sum
before the original output projection/RMS. Initial A/B weights are zero,
giving0.5+0i gates. The predictor has L*D*D/2 independent real parameters in
L*D*D stored tied entries, using the existing per-head real/imaginary layout.
Phase equals atan(v); this is a bounded-phase design, not every complex map.

Exact-real local Lean algebra: ../proofs/ComplexAttentionGate.lean proves seven
statements for positive/bounded amplitude, nonzero denominator, unit phase,
gate norm identity, fixed-gate nonexpansion and local coefficient bounds.
No sorry or extra axioms. No formal differentiation, floating-point,
whole-network Jacobian or convergence theorem is claimed.

Build the scientific types:
  cargo build --offline --release --manifest-path research/main_model/multiplicative_gate_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose,fp32-gate

Gate operations/affines use static FP32; original affines retain FP16 forward/
TF32 backward. Scaled phase normalization avoids overflowing v*v. Sigmoid
and phase derivatives are factored before multiplying compensated operands.
No runtime dtype fallback or GPU FP64. Independent A/B gradients contract
before the unchanged global clip1/AdamW WSD; real/imaginary moments remain
separate. Preserve all original parameter bits/owner order and500-epoch saves.

Numerical command: BINARY gate-validation FRESH_DIRECTORY.
Validation:664implementation cases pass,including648primitive cases and
1400large-phase compensated-gradient controls. Six of189 strict scalar-AD
directions miss their unchanged bounds,all at large gate weights. Twelve CPU
tests,strict Clippy and four executed GPU tests pass. Both sanitizers report
zero errors; ungated-main full-state replay matches exactly. The all-affine-
FP32 diagnostic passes189/189 strict directions and rejects scientific
cell/resume/suite training. All19PTX modules contain no GPU FP64.

Frozen scientific executable:
  runs/main-model-multiplicative-gate-01/multiplicative-gated-transformer
The local Cargo target may contain the FP32 diagnostic; use the frozen binary.
Validation report: ../results/multiplicative-gate-validation-final/REPORT.txt.
Registration: ../results/multiplicative-gate-suite-01/registration.sha256.
Independent prelaunch audit:105unrun cells,zero failures. Benchmark launcher
session22080 started2026-11-08T12:02:39Z,75short followed by30long. Use the live
status file/handle at suite boundaries or on request, without minute polling.

Short result:75/75 independently audited passes,zero audit failures, but
1.1316263831850297 times main's epochs (13.1626% more;5earlier/20tied/50later).
The short epoch promotion gate fails. The user requested the next experiment;
long outcomes are4passed,one stopped,25unrun. Do not resume. No full105coverage
or promotion. See ../results/multiplicative-gate-suite-01/FINAL_RESULTS.txt.
Rust/Kernite/native tools only. No Python or subagents. Preserve all engineering
failures and numerical misses.
