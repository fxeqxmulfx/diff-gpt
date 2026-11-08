Complex-linear attention gate predictor: frozen benchmark running

Plan: ../COMPLEX_GATE_PREDICTOR_PLAN.txt. Parent is the selected ungated
complex-peri-rms-ordinary-mhc4-fp16tf32-v1, with105independently audited passes.
The usual real gate was stopped by the user after its negative short result;
all its data remain separate. This is the next explicitly requested variant.
The frozen105-cell cohort is runs/main-model-complex-gate-01. Its registration
audit passed before launch; results live under ../results/complex-gate-suite-01.

The predictor ell=(A+iB)z uses D*D/2 independent real weights per layer, stored
in D*D expanded entries with exact [[A,-B],[B,A]] ties in the existing head
layout. Separate real sigmoid gates multiply the real and imaginary attention
output components. The sigmoid gate as a whole is not phase-equivariant.
Zero A/B initialization gives0.5 gates with no compensation or RNG draws.

Build:
  cargo build --offline --release --manifest-path research/main_model/complex_gate_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose,fp32-gate

Gate affines are static FP32; original affines stay FP16 forward/TF32 backward.
Contract A/B adjoints before the unchanged global independent-coordinate
clip1 and AdamW/WSD update. Preserve all original owner order, initialization,
normalizations, tied embeddings, mHC, data/seeds/budgets and500-epoch saves.
Real and imaginary second moments remain separate; complex AdamW is queued
as a different experiment. No dynamic precision fallback or GPU FP64.

Numerical command: BINARY gate-validation FRESH_DIRECTORY.
All124 implementation cases,12CPU tests,four executed GPU tests,sanitizers and
exact ungated control replay pass. Five of189 strict scalar-AD directions
miss their bounds; keep these failures. The all-affine-FP32 diagnostic passes
189/189 and cannot train. Evidence: ../results/complex-gate-validation-final.
No scientific learning result or improvement is claimed yet. Use the frozen
cohort executable; the local Cargo target may contain the diagnostic build.

Rust/Kernite/native tools only. No Python or subagents.
