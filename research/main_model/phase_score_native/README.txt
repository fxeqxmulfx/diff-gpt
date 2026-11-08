Squared-imaginary attention score candidate

This isolated Rust/Kernite candidate extends the selected phase-mHC LeRoPE
model with one zero-initialized real score coordinate per layer/head:
ln(N)/d * (Re(Z) + beta*Im(Z)^2/d), Z=sum(conj(q)*k), beta=a/sqrt(1+a*a).
d is real head width; N is the visible causal prefix. Every existing model
component and ordinary AdamW/WSD setting is retained. New coordinates use the
base clip1/wd0.1 group; frequencies retain separate clip1/wd0. Singleton
optimizer ownership handles odd coordinate counts without a padding parameter.
The exact predeclared experiment is in ../COMPLEX_PHASE_SCORE_PLAN.txt.

Build from repository root, with no Python:
cargo build --offline --release --manifest-path research/main_model/phase_score_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose
cargo test --offline --release --manifest-path research/main_model/phase_score_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose
GPU tests are ignored by the ordinary test command; execute explicitly with
--ignored --test-threads=1. phase_score_tests contains independent primitive,
full-model directional and optimizer references. The diagnostic feature
fp32-affine-reference fixes every affine to FP32 and refuses scientific cells.

Native validation passes4800primitive cases,990CPU finite differences,
112optimizer cases,checkpoint continuation,maximum suite shapes and both
CUDA sanitizers. All22generated modules contain noGPUFP64. Preserve147strict
mixed-precision AD misses including6score directions. The fixed FP32 diagnostic
passes26/26score directions and retains24near-zero router-read misses.
These finite tests and the14exact-real Lean statements are not a universal
numerical or convergence guarantee. All parent numerical limitations remain.

The generic runner exposes six engineering arms. Scientific experiments use
only phase-score-lerope via ../run_phase_score_suite.sh and its independent
../phase_score_audit. Other arms are controls,not additional scientific cells.
The fallback profiling command accepts only phase-mhc-lerope and replays the
selected main's trained checkpoint into disposable state without saving it.
Frozen scientific executable,provenance and validation report:
../results/phase-score-validation-final/.
