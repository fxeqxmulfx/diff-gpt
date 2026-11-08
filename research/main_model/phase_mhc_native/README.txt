Phase-conjugated mHC on the selected LeRoPE main

Plan: ../COMPLEX_PHASE_ROUTING_PLAN.txt. Scientific arm phase-mhc-lerope;
architecture complex-phase-mhc-lerope. Three zero-initialized real parameters
per sublayer define u=(1+i*a)/sqrt(1+a*a),with stream0 fixed to1. Skip mixing
becomes D P D^*,using the existing real Sinkhorn P and unchanged read/write gates.
Here D^* means conjugate transpose; the explicit equation in the plan
is y_i=sum_j P_ij*u_i*conjugate(u_j)*x_j plus the unchanged scaled branch.
The selected ordinary real-coordinate AdamW/WSD recipe is retained.

The selected main executable is frozen in ../results/phase-mhc-validation-final/.
Its independently registered suite completed75/75short and30/30long passes in
runs/main-model-phase-mhc-01,with zero audit failures. Geometric-mean epochs
decrease1.8747% short and11.5692% long versus the LeRoPE parent. All14individual
regressions remain recorded. Promotion under the standing user rule:
../results/phase-mhc-suite-01/promotion/decision.json. Complete paired results:
../results/phase-mhc-suite-01/report-final/paired-epochs.tsv.
current.sh selects this exact frozen executable and arm phase-mhc-lerope.
This is empirical first99 coverage,not sustained99 or a numerical proof.
Ten local Lean statements pass with only standard axioms and no warnings:
../proofs/ComplexPhaseRouting.lean. They prove exact-arithmetic phase
normalization,energy-bound equivalence with the fixed underlying mixer,
row-norm bounds,the preserved phase-aligned mode,and failure of ordinary
constant-stream conservation. They do not bound the full nonlinear Jacobian,
prove exact double stochasticity of finite Sinkhorn,or certify FP32 convergence.

Thirteen CPU tests pass,including independent parameter ownership and a
scalar-dual-number check of all67 input/branch/gate/phase adjoints of a small
complex skip map. All1440supplementary normalized-state cases,540independent
CPU finite differences,96optimizer cases,checkpoint continuation,maximum shapes,
memory/synchronization sanitizers and unchanged-control replay pass.
Factoring shared edge coefficients outside the gate dot product repairs the
8normalized-state misses of the initial implementation; zero-phase existing
logits and gradients remain bit-exact to main. Only the phase module PTX changes.
The original extreme-amplitude screen retains189failed cases out of1920.
Full-model strict native AD retains203mixed-precision misses,including one
phase direction. The fixed FP32 diagnostic passes48/48phase directions but
retains147near-zero router misses. All failed checks remain failures; neither
the exact-real proofs nor these finite screens certify all FP32 inputs.
The independent auditor passes15tests and the prelaunch105cell audit has zero
failures. It verifies the exact6Lcapacity increase and shared full initialization
including shifted frequencies,while retaining the original task/seed budgets.
All failed preparation attempts remain in ../results/phase-mhc-preparation-01/.

Build: cargo build --offline --release --manifest-path research/main_model/phase_mhc_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose
GPU tests: set PHASE_VALIDATION_OUTPUT to a fresh existing evidence directory;
run the frozen native-tests executable with phase_routing_tests:: --ignored
--test-threads=1 --nocapture. Keep GPU jobs sequential and use native tools only.
Do not train the all-affine-FP32 diagnostic build. No Python.
Scientific writes remain buffered every500epochs plus required boundaries.
