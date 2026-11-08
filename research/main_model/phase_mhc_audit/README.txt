Independent audit for phase-conjugated mHC on the selected LeRoPE main

This CPU-only Rust crate does not import the native engine or layout code.
It reconstructs complex parameter ties,base/router initialization,the6Lnew real
phase coordinates,the shifted frequency tail,samplers,observation intervals,
optimizer schedule,checkpoints and first99epochs. All shared initial weights,
including frequencies,match the pinned current-main audit. Additional capacity
must be exactly6real coordinates per layer. Phase coordinates use independent
real AdamW moments in the base group; frequencies retain separate clip1/wd0.

The candidate/source/PTX/plan pins match phase-mhc-validation-final. Every frozen
preflight task is checked before launch,including the35dataset fingerprints.
The suite is75short then30long,seeds0/1/2 and original budgets. Any lost main
pass rejects the candidate. Promotion additionally requires a geometric-mean
short epoch ratio below1. No partial-tier mean or throughput-based promotion.

Build: cargo build --offline --release --manifest-path research/main_model/phase_mhc_audit/Cargo.toml
Tests: cargo test --offline --manifest-path research/main_model/phase_mhc_audit/Cargo.toml
Run: main-model-phase-mhc-audit RUN_ROOT FRESH_OUTPUT.json
Fifteen tests and strict Clippy pass. Evidence and frozen auditor:
../results/phase-mhc-suite-01/. Earlier build/lint attempts remain in
../results/phase-mhc-preparation-01/. No Python or sub-agents.
