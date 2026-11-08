Complex-pair AdamW on the selected LeRoPE model

Plan: ../COMPLEX_PAIR_ADAMW_PLAN.txt. Scientific candidate arm pair-lerope;
control arm lerope. Shared variance uses(g_re^2+g_im^2)/2 after independent
gradient contraction and global clip1. Pure real router coordinates remain
ordinary AdamW; alpha retains its independent clip1,wd0 real group.
All architecture,parameter initialization and WSD settings stay unchanged.

Validated frozen executable: ../results/pair-adamw-validation-final/candidate.
All714new checks,12CPU tests,six executed GPU tests,10local Lean statements
and both sanitizers pass. Inherited numerical misses remain. Exact evidence:
../results/pair-adamw-validation-01/REPORT.txt and ../results/pair-adamw-preparation-01/.
Scientific training started2026-11-08T14:16:15Z in runs/main-model-pair-adamw-01,
launcher3084. The independent registration audit verifies105unrun cells and
zero failures;14auditor tests and strict Clippy pass. Registration:
../results/pair-adamw-suite-01/registration.sha256. Run75short then30long,
rejecting any lost selected-main pass. Training completed2026-11-08T14:45:20Z:
105/105passes,zero audit failures,but short epochs increase2.4349% and long
11.7655%. This fails the promotion condition; main stays LeRoPE. All28epoch
regressions remain. Report: ../results/pair-adamw-suite-01/FINAL_RESULTS.txt.
All live handles exited0/consumed. Do not restart this cohort.

Build: cargo build --offline --release --manifest-path research/main_model/pair_adamw_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose
GPU tests: set PAIR_VALIDATION_OUTPUT to a fresh evidence directory and run
the frozen native-tests executable with pair_optimizer_tests:: --ignored
--test-threads=1 --nocapture. Keep GPU jobs sequential.
The fp32-affine-reference build remains diagnostic only and cannot train.
No Python. Buffered scientific saves remain every500epochs plus boundaries.
