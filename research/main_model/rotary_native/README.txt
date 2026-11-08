Shared learned rotary frequencies on the selected complex mHC main

Both rotary cohorts completed105/105. LeRoPE is promoted; Log-Scaled LeRoPE
is rejected for more short epochs. Main stays LeRoPE. The log cohort is
runs/main-model-log-lerope-01; launcher92108 and terminal waiter68339 exited0.
Independent registration: ../results/log-lerope-suite-01/registration.sha256.
Log-Scaled:6.2493% more short epochs (6/55/14 earlier/tied/later),14.1047%
more long (12/0/18);zero audit failures. All32regressions remain recorded.
Report: ../results/log-lerope-suite-01/FINAL_RESULTS.txt. Do not restart.
All three attention-gate cohorts are closed by explicit user stops. None is
promoted. The LeRoPE cohort used complex-peri-rms-ordinary-mhc4-fp16tf32-v1
as control; the next uses complex-peri-rms-ordinary-mhc4-lerope-fp16tf32-v1.
Compiled plan: ../ROTARY_RETEST_PLAN.txt; next: ../LOG_SCALED_LEROPE_PLAN.txt.

For position p and rotary band m, the angle is
  p * 10000^(-2m/d) * exp(alpha_m).
Log-Scaled LeRoPE additionally multiplies each position's angle by ln(p+1).
There are d/2 shared alpha parameters across Q/K,all heads and all layers.
They start at zero without RNG draws. The original common parameter bits and
optimizer-owner order remain unchanged. Ordinary LeRoPE at zero alpha must
match the selected RoPE initialization exactly. Log scaling changes that
initialization's positional outputs and is a separate candidate.

Original parameters retain the selected global independent-coordinate clip1,
AdamW(.9,.95),wd.1,LR.001 and2%warmup/20%cosine schedule. Alpha has a separate
clip1 group,weight decay0,and the same schedule and moments. The legacy
Engine.update path rejects learned rotary. Checkpoints contain every alpha
parameter and moment, with buffered saves every500epochs and at first99,
budget completion or explicit STOP.

GPU tables and their adjoints use fixed FP32. The table is regenerated every
forward, including after parameter changes or checkpoint restore. Only the
position-dependent base tables are cached. A fixed FP32 exponential residual
and compensated base*expm1 phase preserve tiny updates and avoid cancellation
when frequencies decrease. There is no runtime precision fallback or GPU
FP64. Original affine math remains FP16 forward/TF32 backward,FP32 accumulation.

Build scientific candidate:
  cargo build --offline --release --manifest-path research/main_model/rotary_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose
Validate a frozen executable:
  BINARY rotary-validation FRESH_DIRECTORY
  BINARY ptx-audit FRESH_DIRECTORY
The fp32-affine-reference feature is diagnostic only; cell/resume/suite are
disabled. Its compiled target must never be mistaken for the scientific binary.

Current validation revision02: all360 implementation cases pass. These cover
252table cases,96input/frequency adjoint cases,six initialization integrations
and six causal/prefix integrations. Five of160 strict full-model frequency AD
directions remain outside their original bounds; the FP32 diagnostic passes
160/160. Both sanitizers are clean and unchanged-main replay matches exactly.
Eleven CPU tests,strict Clippy and five executed
GPU tests pass. The latter include192independent frequency optimizer cases,
unchanged common-group state,checkpoint continuation and maximum suite shapes.
The21table failures from revision01 and earlier build errors remain recorded.

Six existing Lean statements were rechecked from the transformer project's
LeRoPE source: positivity,zero initialization,positive-frequency coverage,
multiplicative increments and local frequency/score derivatives. Only standard
Lean axioms are used. Evidence: ../results/rotary-preparation-01/lean/ and
lean-check.log. These exact-real statements do not prove floating-point
accuracy or convergence. The table screen uses six magnitude patterns with
|alpha|<=4 and positions
through1024; no arbitrary-frequency or arbitrary-length precision guarantee.

Frozen executable: runs/main-model-lerope-01/rotary-transformer. The local
Cargo target currently contains the FP32 diagnostic; use the frozen binary.
Registration: ../results/lerope-suite-01/registration.sha256. Independent audit
verified105unrun cells and zero failures before launch. Launcher98943 started
2026-11-08T12:40:30Z. Short audit:75/75passes,zero failures,0.9946609165143937
times the prior main's epochs (0.5339% fewer;7earlier/65tied/3later).
The30/30long result has1.0150994601681524 times prior main's epochs
(1.5099% more;12earlier/4tied/14later). Full report:
../results/lerope-suite-01/report-final/REPORT.txt. The registered short-epoch
and coverage rule qualifies promotion; retain all17individual regressions.
Do not restart this completed cohort. Wait for declared experiment boundaries
or live handles without minute metric polling during the next experiment.
No lost main pass is allowed. Report all per-cell epochs and retain inherited
strict AD/FP64 optimizer misses. Rust/Kernite/native tools only.
