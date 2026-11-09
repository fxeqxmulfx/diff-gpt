Old MASH-half Fibonacci model: fresh Rust/Kernite recheck, 2026-11-09

The historical architecture and training recipe solve the one-feature task
in all three fresh native runs. Each completed the predeclared5000updates:

  seed   first observed90%   final correct/4704   final decoded accuracy
    11               1300                4468                94.982993%
    17               1500                4386                93.239796%
    23               1150                4344                92.346939%

Saved final checkpoints reproduce every training/validation metric exactly.
All101observations per seed are retained; first90 is observed every50updates
and is not a sustained-accuracy claim. No retries, retuning or budget changes.
Training and validation cycle splits, encoded examples, decoder bases and all
2560000 sampled training IDs per seed match the prior failed native DiffGPT
cohort by SHA256. New runs: runs/diffgpt-old-recheck-01/seed{11,17,23}.
Summary and validation evidence: research/diffgpt_old_recheck/results/.

This is a fresh architecture/recipe recheck, not a bit-exact replay of the
historical1050/1150/1450update ledger at commit90544cd. Native Gaussian RNG and
draw ordering differ from Torch; the matched data/sampler are the newer native
experiment's. The XSA implementation uses an algebraically equivalent centered
form, exact-zero guards and corrected adjoints, rather than reproducing Torch's
cancellation roundoff. Initial output projection scales match the old model.

Architecture: D128/H1/L1/V512,262016real parameters, hidden341 SwiGLU,
parameter-free Peri/embedding/final/per-head QK RMS, tied embedding/readout,
real XSA, standard RoPE, attention1/sqrt(128), residual gain1.
Recipe: ScheduleFreeAdamW,lr0.03,betas(.9,.98),wd1,eps1e-8,warmup10,no clipping.
Training holds query y; evaluation and saved weights hold average x. Preserve
base z, second moment v and optimizer mode/counters in every checkpoint.
All GPU arithmetic is fixed FP32; no Python, LibTorch or GPU FP64 execution.
The current production DiffGPT remains2D-only; this executable is an isolated
historical1Dcontrol, not a change to the production positional policy.

Before scientific registration, validation passed:
  24 model gradient directions against independent CPU FP64 forward-mode AD.
  Maximum forward logit error5.116e-7; maximum directional error6.140e-8.
  24 ScheduleFree steps versus an explicit CPU (x,z,v) recurrence,
    including mode switches; maximum state error2.194e-7.
  Serialized checkpoint continuation is bit-exact.
  CUDA memcheck and synccheck each report zero errors.
  All13embedded PTX modules have no GPU FP64 instructions.
  Rust formatting and strict Clippy passed.
The first validation and sanitizer passes are retained; final ones also cover
a Clippy-driven equivalent checkpoint-decoding iterator change. These finite
checks are not a universal numerical or convergence guarantee.

Interpretation
The dataset is solvable by the older design with its historical recipe. The
previous additional DiffGPT failures therefore warrant diagnosis. This result
does not isolate architecture as the cause: the newer model used AdamW/WSD
atLR.001,betas(.9,.95),wd.1,clip1 and mixed tensor precision. The old and new
runs also differ in parameterization, capacity and initialization. No new-model
change or promotion was made here, and the earlier36failures remain recorded.

Build:
  cargo build --offline --release --manifest-path research/diffgpt_old_recheck/native/Cargo.toml
  cargo clippy --offline --release --manifest-path research/diffgpt_old_recheck/native/Cargo.toml -- -D warnings
Commands: validate FRESH_DIR; run REGISTERED_ROOT SEED; replay CELL_DIR;
build-info. Training refuses an unregistered/mismatched binary or changed
source, data or sampler, existing cell outputs, and an explicit STOP file.
History/checkpoints are buffered. Save at initialization, first90,500epochs,
budget completion or STOP; periodic writes are never per-update.
Source and protocol remain in git scope; regenerable evidence and runs do not.
