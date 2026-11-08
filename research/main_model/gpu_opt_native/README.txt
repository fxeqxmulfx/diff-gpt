Selected GPU runtime for the mHC architecture

Selected model: complex-peri-rms-ordinary-mhc4-fp16tf32-v1. The completed
gpu-suite-01 cohort passes75/75short and30/30long, with zero independent audit
failures. Geometric-mean epochs improve1.7953% short/11.5413% long versus the
previous mHC main. current.sh uses the immutable results/gpu-opt-05/combined
executable. Full report and selection evidence: results/gpu-suite-01/.

Rust/Kernite only. The fixed numerical configuration used for GPU optimization
is FP16 forward affine products, TF32 backward affine products, and FP32
accumulation, other model operations, master weights and optimizer states.
There is no runtime precision fallback. Alternate BF16-forward and FP32-backward
builds remain available as separate compile-time features.

Build the measured configuration:
  cargo build --offline --release --manifest-path research/main_model/gpu_opt_native/Cargo.toml --features tf32-backward,fused-attention,tiled-transpose

The immutable measured executables and their compiled sources are under
research/main_model/results/gpu-opt-01 through gpu-opt-05. Each iteration has
raw timings, state hashes, gradient reports, sanitizer output and a report.
Use the archived executable to reproduce an iteration; rebuilding the live
source creates the latest revision. Preserve fresh output directories.

Commands (run from the repository root, GPU access required):
  BINARY equivalence FRESH_DIRECTORY
  BINARY stress FRESH_DIRECTORY
  BINARY ptx-audit FRESH_DIRECTORY
  BINARY arithmetic-sum-pow4 1 mhc 20 80 FRESH_OUTPUT.json
  BINARY basis-recall-easy 2 mhc 10 30 FRESH_OUTPUT.json

Profile commands read frozen checkpoints and perform disposable updates.
They never change scientific checkpoints. Timings include forward/backward,
AdamW, host-to-device copies and final synchronization; dataset packing,
evaluation, initialization and disk writes are outside the timed interval.

Placement changes are checked against the frozen FP16/TF32 numerical parent:
ordinary attention fusion, tiled transpose, packed independent warps, ordered
ballot embedding gradients and single-partition GEMM finish fusion. The last
change retains the finish's exact positive-zero addition. Split reductions
and padded half outputs keep their existing finish operation.

Do not confuse placement equivalence with full numerical validation against
the previous main. The numerical parent retains 386 strict scalar-AD misses, and its longer
optimizer comparison against FP64 retains small differences. Four hard cells
passed independent learning audits, and the later complete105-cell suite
passed the epoch/coverage gate. Quartic seed1 still uses44.12% more epochs.
No claim of uniform improvement, all strict checks passing, or a numerical
convergence proof follows from the full-suite result.

The complete native training runner remains available with unchanged data,
initialization, sampler, LR0.001 WSD, original budgets and first99 criterion.
Buffered persistence remains every 500 epochs and at first99/budget/STOP.
