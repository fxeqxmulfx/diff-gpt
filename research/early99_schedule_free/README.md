# Schedule-Free AdamW on inverse-width SSMax

This is the user's requested optimizer experiment on the unchanged shared
inverse-width model. The plan was committed as `d05e74b` before new training.
It tests the parent learning rate (original case LR ×2) and twice that rate
(original case LR ×4), with beta1=.9, beta2=.98, epsilon=1e-8 and the original
case's weight decay.

The model remains Peri-RMS, embedding RMSNorm, SwiGLU, centered XSA, Q/K RMS,
SSMax with log(N)/head-width gain, post-RMS residual scaling, fixed RoPE and
tied embeddings/readout. This experiment adds neither numerical repairs nor
LeRoPE. The original inverse-width model's 40/50 complete-stack numerical
stress result remains an unresolved limitation.

The existing repository `ScheduleFreeAdamW` is used unchanged. Gradients and
decay act at `y=(1−beta1)z+beta1*x`. Held-out observations and checkpoints use
the averaged `x`; the full optimizer state includes `z`, second moments,
update count, maximum effective rate, averaging total and mode. The external
50-completed-update warmup includes its original first zero-rate update;
internal warmup is zero. The rate is constant afterward. Mode conversions
can round in floating point; they are part of the measured procedure.

The loss retains CUDA compilation. The foreach optimizer runs eagerly outside
update graphs. Preparation issues no optimizer updates. Original evaluation
graphs can read the same parameter storage after conversion to `x`. Runtime
is recorded but does not determine promotion.

Both 18-cell screens must finish. An arm advances only with 18/18 passes and
improved paired task medians on at least three of the six fixed tasks versus
the inverse-width native-AdamW parent. Every advancing arm then completes
all 75 short cells. A new winner requires full coverage and a lower complete
geometric mean than the parent's 16.2660501255 epochs. Comparisons retain the
parent, quarter-scale Q/K, original LR×2, original baseline and no-Q/K controls.
No successful-only mean is allowed. Only a new short winner runs the reused
30-cell long diagnostic; a retained parent does not repeat it.

All **28 implementation checks passed**, including two real CUDA cases, in
57.56 seconds. The nine fixed benchmark integrity checks also passed. Exact
outputs, JUnit cases and 64 source/test hashes are in
[validation.json](results/validation.json). No benchmark update preceded these
checks.

Implementation validation covers independent scalar optimizer updates, the
zero-rate first update, exact initial model/RNG/capacity, averaged-weight
scoring, idempotent modes, checkpoint/sampler/RNG recovery, the original
trainer's observation/stopping loop, and real CUDA compiled/eager updates
including short final batches. `readiness.py` requires the complete validation
record and committed source before any benchmark training. It also requires
the previous inverse-width long diagnostic and recovery report to finish.

`CheckedIdentities.lean` checks the local modern Schedule-Free two-buffer
update and recovery of `x` with the actual beta1=.9. Three existing library
identities compile without warnings or extra axioms. The fixed-parameter
`trainingRun_pack` is checked for context, not asserted to directly describe
the externally changing warmup rate. Per-step equivalence applies at each
supplied rate; packing depends only on the fixed beta1. These are real
algebraic statements, not nonconvex convergence or floating-point guarantees.
The architecture's prior real-arithmetic normalization/score bounds still
apply pointwise at either weight state.

The earlier negative Schedule-Free recall study is retained in
`research/recall_ablation/results/summary.json`; this experiment does not
overwrite or reinterpret it.

Commands, from the repository root with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=.`:

```sh
.venv/bin/python -m research.early99_schedule_free_reporting.run_screens --run
.venv/bin/python -m research.early99_schedule_free_reporting.audit --name peri-rms-inverse-width-sf-adamw-lr2 --name peri-rms-inverse-width-sf-adamw-lr4 --output research/early99_schedule_free/results/audit-screen.json
.venv/bin/python -m research.early99_schedule_free_reporting.verify_screen
# Commit complete screen evidence before full-suite training.
.venv/bin/python -m research.early99_schedule_free_reporting.run_full --run
.venv/bin/python -m research.early99_schedule_free_reporting.verify_results --write
# Commit the audited short selection; only a new winner needs the next command.
.venv/bin/python -m research.early99_schedule_free_reporting.run_long --run
```

Primary reference: [official Schedule-Free implementation](https://github.com/facebookresearch/schedule_free).
