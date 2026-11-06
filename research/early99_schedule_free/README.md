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

Both screens completed: retain native AdamW

The source freeze was `4345819`. Both declared rates passed all 18 cells, but
both improved only **two of six** parent task medians, below the declared three.
Neither advances to the full 75-cell suite. The inverse-width native-AdamW parent
remains selected; its already completed long diagnostic is not repeated.

| Screen task | SF at parent LR | SF at twice parent LR |
|---|---:|---:|
| Boolean AND | 0.7500 | 0.6667 |
| Division p193 | 1.3077 | 1.7857 |
| Division p97 | 2.8000 | 3.2000 |
| Key-value binding | 5.7000 | 1.6500 |
| Square sum | 7.8319 | **[3.5504, 4.6891]** |
| Linear transport | 0.7500 | 0.7500 |

These are median paired candidate/parent epoch ratios; lower is earlier.
The doubled-rate square-sum interval counts interrupted work. Its seed 0 was
restored from update 7,049 after the last saved observation at 7,524: at least
475 updates, 117,600 examples, or 25 epochs were replayed. The restored trajectory
crossed at 433 canonical epochs, but actual executed exposure is **at least
458 epochs**, with unknown additional work before process loss. Both the original
history and full checkpoint remain archived. The ordinary canonical comparison
files retain their original numbers; the
[physical-exposure reports](../early99_schedule_free_recovery/results/screen-summary.json)
provide the primary recovery-aware bounds. For every possible additional replay
cost, square sum remains a regression and both promotion decisions stay false.

The worst first-rate regression is square-sum seed 2, **123→1,038 epochs (8.439×)**.
The worst doubled-rate regression is key-value binding seed 1,
**20→156 epochs (7.8×)**. All five controls are retained for each rate. No screened
or successful-only geometric mean is reported. These are negative results for
the declared settings, not a general impossibility result for iterate averaging.

The [complete audit](../early99_schedule_free_recovery/results/complete-audit.json)
checks all **36 checkpoints**, both **61-file** source archives, full histories,
optimizer moments/averaging clocks, original initialization and pairing, the
30 pre-interruption cells and the first-rate reports. There are zero training
errors and one declared resumption. Six independent tests validate the median
bounds and refusal to infer a result from a missing seed. The initial 28
implementation/CUDA checks, nine benchmark checks and three Lean identities
remain intact. Observed trainer segments total 453.51 seconds for the first rate
and at least 406.72 for the doubled rate; unobserved pre-loss time is excluded.
