# First-99% benchmark

This is the user's fixed list of 35 tasks from the current-GPT sweep. The
historical seed-1 time to first 99% held-out exact accuracy selects two disjoint
tiers: **25 short tasks** at most 60 seconds, and **10 long tasks** above 60 and
at most 300 seconds. The three constant-output controls are excluded.
`suite.json` records every task, dataset fingerprint, model size, batch, budget,
reference crossing and training-set size. Tier membership stays fixed.

Every task runs model/shuffle seeds **0, 1, 2**, with fixed data seed 1. The short
suite has 75 cells; long has 30; their union has 105. Optimization uses the short
suite. The long suite is a subsequent regression check.

Success is **one exhaustive held-out observation with at least 99% exact
sequences**. Delay and final retention do not enter this criterion. Numeric
probes now use exact generated vectors, not their former MSE target. Validation
uses the original frozen primary held-out split; final test, where present, is
reported by the underlying runner but does not alter the first-crossing rule.

Epochs are actual training examples presented divided by training-set size.
The counter includes the short final batch and the documented zero-rate first
update. CUDA capture warmup consumes no scientific steps. Observation is every
`min(original_interval, updates_per_epoch)` updates, so the first-crossing bracket
is at most one epoch wide. This is finer than the historical sweep: compare
candidates against a fresh three-seed baseline with this same schedule.

The original update budgets remain fixed. Historical time thresholds select
tasks; they do not impose wall-clock timeouts on a new run. Setup/compilation
and evaluation overhead remain visible in recorded seconds, while epochs are
the optimization measure. A resumed run's clock resets; its epoch counter does
not. Misses remain budget-censored. Per-task median/worst and aggregate geometric
mean require all expected seeds to pass.

All previous sweep processes were stopped at the user's request; the old queue
remains stopped. The fresh short baseline is now complete: **75/75 passed**.
Historical seed-1 crossings in the manifest remain references, not comparison
baselines.
[`historical_seed1.csv`](historical_seed1.csv) gives the available epoch counts;
[`validation.json`](validation.json) records the harness checks.

```bash
# Plan and reports do not train.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.run --write-plan
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.run --report --tier short

# Explicit training commands for the next authorized experiment.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.run --run --tier short
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.run --run --tier long

.venv/bin/pytest -q tests/diff_gpt/application/test_early99_benchmark.py
```

The dispatcher uses a fresh process per case/seed, resumes its own checkpoints,
and stops on execution errors. `--case NAME` runs all three seeds for one case;
`--case NAME --seed N` is an explicit single-cell diagnostic. Use `--tier all` for
the union. Outputs go to `runs/early99/baseline`, with archived suite/driver
sources, per-seed `early99_result.json`, and `summary-short.json/.csv` (or long/all).
An existing output rejects changed suite/driver sources. The baseline reuses
the verified public GPT/native AdamW runner; it preserves its frozen model check.

The optimization handoff is [`ASTRA_TASK.txt`](ASTRA_TASK.txt). Candidate
model/optimizer factories are in [`experiments/factories.py`](experiments/factories.py),
with separate output directories and recorded configurations. The original
baseline runner, model reference, manifest and data/scoring code are unchanged.
The candidate loop reuses the original sampler, stepper, evaluators and stopping
logic. Exact CPU checkpoint equivalence and CUDA capture/update checks cover
weights, moments, RNG, short batches, zero-rate first updates and recapture.

## Measured short-suite result

The selected candidate is [`configs/adamw-lr2.json`](configs/adamw-lr2.json): multiply
every task's native AdamW learning rate by **2**. Betas remain `(0.9, 0.98)`, epsilon
`1e-8`, weight decay `0.1`, and warmup 50 completed updates. Model architecture,
initialization, parameter counts, dtype and all task/scoring settings match the
baseline. This is the best of the baseline and one candidate tested, not an
exhaustive search.

| Short suite, all 75 paired cells | Baseline | AdamW LR ×2 |
| --- | ---: | ---: |
| Exact held-out first-99 coverage | 75/75 | 75/75 |
| Geometric mean first-99 epochs | 57.1653 | 34.2499 |
| Summed trainer seconds | 2066.37 | 1297.23 |

The geometric mean paired epoch ratio is **0.599139**: **40.1% fewer epochs**.
There are **55 earlier, 7 tied and 13 later** cells. The worst regression is
`power-division-p193-k2`, seed 2, at **5.8×** baseline epochs. Key-value recall and
some larger-prime division tasks also regress. Per-task three-seed medians and
worst cases, all paired ratios and observation brackets are retained in
[`results/adamw-lr2-short.json`](results/adamw-lr2-short.json),
[`cell CSV`](results/adamw-lr2-short.csv) and
[`task CSV`](results/adamw-lr2-short-tasks.csv).

Trainer seconds include model/data setup, compilation, exhaustive scoring and
final test scoring; they exclude dispatcher/process startup and the separate CPU
data preflight. No measured cell was resumed. Parameter counts are unchanged
in every pair, ranging from 100,736 to 1,183,488 depending on the fixed task.

The 15-cell screen and its regressions are in
[`results/screen-adamw-lr2.json`](results/screen-adamw-lr2.json). The selection was
frozen in [`results/selection.json`](results/selection.json) before starting any
long-suite training. The long check is complete: one fresh baseline and one
candidate run for each of its 30 cells, with no tuning against that check.

## Long-suite regression check

Coverage **regressed from 29/30 baseline passes to 26/30 candidate passes**.
The candidate introduced four budget misses, each at 100,000 updates
(5263.1633 epochs):

| Task | Censored candidate seeds |
| --- | --- |
| `arithmetic-sum-pow4` | 0, 1 |
| `arithmetic-sum-cube` | 1, 2 |

It recovered the baseline miss on `recurrence-fibonacci-k1-c2-lgt`, seed 0:
the candidate crossed at 3577 epochs, while baseline exhausted 10,000 epochs.
Among the 25 pairs in which both runs passed, 13 candidate crossings were earlier
and 12 were later. The worst observed paired ratio was 3.2647× on `basis-parity20`,
seed 1. The five comparisons containing a censored run are retained, and **no long
paired geometric mean is reported**. Summed trainer time was 3628.01 seconds for
baseline and 3238.05 seconds for the candidate, including the budget misses.

The short-suite improvement therefore comes with measured long-suite coverage
and epoch regressions. The selected configuration was not changed after this
check. All 30 comparisons, three-seed task statistics and censoring bounds are in
[`results/adamw-lr2-long.json`](results/adamw-lr2-long.json),
[`cell CSV`](results/adamw-lr2-long.csv) and
[`task CSV`](results/adamw-lr2-long-tasks.csv). The compact combined result is
[`results/summary.json`](results/summary.json).

All **210 scientific runs** completed their first crossing or original budget.
The final audit checked all 105 pairs, preserved checkpoints/histories/source
snapshots, optimizer update counts and unchanged model/task/candidate provenance.
There were no execution errors or resumed clocks. Validation details are in
[`results/validation.json`](results/validation.json).

```bash
# Candidate source/configuration archives reject changes within an existing output.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.run --run --configuration research/early99/configs/adamw-lr2.json --tier short
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr2 --tier short

# Completed long regression check; existing results are reused, not retrained.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.run --run --configuration research/early99/configs/adamw-lr2.json --tier long
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr2 --tier long

# Full integrity, oracle, CPU equivalence and CUDA capture checks: 31 passed.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_early99_experiments.py tests/diff_gpt/application/test_early99_benchmark.py tests/diff_gpt/application/test_current_grokking.py tests/diff_gpt/application/test_current_grokking_adapters.py
```

All raw histories, checkpoints, source snapshots and fingerprints remain under
`runs/early99/baseline` and `runs/early99/candidates/adamw-lr2`.
