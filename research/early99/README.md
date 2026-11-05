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

## Round 1: measured short-suite result

The round-1 selected candidate is [`configs/adamw-lr2.json`](configs/adamw-lr2.json): multiply
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

## Round 1: long-suite regression check

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

All **210 round-1 scientific runs** completed their first crossing or original budget.
The final audit checked all 105 pairs, preserved checkpoints/histories/source
snapshots, optimizer update counts and unchanged model/task/candidate provenance.
There were no execution errors or resumed clocks. Validation details are in
[`results/validation.json`](results/validation.json).

```bash
# Recompute comparisons from the preserved round-1 histories.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr2 --tier short
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr2 --tier long

# Full integrity, oracle, CPU equivalence and CUDA capture checks.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest -q tests/diff_gpt/application/test_early99_experiments.py tests/diff_gpt/application/test_early99_benchmark.py tests/diff_gpt/application/test_current_grokking.py tests/diff_gpt/application/test_current_grokking_adapters.py
```

All raw histories, checkpoints, source snapshots and fingerprints remain under
`runs/early99/baseline` and `runs/early99/candidates/adamw-lr2`.

## Round 2: further optimizer experiments

The [predeclared round-2 plan](results/round2/plan.json) tests two separate changes
from LR ×2: increase the rate multiplier to **3**, or retain LR ×2 and reduce
AdamW **beta2 to 0.95**. Both use beta1 0.9 and retain the original warmup,
weight decay, model, data, seeds, cadence and budgets. The six-task screen uses
all three seeds and requires 18/18 passes plus improved median paired epochs on
at least three tasks relative to LR ×2. No screen-only aggregate is reported.

Both configurations passed 18/18 screen cells. LR ×3 improved three task medians
and advanced to all 75 short cells. Beta2 0.95 improved only two and was rejected
at the declared gate. All screen observations, including regressions, are in
[`screen-adamw-lr3.json`](results/round2/screen-adamw-lr3.json) and
[`screen-adamw-lr2-beta95.json`](results/round2/screen-adamw-lr2-beta95.json).

Constant LR ×3 completed **75/75** short cells, but did not beat LR ×2.
Its geometric mean was **35.3367 epochs**, versus **34.2499** for LR ×2:
the paired ratio **1.031730** means **3.17% more epochs**. There were 41 earlier,
10 tied and 24 later cells; the worst ratio was **20.625×** on
`power-division-p97-k3`, seed 0. Against the original baseline it used 38.19%
fewer epochs (ratio 0.618149). Summed trainer time was 1470.06 seconds.
All paired records and task medians/worst cases are in
[`LR ×3 versus LR ×2`](results/round2/adamw-lr3-vs-lr2-short.json) and
[`LR ×3 versus baseline`](results/round2/adamw-lr3-vs-baseline-short.json), with
matching cell/task CSV files. The [short summary](results/round2/short-summary.json)
retains the regressions. All 93 new runs passed the
[checkpoint and source audit](results/round2/audit-before-schedule.json).

The user then requested a learning-rate schedule. The separate
[cosine plan](results/round2/schedule-plan.json) retains the 50-update warmup,
starts at LR ×3, and decays to LR ×1.5 over 200 actual epochs after warmup.
Final round-2 selection waits for this additional short-suite experiment.

The round-2 factories record explicit moment coefficients; their default settings
still match the frozen baseline. CUDA checks cover the baseline, LR ×2, LR ×3
and LR ×2 with beta2 0.95 against native eager AdamW. The combined integrity,
oracle, CPU and CUDA suite passed **33 tests**; the separate report-destination
check also passed. See [validation](results/round2/validation.json).

Candidate source/configuration archives are immutable. The current driver
therefore rejects training into the old LR ×2 output, whose archived schema
predates explicit moment fields. Its histories remain readable by the comparison
tool. Replaying round 1 requires its archived sources or commit `fde92f2` with a
fresh output directory; do not replace its plan or expected hashes.

```bash
# Current round-2 configuration; completed cells are reused.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.run --write-plan --configuration research/early99/configs/adamw-lr3.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.run --run --configuration research/early99/configs/adamw-lr3.json --tier short

# Separate report destinations preserve both controls.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr3 --tier short --output runs/early99/candidates/adamw-lr3/comparisons/baseline
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -m research.early99.experiments.compare --candidate runs/early99/candidates/adamw-lr3 --baseline runs/early99/candidates/adamw-lr2 --tier short --output runs/early99/candidates/adamw-lr3/comparisons/incumbent
```

Short-suite observations alone determine selection. The long suite has already
been observed in round 1; a later check of a new winner is a reused regression
diagnostic, not new holdout evidence, and cannot affect selection or tuning.
