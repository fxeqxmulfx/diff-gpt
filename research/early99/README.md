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

All previous sweep processes were stopped at the user's request. No scientific
training for this new three-seed suite has been started. Historical seed-1
crossings in the manifest are references, not a completed three-seed baseline.
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
model/optimizer factories belong in this benchmark's experiment layer, with
separate output directories and recorded configurations. The baseline model
reference and task/data/scoring contracts must remain intact.
