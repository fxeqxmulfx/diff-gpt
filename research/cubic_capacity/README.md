# Cubic-sum depth and width study

The user requested larger depth and width after the completed
[existing-model diagnostic](../transformer_path_diagnostic/README.md).
The [plan](plan.json) declares three larger sizes: two layers at width 128,
one layer at width 256, and two layers at width 256. Head width stays 32;
widening therefore uses eight heads instead of four.

Each size runs the original architecture at LR1 and LR2, and the complete
normalized inverse-width architecture at LR2, with all three seeds: **27 new
runs**. The nine current-size controls are reused. Data, batch size, 100,000-update
budgets, exact held-out oracle, one-epoch observation cadence, warmup and optimizer
settings retain their original definitions. The normalized depth coefficient
remains `1/sqrt(2L)`; this is an explicit part of the depth intervention.

This task was chosen after observing the long-suite failures. Complete every
declared run, preserve all failures and compare coverage before first-99 epochs.
This study does not alter the frozen 75-cell benchmark or select its winner.
Additional model capacity and compute are recorded rather than treated as free.

The [initialization inventory](results/inventory.json) verifies all 27 declared
configuration/seed combinations: paired parameter values/order and RNG, raw tied
weights, common rotary buffers, head width, exact gain and depth coefficient,
training data and two complete shuffled epochs. Parameter counts are 405,376
(depth), 811,520 (width), and 1,598,208 (both), versus 208,896 currently.

Validation passes **22 checks**: 20 CPU checks and two actual CUDA cases at the
largest size. Captured updates match uncaptured compiled updates under the
existing tolerances; compiler rounding is checked separately. CUDA preparation
preserves initialization, nonempty moments and RNG. Actual interrupted CPU
trainer continuations restore parameters, moments, sampler and RNG exactly.
The existing nine benchmark and six execution-journal checks also pass.

The [validation record](results/validation.json) freezes training sources and
evidence before execution. Separate output directories and process locks protect
the completed controls. Every genuine update is journaled so any future recovery
retains replay exposure. All 27 runs must finish; budget misses remain outcomes.

Training launched from source freeze `13cfe6c`. The complete declared grid remains
in progress. The separate auditor reconstructs initialized tensors, complete
oracles, first crossings, checkpoints, moments and actual execution exposure.
Four reporting checks pass, rejecting omitted failures, duplicated seeds,
incorrect crossing termination and collapsed replay bounds. All nine reused
controls have also passed the new deep audit.

The final report will retain all 36 cells, 12 size/arm groups and 51 paired-seed
contrasts. Historical controls predate the new per-update execution journal, so
their recorded runtimes remain visible but runtime ratios against them are
omitted. Capacity and first-99 epoch comparisons remain valid within the stated
task and seed scope. Reports remain separate from the fixed benchmark ranking.

The [initialization audit](../cubic_initialization_diagnostic/README.md) confirms
that widening under the unchanged `Normal(0, 0.02)` recipe changes activation
and local gradient scales. Those changes are part of this declared intervention.
The first complete three-seed contrast is the original architecture at LR 0.001:
the one-layer controls reach 99% held-out accuracy on **3/3 seeds**, while the
two-layer model reaches it on **0/3** within 100,000 updates per seed. Width stays
128; parameters increase from 208,896 to 405,376. The
[complete group audit](results/partial-group-cubic-depth-original-lr1.json)
verifies all six native checkpoints, source archives, initialized tensors,
optimizer states and complete observations, with no resumption or replay.

| Seed | One-layer first-99 epochs | Two-layer peak held-out | Two-layer final held-out | Two-layer final train |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3,531 | 97.05% | 88.06% | 99.87% |
| 1 | 3,197 | 96.94% | 86.87% | 99.98% |
| 2 | 4,315 | 87.40% | 85.67% | 99.91% |

Each two-layer miss consumes 5,263.163265 epochs. No first-99 average or speed
ratio is assigned to those misses. The [learning curves](results/partial-group-cubic-depth-original-lr1.png)
show every seed and both complete splits; [SVG](results/partial-group-cubic-depth-original-lr1.svg)
and the [paired comparison CSV](results/partial-group-cubic-depth-original-lr1.csv)
are also available. Earlier individual seed audits remain preserved.

The second complete contrast, at **LR 0.002**, has **2/3 passes at two layers**
versus **1/3 at one layer**. Seed 2 newly passes with added depth, seed 1 still
misses, and seed 0 needs **28.82% more epochs**. Parameters increase by 94.06%.
The [complete six-cell audit](results/partial-group-cubic-depth-original-lr2.json)
checks all native states, initialized tensors, source archives, exact oracles
and first crossings, with no resumption or replay.

| Seed | One-layer first-99 epochs | Two-layer first-99 epochs | Paired outcome |
| --- | ---: | ---: | --- |
| 0 | 3,574 | 4,604 | 28.82% more epochs with depth |
| 1 | Budget miss | Budget miss | Both censored |
| 2 | Budget miss | 4,442 | Newly passes with depth |

Seed 2 crosses at 84,398 updates, bracketed by (4,441, 4,442] epochs, with
4,658/4,705 exact held-out examples. Seed 1 exhausts 100,000 updates: the
two-layer peak/final held-out accuracies are 36.83%/35.52%, versus 98.19%/96.41%
for one layer. The one-layer seed-2 miss peaks at 98.38%. All failures remain
in the comparison. Neither group passes all three seeds, so no aggregate
first-99 mean or epoch ratio is reported; equal consumed budgets are not ties.

The [learning curves](results/partial-group-cubic-depth-original-lr2.png) retain
all six train/held-out histories; [SVG](results/partial-group-cubic-depth-original-lr2.svg)
and [paired CSV](results/partial-group-cubic-depth-original-lr2.csv) are available.
The earlier [seed-0](results/partial-cell-cubic-depth-original-lr2-seed0.json) and
[seed-1](results/partial-cell-cubic-depth-original-lr2-seed1.json) audits are preserved.

Each same-size/seed LR1/LR2 pair has identical initialized tensors. At two layers,
changing LR from 0.001 to 0.002 changes observed coverage from 0/3 to 2/3;
the one-layer LR 0.001 control still has 3/3 coverage on this task. These outcomes
do not establish a universal benefit from added depth or choose an initializer.

The third complete contrast, the **normalized inverse-width model at LR 0.002**,
has **0/3 passes at both one and two layers**. All six runs exhaust 100,000 updates.
The [complete six-cell audit](results/partial-group-cubic-depth-normalized-lr2.json)
verifies every native checkpoint, complete observation history, source archive,
initialized tensor, optimizer state and execution record, with no resumption or
replay. Each run consumes 5,263.163265 epochs. No first-99 mean or epoch ratio is
assigned; equal consumed budgets do not make these misses first-99 ties.

| Seed | Layers | Peak held-out | Final held-out | Final train | First-99 epochs |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 98.66% | 96.49% | 100.00% | Budget miss |
| 0 | 2 | 98.94% | 90.80% | 99.94% | Budget miss |
| 1 | 1 | 98.77% | 98.62% | 100.00% | Budget miss |
| 1 | 2 | 98.04% | 88.52% | 99.94% | Budget miss |
| 2 | 1 | 98.36% | 89.31% | 99.57% | Budget miss |
| 2 | 2 | 98.75% | 92.84% | 99.85% | Budget miss |

The three deeper runs consume 6,077.309565 trainer seconds in total. Historical
runtime ratios remain omitted because journal overhead differs. The depth
intervention adds 94.06% more parameters and changes the declared residual
coefficient from `1/sqrt(2)` to `1/2`. It does not recover any of the three
normalized seeds within this budget. This result does not settle width,
initializer choice, normalizer choice or the subsequent usual-gain comparison.

The [learning curves](results/partial-group-cubic-depth-normalized-lr2.png)
retain all six complete train/held-out histories and budget endpoints;
[SVG](results/partial-group-cubic-depth-normalized-lr2.svg) and
[paired CSV](results/partial-group-cubic-depth-normalized-lr2.csv) are available.
The earlier paired [seed-0](results/partial-cell-cubic-depth-normalized-lr2-seed0.json)
and [seed-1](results/partial-cell-cubic-depth-normalized-lr2-seed1.json) audits
remain preserved.

**9/27 new cells** are complete: all nine depth runs are audited. The one-layer,
width-256 original model at LR 0.001 is running seed 0. All 18 remaining width
and combined depth/width cells, and the subsequent usual-gain test, keep their
declared order. The full study remains open; no initializer or fixed-75 winner
is selected by these one-task contrasts.

The partial reporter requires all three seeds and writes separate artifacts
that cannot close the full study. Its incomplete-group guard rejected the
two-seed subset before the first group's third result existed. Reproduce all
three completed six-cell audits and figures without retraining:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_group --name cubic-depth-original-lr1
MPLCONFIGDIR=/tmp/cubic-partial-mpl /usr/bin/python3 -m research.cubic_capacity_reporting.plot_partial --name cubic-depth-original-lr1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_group --name cubic-depth-original-lr2
MPLCONFIGDIR=/tmp/cubic-partial-mpl /usr/bin/python3 -m research.cubic_capacity_reporting.plot_partial --name cubic-depth-original-lr2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_group --name cubic-depth-normalized-lr2
MPLCONFIGDIR=/tmp/cubic-partial-mpl /usr/bin/python3 -m research.cubic_capacity_reporting.plot_partial --name cubic-depth-normalized-lr2
```

Reproduce the earlier single-seed pairs without retraining:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_cell --name cubic-depth-original-lr2 --seed 0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_cell --name cubic-depth-original-lr2 --seed 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_cell --name cubic-depth-normalized-lr2 --seed 0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_cell --name cubic-depth-normalized-lr2 --seed 1
```

After the dispatcher finishes, run the separate auditor, reporter and plotter:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.audit
PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.report
MPLCONFIGDIR=/tmp/cubic-capacity-mpl /usr/bin/python3 -m research.cubic_capacity_reporting.plot
```
