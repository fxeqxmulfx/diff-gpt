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

That contrast accounts for the first three new cells. Adding depth under this
particular original-model recipe did not improve coverage within the fixed budget.
The higher-rate three-seed contrast, width and enlarged normalized-model results
remain incomplete.

The fourth completed cell, two layers at original LR 0.002, **passes seed 0 at
4,604 epochs** (87,476 updates). Its one-layer LR 0.002 control passes at 3,574
epochs: the deeper model takes **28.82% more epochs** with **94.06% more parameters**.
The [paired single-cell audit](results/partial-cell-cubic-depth-original-lr2-seed0.json)
checks both full trajectories, checkpoints, optimizer states, sources and
initialized tensors, with no resumption or replay. The crossing bracket is
(4,603, 4,604] epochs; final exact held-out accuracy is 4,658/4,705 = 99.0011%.

The same two-layer seed exhausted its budget at LR 0.001, using the exact same
initialized parameter tensors. This is a single-seed learning-rate observation;
all three seeds must finish before interpreting the LR 0.002 group's coverage.

The fifth completed cell, **seed 1 at two layers and LR 0.002**, exhausts all
100,000 updates without 99%. Its peak held-out accuracy is **36.83%**, final
held-out accuracy 35.52% and final training accuracy 99.55%. The one-layer
control also exhausts its budget, peaking at 98.19% and ending at 96.41%
held-out accuracy. The [seed-1 pair audit](results/partial-cell-cubic-depth-original-lr2-seed1.json)
verifies both complete trajectories and native states, including 5,263.163265
epochs each and no resumption or replay. Both first-99 values and their ratio
remain undefined; equal consumed budgets do not make them first-99 ties.

**5/27 new cells** are complete. Seed 2 at depth two and LR 0.002 is running.
This group's one pass and one miss are incomplete three-seed evidence. The
remaining 22 cells and the subsequent usual-gain test keep their declared order.

The partial reporter requires all three seeds and writes separate artifacts
that cannot close the full study. Its incomplete-group guard rejected the
two-seed subset before the third result existed. Reproduce this six-cell audit
and figure without retraining:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_group --name cubic-depth-original-lr1
MPLCONFIGDIR=/tmp/cubic-partial-mpl /usr/bin/python3 -m research.cubic_capacity_reporting.plot_partial --name cubic-depth-original-lr1
```

Reproduce the later single-seed pair without retraining:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_cell --name cubic-depth-original-lr2 --seed 0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.partial_cell --name cubic-depth-original-lr2 --seed 1
```

After the dispatcher finishes, run the separate auditor, reporter and plotter:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.audit
PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.report
MPLCONFIGDIR=/tmp/cubic-capacity-mpl /usr/bin/python3 -m research.cubic_capacity_reporting.plot
```
