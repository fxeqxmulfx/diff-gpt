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
The first two completed new cells (two layers, width 128, original LR1) both
exhaust their 100,000-update budgets. Seed 0 peaks at 97.05% held-out accuracy
and finishes at 88.06%; seed 1 peaks at 96.94% and finishes at 86.87%. Final
training accuracy is 99.87% and 99.98%, respectively. The
[seed-0 audit](results/partial-audit-depth-original-lr1-seed0.json) and
[seed-1 audit](results/partial-audit-depth-original-lr1-seed1.json) verify native
checkpoints, sources, initialization, optimizer states and complete observations,
with no resumption or replay. **2/27 new cells** are complete; seed 2 is running.
This is not yet a three-seed conclusion about depth or the enlarged normalized model.

After the dispatcher finishes, run the separate auditor, reporter and plotter:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.audit
PYTHONPATH=. .venv/bin/python -m research.cubic_capacity_reporting.report
MPLCONFIGDIR=/tmp/cubic-capacity-mpl /usr/bin/python3 -m research.cubic_capacity_reporting.plot
```
