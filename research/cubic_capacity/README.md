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

Status: implementation validated; commit the source freeze before launching the
complete declared grid. Reports remain separate from the fixed benchmark ranking.
