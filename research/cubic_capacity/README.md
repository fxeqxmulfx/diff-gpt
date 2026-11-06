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

Status: sizes, controls, seeds and budgets declared; validation and training pending.
