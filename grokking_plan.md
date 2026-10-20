# Plan: grokking on real time series

Started: 2026-10-20.

## Goal

Move from algorithmic tasks (division mod 97, recurrences mod 97) to real
multivariate series such as weather, and find settings that generalize quickly.

**Priority: fast generalization first; delayed grokking only as a fallback.**
Screen for fast learning before committing to long runs. If fast generalization
exists, use it: an absent memorization plateau is a success, not a reason to
retune. Investigate delayed grokking only for tasks where the fast screens fail.
Rank successful settings by training FLOPs to the target skill, with wall-clock
time reported alongside it, and confirm the selected settings on three seeds.
Use a 250-epoch first screen, reporting both 0.9 and 0.99 skill crossings.
Expand that screen across model, optimizer and codec settings before extending
one promising setting to a long run. A failed screen means no fast setting was
found within that tested grid and budget; it does not establish nonexistence.

Execution uses seeds 11, 17 and 23 on the local RTX 3050. Run all short screens
in a family before its long fallbacks. Keep unsuccessful screens and failed
replications in the results. A setting measured on three seeds counts as a
confirmed target result only when all three cross that target.

## Starting point

- Delayed grokking is confirmed on three seeds for two tasks:
  - div97, k=0 (width 128, one block, four heads): first test accuracy
    above 99% at epochs 118, 184 and 173 (seeds 11, 17, 23);
  - Fibonacci mod 97, k=1, context 2, one head, field readout: first 99% at
    epochs 482, 531 and 324.
- Division across rows with context 48 reaches 99.8%, but by copying the
  period: every orbit has a period dividing 48. This is not grokking.
- Open: short-context division (context 4, about 1% decoded accuracy) and
  key-value binding in the MQAR-÷97 ladder (N=2: 75% train, 27% test). Both
  continue in parallel; neither is a prerequisite for this plan.
- All of these were measured at commit a5d16eb, before Peri-RMSNorm and Q/K
  RMSNorm. The current code rejects those checkpoints.
- That historical commit also predates an optimizer implementation update;
  fresh anchor differences cannot be attributed to normalization alone.

## Constraints

The model stays the plain GPT and the optimizer stays ScheduleFreeAdamW; their
code does not change, and series-specific logic stays in the encoder/decoder.
We suspect, however, that their parameters will have to change:

- model: width, depth, heads, context (`block_size`), `rope_base`, `rope_layers`;
- optimizer: learning rate, weight decay, betas, warmup;
- training protocol: batch size and batching, epochs, a second warmup
  (`recurrence_restart`);
- codec, on the data side: order k, vocabulary, domain, tokenizer. Report
  decoded error as well as skill when coarsening the grid, so speed cannot hide
  a loss of numerical resolution.

Reasons:

- The anchor settings were screened on div97 and already differ between
  tasks: Fibonacci groks with one head but not with four, and the
  period-copy division used learning rate 0.003 instead of 0.01.
- Peri-RMSNorm and Q/K RMSNorm may move the best settings.
- Real series differ from div97 in size, noise and tokens per parameter.

Each change is screened on one seed, confirmed on three seeds on the GPU and
compared by training FLOPs. Prefer the fastest reproducible generalization;
delayed generalization is useful when no fast setting succeeds.

## What to expect from real series

Grokking needs a rule that is simpler than memorization; noise has no rule.
The model learns smooth structure such as the daily cycle and persistence
early, and most of the loss on a real series is irreducible noise. So a
jump like div97's from chance to 100% on a whole weather forecast is not
expected. Seek fast generalization in the deterministic parts of real data,
measured per column against a known ceiling, not in the total loss. Where fast
learning fails, test whether longer training produces delayed generalization.

## Finding: exact formulas in the weather data

`datasets/long_term_forecast/weather/weather.csv` (Jena, 2020) has 52,696
rows of 21 columns at 10-minute steps. Several columns are functions of
others in the same row, exact up to the recorded rounding of 0.01:

| Column | Formula | Max abs residual |
|---|---|---:|
| VPact | VPmax · rh / 100 | 0.0096 |
| VPdef | VPmax − VPact | 0.0100 |
| H2OC | 1000 · VPact / p | 0.0101 |
| sh | 622 · VPact / (p − 0.378 · VPact) | 0.0081 |
| Tpot | (T + 273.15) · (1000 / p)^0.286 | 0.0201 |
| VPmax | ≈ 6.1078 · exp(17.27 T / (T + 237.3)) | 0.0706 |

- VPmax follows the Magnus formula with slightly different constants.
- rho does not match the dry-air density (residuals up to 11 g/m³); leave it
  out until its formula is known.

DiffGPT reads a table row by row, and each of these columns follows its
arguments within the row: p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh,
H2OC, and so on. At k=0 every weather row therefore already contains
equations like div97's, on real distributions; VPact = VPmax · rh / 100 is
the multiplication of two real numbers. At k=1 the formulas relate levels,
not differences, so the model has to integrate, as in the recurrence
experiments.

The file marks missing values with −9999: wv in 1 row, max. PAR in 30, OT
in 50. Clean them before any use.

## Steps

Each step changes one thing.

### 0. Re-anchor on the current model

Re-run the div97 confirmation and the Fibonacci field-readout runs, seeds
11, 17 and 23, with the current GPT, into new output directories. This is
cheap and shows how Peri-RMSNorm and Q/K RMSNorm moved the time and training
FLOPs to 99%; record plateau lengths as diagnostics. Every later step compares
against these anchors. If an anchor generalizes quickly at its old settings,
keep it even if delayed grokking disappears. If it fails to generalize quickly,
re-screen the learning rate and weight decay first
(`benchmarks/grokking/plans/div97_screen.json`), then investigate a longer run
as a fallback. This is the first place where parameters may have to change.

### 1. Noise tolerance on the anchors

Replace the answers of a fraction ε of div97 training equations with random
residues, and the supervised next value of a fraction ε of Fibonacci
training windows; test on clean held-out data. Sweep ε over 0, 1, 5 and
20%. This answers the main question for real series: whether fast generalization
survives noise, how much extra compute it needs, and, if fast learning fails,
whether delayed generalization survives.

### 2. Weather formulas as independent equations, k=0

One row is one equation without temporal context, for example
[rh, VPmax] → VPact.

Screen for fast learning to the quantization ceiling first. Extend training
to search for delayed grokking only if those screens do not succeed.

- Split like div97: hold out pairs of input tokens, not rows or months, so
  that every value occurs in training while held-out combinations do not.
  Check that every value token of each input occurs in the training split.
- A split by time does not work. Rows 10 minutes apart are near duplicates,
  and new months bring values that never occurred in training, whose
  embeddings are untrained.
- Compute the ceiling by tabulation, like
  `optimal_full_table_token_accuracy` in `recurrence.py`. Quantized inputs
  do not fully determine the quantized output, so the ceiling is below 100%.
  Count the complete source table before averaging entropy over the frozen
  evaluation inputs. Tabulating only a capped evaluation sample understates
  ambiguity and gives an artificially high ceiling. Preserve original curves
  when correcting a reference, and recompute their skill and target crossings.
- Controls: a random-label target and the marginal (unigram) baseline.
  Also compare literal copying of each input token and a single-input
  conditional table fitted only on training examples. Select its smoothing
  strength by training leave-one-out CE. Independent column normalization can
  make H2OC or sh almost a copy of VPact, so a 0.9 unigram-relative skill alone
  does not establish use of both operands.
  For input-independent random labels, report raw CE, accuracy and the known
  marginal floor; disable rule-skill crossings because the population marginal
  already attains that floor. Preserve historical raw curves when auditing
  their original normalization.

### 3. The formulas inside the time series

Train the full multivariate DiffGPT on the weather rows, first at k=0, then
at k=1, with loss and accuracy per column. First test whether the formula
columns reach their ceilings quickly. If they do not, test the fallback
hypothesis that they descend late, while stochastic columns such as wind and
rain only overfit.

Report a restricted-input formula comparator for k=0: conditional entropy given
the formula's two same-row operands on that run's grid. Additional context can
improve on it, so it is not the full model's Bayes floor. At k=1 the level
formula does not determine native derivative-token labels; keep that floor
explicitly unknown instead of reusing the k=0 ceiling.

The aggregate 10% validation gain is a short-run diagnostic. Also retain the
250-epoch result and extend the selected k=0 and replicated k=1 trajectories
to 1500 epochs on three seeds when individual formula columns still have
reference gaps. This tests late per-column improvement, including the wind
and rain controls, without retuning the successful short-run settings.

### 4. Synthetic series with a known Bayes floor

Generate AR and seasonal AR series and pass them through the same codec.
Memorization is a training cross-entropy below the Bayes floor;
generalization is a test cross-entropy approaching it. The skill scale is
exact here, so fast generalization, delayed grokking and epoch-wise double
descent can be distinguished. Prefer the fast setting whenever it exists.

Use both an exactly encoded lattice process and a continuous linear Gaussian
process. For Gaussian AR, integrate the stationary conditional distribution
over the actual input and output quantization bins, including clipping tails;
innovation differential entropy alone is not a token cross-entropy floor.
Fit its codec on an independent training-only calibration realization. Report
quadrature convergence and condition the oracle on any target-vocabulary
coverage filter applied to evaluation. Use context equal to the AR lag so that
the stated conditional law includes every observed value of the relevant phase.

### 5. Real forecasting with little data

Weather, ETTh1 and a subset of M4 Hourly. Compare against the same model
with early stopping at equal training FLOPs. Prefer fast generalization and
use long training when fast screens fail. Controls: shuffled time, and
seasonal naive (one day is 144 steps of the weather data).

Evaluate free-running horizons of 144 weather rows, 24 ETTh1 rows and 48 M4
Hourly rows at frozen test origins, using generated rows as subsequent inputs.
Report persistence as well as seasonal naive, decoded errors per column and
training-normalized errors. Compare the final, full-budget validation-best and
first-patience checkpoints on identical eligible origins. These are internal
chronological M4 holdouts, not official M4 challenge scores.

## Criterion for continuous data

Replace accuracy by a skill s = (L_ref − L) / (L_ref − L_floor) in [0, 1],
where L_ref is the loss of a baseline and L_floor is the ceiling or the
Bayes floor. Report the first crossings of test s = 0.9 and 0.99, their training
FLOPs and wall-clock times. Select fast, reproducible crossings first; no
memorization plateau is required for success.

For runs where fast learning fails, classify delayed grokking with the div97
rule:

- a plateau: training s ≥ 0.99 and test s ≤ 0.1 for at least 25 epochs;
- then a rise of test s to 0.9 within 250 epochs, and a crossing of 0.99.

On real data the floor is unknown. There, late generalization means a phase
in which the train/test gap grows, followed by a final test loss below the
early-stopping minimum by more than the spread of three seeds.

## Infrastructure

- One runner for series, built on the training loop of
  `benchmarks/grokking/recurrence.py`: both complete splits evaluated every
  epoch, both Schedule-Free weight states, `history.csv`, checkpoints with
  the architecture marker, resume. It adds:
  - a dataset registry (generators, weather formulas, real loaders) with
    frozen splits and hashes;
  - metrics and ceilings per column;
  - the criterion above.
- Save the actual query weights as well as the averaged weights. Algebraically
  reconstructing query weights from packed optimizer tensors can change sharp
  argmax decisions through floating-point cancellation. Verify the saved states
  strictly and require bitwise equivalence between interrupted/resumed and
  uninterrupted GPU training.
- Do not use `diff_gpt.train.train()` for these runs: it restores the best
  checkpoint by validation loss and so hides late generalization.
- Leakage guards: domain and scale from the training split only; a gap
  between splits of at least the context length; every test token occurs in
  training.
- Fast tests in the style of `tests/grokking`:
  - splits are disjoint and do not leak;
  - ceilings are right on small tables;
  - generators are deterministic (hash fixtures);
  - the criterion fires correctly on synthetic curves;
  - resume continues training exactly.

## Pitfalls

- Shortcuts: copying the period, as with context 48; on weather,
  persistence (repeating the previous row). Step 2 has no temporal context
  for this reason.
- Copying an operand after separate column normalization can explain fast
  same-row scores. Report the training-only single-operand controls alongside
  every independent formula result.
- Quantization: the ceiling is below 100%; report skill relative to it.
- Averaged and query weights differ; report both, as the anchors do.

## First steps

Steps 0 and 2: re-anchor div97 and Fibonacci on the current model, and check
whether VPact = VPmax · rh / 100 generalizes quickly on real rows. Both are
cheap and show whether fast learning carries over from div97 to real
distributions. Keep successful fast settings; investigate delayed grokking
only where fast settings cannot be found.

## Execution evidence (updated 2026-10-21)

The loop is paused at the user's request (2026-10-21). All experiment workers,
queued extensions and final validation watchers have stopped. Sources, data
fingerprints, configurations, both weight states, curves and strict verification
records remain under `runs/grokking-series`; `pause-status.json` records the stop.
The model and optimizer implementation files remain unchanged. The experiment
matrix and final completion audit are unfinished.

| Completed measurement | Seeds 11 / 17 / 23 |
|---|---|
| Current div97 anchor: first 99% accuracy | 24 / 36 / 81 epochs |
| Current Fibonacci anchor: first 99% accuracy | 3511 / 496 / 323 epochs |
| Div97 with 1% corrupted training answers: first 99% | 72 / 66 / 66 epochs |
| Div97 with 5% corruption: first 99% | 111 / 138 / 165 epochs |
| Div97 with 20% corruption: first 90% | 194 / 105 / 132 epochs |
| Faster Fibonacci setting, clean: first 90% | 159 / 136 / 121 epochs |
| Faster Fibonacci setting, 1% corruption: first 90% | 155 / 144 / 123 epochs |
| Lattice AR, width 64: first 0.9 expected-CE skill | 34 / 22 / 19 epochs |
| Lattice seasonal AR: first 0.9 expected-CE skill | 5 / 4 / 4 epochs |
| Multivariate weather k=0: 10% validation CE improvement | 3 / 3 / 3 epochs |
| Multivariate weather k=1, 2048 training windows: same target | 5 / 5 / 4 epochs |
| Weather forecasting, context 8: same validation target | 5 / 5 / 4 epochs |
| ETTh1 forecasting, context 24: same validation target | 2 / 2 / 2 epochs |
| M4 Hourly forecasting, context 24: same validation target | 2 / 1 / 2 epochs |

The Fibonacci noise adapter excludes four seed-23 test windows containing
tokens absent from clean training, identically at every corruption fraction.
The original anchor above retains its original complete split. These are
distinct recorded populations.

The first k=1 weather selection (512 training windows) passed on two of three
seeds. The next successful screened setting, with 2048 windows, passed on all
three. Retain the failed replication and use the replicated setting. Quick
validation gains do not establish useful multi-step forecasts. The complete
three-seed free-running comparison at eight eligible origins per seed gives
mean normalized MSE (final GPT / seasonal naive) of 16.344 / 0.367 on weather,
1.278 / 0.382 on ETTh1 and 1.698 / 0.301 on M4 Hourly. All nine shuffled-time
confirmations miss the validation target. Preserve these negative forecast
results when reporting the quick validation crossings.

The lattice seasonal-AR trajectories overfit after their fast crossings.
Their validation-selected checkpoints at epochs 5 / 5 / 9 retain test skill
0.912 / 0.917 / 0.929. Strict replay of both optimizer weight states passes for
all three saved checkpoints; their hashes and replay paths are indexed in
`runs/grokking-series/retained-seasonal-fast-models.json`. Keep these fast models
instead of selecting their worse epoch-250 weights.

Strict state replay and exact GPU continuation are recorded in
`runs/grokking-series/checkpoint-format2-audit.json` and
`runs/grokking-series/gpu-resume-format2-audit.json`.
`runs/grokking-series/library-source-audit.json` verifies all 25 tracked Python
library files against HEAD; model and optimizer code are unchanged.

Historical/current anchor comparisons now include FLOPs and measured times in
`runs/grokking-series/anchor-comparison.json`. Div97's current 99% crossings use
0.203 / 0.196 / 0.468 of the historical training FLOPs. Fibonacci's ratios are
7.284 / 0.934 / 0.997; normalization and optimizer changes are confounded.

The first H2OC independent-equation screen reaches 0.9 unigram-relative skill
at epoch 1, but a training-only VPact conditional table already reaches 0.936.
Its literal token-copy accuracy is 83.69%. Retain this shortcut comparison
when interpreting the fast score, and confirm the selected configuration on
three seeds before claiming a replicated result.

Independent VPact screens tested model width, heads, learning rate, decay,
beta1, data amount, domain and binary input representation. No 0.9 full-table
skill was found in that initial 250-epoch grid; the best was about 0.855.
The follow-up short screens with all training equations, widths 32 / 64 and
training batches of 512 / 4096 are complete for all four formulas. VPact still
misses 0.9 skill in all three selected 1500-epoch fallbacks; its three random-label
controls are complete. Peak VPact skill is 0.864 / 0.797 / 0.815 and final skill
is 0.855 / 0.671 / 0.713: the longer budget does not produce a delayed crossing.
VPdef seeds 11 and 17 completed 1500 epochs without reaching 0.9. Seed 23 was
stopped after epoch 598; its last durable checkpoint is epoch 500. H2OC and sh
confirmations and the remaining random-label controls are unfinished.
The remaining Fibonacci noise fallbacks
are bounded at 4000 epochs for 5% and 1500 for 20%; the 5% run had already
reached epoch 3900 before deferral. A miss at these budgets is a bounded
negative result, not proof that longer generalization is impossible.

Gaussian controls, the remaining Fibonacci noise settings and the multivariate
weather extensions have not run; their queued jobs were stopped.
Weather/ETTh1/M4 forecasting is complete;
the six measured settings per dataset varied width, heads, learning rate,
RoPE base and training sample count. All used k=0 and the original domain;
the current reproduction recipe also offers derivative and domain variants,
which are not part of the reported measurements. `series_report.py` combines
all source versions without merging
unlike configurations and reports FLOPs, timings, seed ranges, target pass
counts, per-column metrics and equal-budget validation selection. Historical
format-1 noisy-anchor attempts are retained as superseded diagnostics.

The report writes separate `confirmed-target-rankings.csv` and
`confirmed-validation-rankings.csv` under `runs/grokking-series/fast-first-current`.
The first labels accuracy, known-floor skill and empirical formula skill
separately; the second ranks the practical validation target. Both include
training FLOPs, measured time and whether every seed crossed within 250 epochs.
`runs/grokking-series/completion-audit.json` checks the required run matrix,
data and source hashes, strict state verification, cost accounting, controls
and retained fast checkpoints. Completion requires this audit and the final
test checks to pass, followed by review of the results in this document.

Short-context evaluation batches were increased from 256 to 4096 rows, while
keeping every row and every training update. At epoch 250, model, query,
optimizer, sampler and RNG tensors are bitwise identical to the original
evaluation schedule. Full-run time fell from 164.92 to 119.45 seconds for this
comparison; per-state scoring measured 0.52 versus 0.15 seconds. The numerical
metric difference is below 1.5e-7. Evidence is in
`runs/grokking-series/evaluation-batch-audit.json` and
`runs/grokking-series/evaluation-batch-training-audit.json`. New formula runs
are under `formula-batched`; earlier runs remain available.
