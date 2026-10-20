# Plan: grokking on real time series

Started: 2026-10-20.

## Goal

Move from algorithmic grokking (division mod 97, recurrences mod 97) to real
multivariate series such as weather, and find where grokking appears there.

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

## Constraints

The model stays the plain GPT and the optimizer stays ScheduleFreeAdamW; their
code does not change, and series-specific logic stays in the encoder/decoder.
We suspect, however, that their parameters will have to change:

- model: width, depth, heads, context (`block_size`), `rope_base`, `rope_layers`;
- optimizer: learning rate, weight decay, betas, warmup;
- training protocol: batch size and batching, epochs, a second warmup
  (`recurrence_restart`);
- codec, on the data side: order k, vocabulary, domain, tokenizer.

Reasons:

- The anchor settings were screened on div97 and already differ between
  tasks: Fibonacci groks with one head but not with four, and the
  period-copy division used learning rate 0.003 instead of 0.01.
- Peri-RMSNorm and Q/K RMSNorm may move the best settings.
- Real series differ from div97 in size, noise and tokens per parameter.

Each change is screened on one seed, confirmed on three seeds on the GPU and
compared by training FLOPs.

## What to expect from real series

Grokking needs a rule that is simpler than memorization; noise has no rule.
The model learns smooth structure such as the daily cycle and persistence
early, and most of the loss on a real series is irreducible noise. So a
jump like div97's from chance to 100% on a whole weather forecast is not
expected. Grokking should be sought in the deterministic parts of real data,
measured per column against a known ceiling, not in the total loss.

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
cheap and shows how Peri-RMSNorm and Q/K RMSNorm moved the plateau and the
first 99% epoch; every later step compares against these anchors. If an
anchor no longer groks at its old settings, re-screen the learning rate and
weight decay first (`benchmarks/grokking/plans/div97_screen.json`). This is
the first place where parameters may have to change.

### 1. Noise tolerance on the anchors

Replace the answers of a fraction ε of div97 training equations with random
residues, and the supervised next value of a fraction ε of Fibonacci
training windows; test on clean held-out data. Sweep ε over 0, 1, 5 and
20%. This answers the main question for real series: whether grokking
survives noise, and how the plateau grows with it.

### 2. Weather formulas as independent equations, k=0

One row is one equation without temporal context, for example
[rh, VPmax] → VPact.

- Split like div97: hold out pairs of input tokens, not rows or months, so
  that every value occurs in training while held-out combinations do not.
  Check that every value token of each input occurs in the training split.
- A split by time does not work. Rows 10 minutes apart are near duplicates,
  and new months bring values that never occurred in training, whose
  embeddings are untrained.
- Compute the ceiling by tabulation, like
  `optimal_full_table_token_accuracy` in `recurrence.py`. Quantized inputs
  do not fully determine the quantized output, so the ceiling is below 100%.
- Controls: a random-label target and the marginal (unigram) baseline.

### 3. The formulas inside the time series

Train the full multivariate DiffGPT on the weather rows, first at k=0, then
at k=1, with loss and accuracy per column. Hypothesis: the formula columns
descend late to their ceilings, while stochastic columns such as wind and
rain only overfit.

### 4. Synthetic series with a known Bayes floor

Generate AR and seasonal AR series and pass them through the same codec.
Memorization is a training cross-entropy below the Bayes floor;
generalization is a test cross-entropy approaching it. The skill scale is
exact here, so grokking can be told apart from epoch-wise double descent.

### 5. Real forecasting with little data and long training

Weather, ETTh1 and a subset of M4 Hourly. Compare against the same model
with early stopping at equal training FLOPs. Controls: shuffled time, and
seasonal naive (one day is 144 steps of the weather data).

## Criterion for continuous data

Replace accuracy by a skill s = (L_ref − L) / (L_ref − L_floor) in [0, 1],
where L_ref is the loss of a baseline and L_floor is the ceiling or the
Bayes floor. Then apply the div97 rule:

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
- Quantization: the ceiling is below 100%; report skill relative to it.
- Averaged and query weights differ; report both, as the anchors do.

## First steps

Steps 0 and 2: re-anchor div97 and Fibonacci on the current model, and check
whether VPact = VPmax · rh / 100 groks on real rows. Both are cheap and show
whether grokking carries over from div97 to real distributions.
