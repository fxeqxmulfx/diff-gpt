# Plan: the GIFT-Eval and TIME leaderboards

Started: 2026-10-21. Status: plan only; nothing has been run yet.

## Goal

Enter diff-gpt in two public forecasting leaderboards, with honest and
reproducible results:

- **GIFT-Eval** (Salesforce) has 97 configurations. Entries are submitted as a
  pull request to github.com/SalesforceAIResearch/gift-eval and shown in the
  Hugging Face Space Salesforce/GIFT-Eval.
- **TIME** (Real-TSF, ICML 2026) has 98 tasks. Entries are submitted as a pull
  request to the Hugging Face dataset Real-TSF/TIME-Output and shown in the
  Space Real-TSF/TIME-leaderboard.

Both leaderboards divide each model's MASE and CRPS by seasonal naive's and
combine the ratios by a geometric mean over the configurations. A score of 1.0
equals seasonal naive, and lower is better. The first target is to get below
1.0 on validation. The leaderboard tables below show where an entry would
land; they are not commitments.

Submitting is an outward-facing action. It happens only after the user's
explicit go-ahead, given separately for each benchmark and each entry.
Everything before that point stays local.

## Starting point

- The current plain GPT has been measured on real forecasts only in the
  low-data settings of `grokking_plan.md`, and there it trails seasonal naive.
  These are free-running results averaged over three seeds, as mean
  training-normalized MSE:

  | Dataset / horizon | Final GPT | Seasonal naive |
  |---|---:|---:|
  | Weather / 144 | 16.344 | 0.367 |
  | ETTh1 / 24 | 1.278 | 0.382 |
  | M4 Hourly / 48 | 1.698 | 0.301 |

  Those models saw only 8 or 24 context rows, while seasonal naive reads a full
  season. The comparison says nothing about longer contexts. It does show that
  nothing here is known to beat seasonal naive yet.
- Earlier model versions reached sMAPE 13.11, MASE 1.018 and OWA 0.569 on M4
  Hourly (`improvement_plan.md`). They relied on model-side additions that have
  since been removed: a mixture head, exact token features, channel embeddings
  and Attention Residuals.
- Lessons from `improvement_plan.md` that carry over:
  - On M4 Daily the forecast was about as good as naive (OWA 1.00). The largest
    step, 19.5, against a median step of 0.02, made the grid too coarse to
    represent ordinary moves. A quantile domain fixes the grid, but
    random-walk-like series leave little to gain anyway.
  - Strong seasonality suits the method.
  - Free-running samples at temperature 1 are over-dispersed. On ETTh1 the
    nominal 80% band covered 95%. `calibrate_temperature` picks the temperature
    on validation windows by CRPS.
- The library already has what a probabilistic entry needs, one context at a
  time: `predict_samples`, `predict_quantiles`, `predict_point` (the marginal
  median for MAE) and `calibrate_temperature`. `predict_many` batches contexts
  of one shape, but it draws only one trajectory per context.
- Two features are on master since 2026-10-21 (commit 351f8ca):
  - per-channel ranges [lo, hi], via `get_domain_of_definition(..., symmetric=False)`;
  - missing values: NaN is filled in the input and masked out of the loss.

  Both benchmarks contain missing values.
- A forecast must fit in the context: (context rows + H − k) · F ≤
  `block_size`. `DiffGPT._forecast_order` raises an error otherwise.

## Constraints

- The model stays the plain GPT and the optimizer stays ScheduleFreeAdamW;
  their code does not change. Benchmark-specific logic goes in the
  encoder/decoder, the data loaders, the sampling code and the evaluation
  adapter. Model and optimizer parameters may change.
- Hardware: one RTX 3050 Laptop GPU with 4 GB of memory, and 123 GB of free
  disk on 2026-10-21. Every estimate below is checked against both.
- Python versions: diff-gpt requires Python ≥ 3.14. gift-eval and timebench
  pin gluonts ~0.15.1 and numpy ~1.26, which do not run on 3.14. The
  evaluation therefore runs in a separate uv environment on Python 3.11, and
  the two sides exchange files. gluonts never enters diff-gpt.
- Compare changes by training FLOPs and validation scores, averaged over seeds
  11, 17 and 23 on the GPU.
- Select settings on validation only. Test windows are scored once per
  candidate submission and are never used to choose a setting.
- Do not choose checkpoints by validation loss. `diff_gpt.train.train()`,
  which `DiffGPT.train` runs, keeps the final weights by default; leave its
  `use_early_stopping` and `save_best` off, because both select by validation
  loss and so hide late generalization. That loss is also one-step
  cross-entropy, not the free-running forecast the boards score. In the
  low-data runs of `grokking_plan.md`, the checkpoint with the best validation
  loss forecast better than the final one on weather (normalized MSE 11.668
  against 16.344) and worse on ETTh1 (1.736 against 1.278). Pass an `on_eval`
  callback instead: it runs at every evaluation while the model holds the
  scored weights, so it can write the curves to `history.csv`, save periodic
  checkpoints and score free-running forecasts. Choose the checkpoint by
  validation MASE and CRPS of those forecasts.
- New code passes the repository's strict ruff checks and the fast tests,
  without mocks or monkeypatching.
- Commit only when the user asks, and stage explicit paths, because other
  sessions share the working tree.

## The two benchmarks

### GIFT-Eval

**Data and windows**

- Test data: `hf download Salesforce/GiftEval --repo-type=dataset`
  (1.60 GB, Apache-2.0). It loads through
  `Dataset(name, term, to_univariate, storage_env_var="GIFT_EVAL")` from
  `src/gift_eval/data.py`.
- There are 97 configurations, listed in `scripts/dataset_configurations.csv`:
  55 short, 21 medium and 21 long.
- Horizon H for the short term:
  - M4: Y 6, Q 8, M 18, W 13, D 14, H 48;
  - all other data: M 12, W 8, D 30, H 48, minute data 48, second data 60.

  Medium is 10H and long is 15H, so the longest horizon is 900 steps.
- Number of windows: min(max(1, ⌈0.1 · shortest series length / H⌉), 20).
  M4 always gets 1. (The paper lists 2 windows for M4 Hourly; the code, which
  produced the leaderboard, uses 1.)
- The test split is the last windows · H steps of each series, cut into
  non-overlapping windows. Each window's context is the full history before
  it.
- `validation_dataset` ends where the test split starts. `training_dataset`
  ends H steps earlier. Both splits are defined per configuration.
- Eight datasets are multivariate; jena_weather has 21 variates. The notebooks
  use `to_univariate=True`, but joint forecasting is also accepted.
- Workload: 371,330 univariate windows and 40.67M forecast points (9.78M short,
  13.33M medium, 17.56M long). The largest configurations are temperature_rain
  (96,216 windows), m4_monthly (48,000) and bitbrains_fast_storage/5T/short
  (45,000).

**Metrics and ranking**

- Scoring call:
  `evaluate_model(..., axis=None, mask_invalid_label=True, allow_nan_forecast=False, seasonality=get_seasonality(freq))`.
- CRPS is MeanWeightedSumQuantileLoss over the quantiles 0.1 to 0.9.
- MASE seasonality: 10S 360, 5T 288, 10T 144, 15T 96, H 24, M 12, Q 4,
  D/W/A 1.
- The Space divides each configuration's score by Seasonal_Naive's, then takes
  the geometric mean over the 97 configurations. Ranks are mean
  per-configuration ranks over all 153 entries.

**Submission**

- A PR adds `results/<MODEL>/all_results.csv` and `config.json`.
  - The CSV has 98 lines and 15 columns, with one row for each of the 97
    configuration keys.
  - `config.json` holds model, model_dtype, model_link, code_link, org,
    model_type, testdata_leakage and replication_code_available.
- Review steps: `python scripts/validate_results.py results/<MODEL>` must pass,
  then the Salesforce CLA bot, and then a maintainer reruns the model on a
  subset.
- model_type rules:
  - zero-shot: no GIFT-Eval train split was used;
  - pretrained: the train splits were included in a single training run;
  - fine-tuned: an extra training stage on the train splits;
  - deep-learning: per-dataset models trained from scratch;
  - routers and ensembles are relabeled agentic.
- Leakage counts only the test split:
  - GiftEvalPretrain is safe;
  - full LOTSA leaks;
  - the Chronos corpus is safe only without hierarchical_sales, loop_seattle,
    m_dense, restaurant and sz_taxi.

**Where entries land (2026-10-21; normalized MASE and CRPS)**

Agentic entries lead; the best MASE is 0.610 (EXAONE-Forecast-Agent).

| Model | Note | Params | MASE | CRPS |
|---|---|---|---:|---:|
| Timestral-209M | zero-shot | 209M | 0.658 | 0.446 |
| TimesFM-3 | zero-shot | 331M | 0.667 | 0.456 |
| Chronos-2 | pretrained | 119.5M | 0.698 | 0.485 |
| Timestral-5M | zero-shot | 5.21M | 0.705 | 0.479 |
| TiRex | zero-shot | 35M | 0.716 | 0.488 |
| TTM-R3-PT | pretrained | 1.41M | 0.724 | 0.520 |
| FlowState-9.1M | zero-shot | 9.1M | 0.726 | 0.502 |
| tinycast | zero-shot | 146,505 | 0.774 | 0.545 |
| xLSTM-Mixer | type not checked | | 0.780 | 0.510 |
| goia-forecast-nano-v0 | zero-shot, one consumer GPU | 4.73M | 0.815 | 0.553 |
| PatchTST | deep-learning baseline | | 0.849 | 0.587 |
| iTransformer | deep-learning baseline | | 0.893 | 0.620 |
| TFT | deep-learning baseline | | 0.915 | 0.605 |
| N-BEATS | deep-learning baseline | | 0.938 | 0.816 |
| Seasonal_Naive | | | 1.000 | 1.000 |
| DLinear | deep-learning baseline | | 1.061 | 0.846 |
| DeepAR | deep-learning baseline | | 1.343 | 0.853 |

Chronos-T5 is the only family on the board that tokenizes values (4096 bins),
and every one of its sizes leaks; chronos-t5-small scores 0.892 / 0.663. A
token-level GPT without leakage would be the first on the board.

### TIME

**Data and windows**

- Data: HF `Real-TSF/TIME` (220 MB, Arrow files read with `load_from_disk`,
  CC BY-NC 4.0). Code: github.com/zqiao11/TIME (package `timebench`). Paper:
  arXiv 2602.12147.
- There are 98 tasks from 39 sources: 24 dataset/frequency configurations
  with three terms each, plus 26 with one term. Horizons run from 3 to 864
  steps.
- Test windows: the last `test_length` steps, cut into floor(test_length/H)
  non-overlapping windows. A `val_length` block sits right before them.
- There is no fixed context. Each window gets the full history before it,
  including earlier test windows, and the model truncates it.
- All variates are targets; univariate models use `to_univariate`.
- Workload: about 111k univariate windows and 7.8M forecast steps.
  Azure2019-D/I make up about 49% of the steps, at H = 288.

**Metrics and ranking**

- Forecasts are nine quantiles, 0.1 to 0.9. In the paper, sample-based models
  draw 100 samples.
- MASE uses the 0.5 quantile, scaled by the seasonal naive error over the
  window's history. Seasonality: 5T 288, 15T 96, 20T 72, 30T 48, H 24, B 5,
  D 1, W 1, M 12, Q 4.
- CRPS is the mean over quantile levels of 2 · Σ pinball / Σ|y|.
- Each task's value is a nanmean, divided by seasonal_naive's value. The
  leaderboard takes the geometric mean over the 98 tasks and sorts by MASE.
- The app trusts the submitter's `metrics.npz`.

**Submission**

- Upload with:
  `api.upload_folder(folder_path="output/results/<model>", path_in_repo="results/<model>", repo_id="Real-TSF/TIME-Output", repo_type="dataset", create_pr=True)`.
- The folder holds 98 subfolders `<dataset>/<freq>/<term>`. Each contains
  `predictions.npz`, `metrics.npz` and `config.json`, written by
  `save_window_predictions(..., output_base_dir="output/results/<model>")`.
  The saver does not add the model name to the path, hence the
  `output_base_dir` argument.
- Template: `experiments/chronos2.py`. Local scoring:
  `scripts/compute_local_leaderboard.py`.
- TIME requires no metadata and sets no compute limit, no deadline and no
  written pretraining policy. A model trained on the Chronos corpus was
  accepted as zero-shot, and earlier PRs were merged after short reviews.

**Where entries land (2026-10-21, 40 models)**

| Model | Params | MASE | CRPS |
|---|---|---:|---:|
| TimeRSI-1.5 (per-task agents over Timestral-209M) | | 0.614 | 0.513 |
| Timestral-209M | 209M | 0.626 | 0.522 |
| Timestral-28M | 28M | 0.638 | 0.534 |
| Timestral-5M | 5.2M | 0.652 | 0.544 |
| Chronos-2 | 119M | 0.662 | 0.556 |
| TimesFM-2.5 | 200M | 0.669 | 0.567 |
| TiRex | 35M | 0.683 | 0.573 |
| Fracast-0 | 85,001 | 0.768 | 0.649 |
| seasonal_naive | | 1.000 | 1.000 |

### Side by side

| | GIFT-Eval | TIME |
|---|---|---|
| Scored units | 97 configurations | 98 tasks |
| Forecast format | gluonts Forecast with quantiles and "mean" | 9 quantiles, float16 |
| Who computes the scores | the submitter; maintainers rerun a subset | the submitter, trusted |
| What must be declared | model type, leakage, code | nothing |
| Data license | Apache-2.0 | CC BY-NC 4.0 |
| Training data provided | train splits and GiftEvalPretrain | none |
| Univariate windows | 371,330 | about 111k |
| Forecast points | 40.67M | about 7.8M |
| Longest horizon | 900 | 864 |

## Tracks

### Track A: one model per configuration (GIFT `deep-learning`)

- Train a DiffGPT from scratch for each configuration on that configuration's
  own `training_dataset`.
- Score it on the validation window: the H steps between the train and test
  splits, one window per series.
- The submitted model may then be refit on `validation_dataset` with the
  selected settings and the same budget. The GIFT feedforward notebook also
  trains on that split.
- A model may serve several terms of one dataset/frequency only if it was
  trained on the earliest of their train splits. The short-term train split
  runs into the long-term test windows. For example, on ett1/H the short
  train split ends 1,008 steps before the end of the series, while the long
  test windows cover the last 2,160 steps.
- Per-configuration settings are chosen on validation: order, normalization,
  context, V, model size and temperature.
- Use one model family throughout. Choosing per configuration between DiffGPT
  and another model, seasonal naive included, counts as a router and would be
  relabeled agentic.
- Label: deep-learning, leakage No, replication code Yes. Compare against the
  deep-learning baselines (PatchTST 0.849 / 0.587).
- Expect weakness on configurations with few or short series, such as
  hospital, car_parts, covid_deaths and restaurant: a model trained from
  scratch has little to learn from there.
- Optional on TIME: per-task models trained on the history before each task's
  validation block. TIME has no rule against it; the PR must say so.

### Track B: one global model, zero-shot (GIFT `zero-shot`)

- Pretrain one DiffGPT on a subset of GiftEvalPretrain (Apache-2.0),
  optionally adding synthetic series, and never using a GIFT-Eval train split.
  The GIFT label is zero-shot, leakage No. The same weights then run on TIME,
  also zero-shot.
- Corpus size: 975 GB in total, of which cmip6_* and era5_* take 856 GB. The
  remaining 119 GB include 63 GB of buildings_900k. With 123 GB of free disk,
  download a measured subset without cmip6, era5 and most of buildings_900k
  (about 56 GB or less).
- For the TIME entry, also leave out azure_vm_traces_2017,
  bitcoin_with_missing, fred_md and cdc_fluview_*, the sources related to
  TIME's. They probably overlap only older history, but dropping them costs
  little.
- Selection: checkpoints, settings and the temperature are chosen on a
  hold-out from the pretraining corpus (held-out series, last windows). GIFT
  validation windows are scored once per candidate as a go/no-go check, never
  for tuning, so the zero-shot label stays honest.
- Model size: start at 1–10M parameters. The nearest precedents:
  - goia-forecast-nano-v0: 4.73M parameters, trained from scratch on a single
    consumer GPU, 0.815;
  - tinycast: 146,505 parameters, about 7.8 h on eight RTX 3090s, 0.774;
  - Timestral-5M: 0.705;
  - TTM-R3: 1.41M parameters, pretrained, 0.724.
- Two new pieces are needed:
  - per-window normalization in the loader, so that one vocabulary serves
    series of every scale;
  - a streaming loader over the Arrow files, since the corpus will not fit the
    in-memory DataFrame loader.
- Variant: the same run plus the GIFT train splits would be labeled
  pretrained, still without leakage. Decide on it after the zero-shot result.

## Evaluation pipeline

The pipeline has three stages in two environments, with files passed between
them:

1. **Export** (eval environment, Python 3.11, with gift-eval and timebench).
   For every configuration or task, write each window's context, truncated to
   the context limit, together with:
   - the configuration key, item id and window index;
   - the frequency, H and seasonality;
   - the order in which the scorer will iterate the windows.

   For Track A, also export the train and validation series.
2. **Forecast** (diff-gpt environment). Normalize each window, sample S
   trajectories, compute the quantiles and the mean, invert the
   normalization, and write one array per configuration.
3. **Score** (eval environment).
   - GIFT: wrap the arrays as gluonts `QuantileForecast` objects with the keys
     "0.025", "0.1", …, "0.9", "0.975" and "mean". MSIS reads the 0.025 and
     0.975 quantiles. Replay them through `evaluate_model` with the notebook's
     arguments and write `all_results.csv`.
   - TIME: pass (N, 9, H) quantiles to
     `save_window_predictions(..., output_base_dir="output/results/<model>")`,
     then score locally with `compute_local_leaderboard.py`.

Adapter details:

- **Variates.** Forecast univariately first (`to_univariate=True`) on both
  benchmarks. Joint multivariate forecasting is a later option; F columns use
  F positions of `block_size` per row.
- **Normalization.** Screen two per-window schemes: dividing by the context's
  mean absolute value (as Chronos does), and z-normalizing with the context's
  mean and standard deviation.
- **Domain.** Set the domain on the normalized steps with
  `get_domain_of_definition(..., quantile=q)`. Use `symmetric=False` where the
  steps lean to one side, such as counts at k = 0.
- **Context length.** Use at least two seasons where the history allows it
  (5T: 576 rows; 10S: 720), within (C + H − k) ≤ `block_size` for univariate
  windows. The longest horizons, 900 on GIFT and 864 on TIME, need about
  1,412 positions at C = 512, so `block_size` 2,048.
- **Batching.** Extend `predict_many` to S samples per context: each batch
  holds contexts × samples, with one prefill per context. To let windows share
  a shape, truncate each context to the largest of a few bucket lengths it
  fills (C, C/2, C/4, …).
- **Missing values.** Contexts are filled with `fill_missing`, and the loss
  masks targets that read a NaN. A context with no values at all gets a
  constant forecast of 0, counted in the run log. Forecasts must never contain
  NaN (`allow_nan_forecast=False`).
- **Point and mean forecasts.** MASE[0.5] uses the 0.5 quantile, the marginal
  median, as in `predict_point(metric="mae")`. The "mean" key is the sample
  mean.
- **Temperature.** Pick it with `calibrate_temperature` (criterion "crps"):
  on validation windows for Track A, on the pretraining hold-out for Track B.
- **Seeds.** Fix the sampling seeds and record them, along with S and the
  temperature, in `config.json`.
- **Locations.** Outputs go under `runs/leaderboards/`. Data goes under
  `datasets/gift-eval`, `datasets/time` and `datasets/gift-eval-pretrain`. All
  of these are ignored by git.
- **Code.** The adapter lives in `leaderboards/` in this repository, with its
  own `pyproject.toml`: Python 3.11, and gift-eval and timebench pinned to
  commits. Its ruff config extends the repository's with target py311, so the
  whole repository stays ruff clean. Keeping it here lets `code_link` point at
  the diff-gpt repository.
- **Fast tests** cover:
  - window order and keys;
  - the normalization round trip;
  - quantile shapes;
  - the replayed seasonal naive matching the scorer's own.

## Compute (estimates until M2 measures them)

One univariate value is one token, so the number of decoded tokens is the
number of forecast points times S:

| | Forecast points | Tokens at S = 20 | Tokens at S = 100 | Prefill tokens at C = 512 |
|---|---:|---:|---:|---:|
| GIFT-Eval | 40.67M | 0.81B | 4.07B | 190M |
| TIME | about 7.8M | 0.16B | 0.78B | 57M |

At a sustained 100k generated tokens per second, a GIFT run at S = 100 would
take about 11 hours and a TIME run about 2 hours; S = 20 takes a fifth of that.
The real rate depends on the model size, the batch size and how much KV cache
fits in 4 GB. M2 replaces these estimates with measurements. Training costs
for Track A (up to 97 models) and Track B (pretraining) are measured in FLOPs
and hours in M3 and M5, before the full runs are scheduled.

## Steps

### M0. Merge the worktree

Done on 2026-10-21: the asymmetric-domain worktree (ranges and NaN handling)
is commit 351f8ca on master. The benchmarks that read `encode`'s scale as an
array (the grokking ones, the `series_*` runners among them, and
`m4_bench.py`) still have to be updated by their owner; their tests fail
until then.

### M1. Evaluation environment and seasonal naive

- Create the uv environment in `leaderboards/` and download GIFT-Eval
  (1.60 GB) and TIME (220 MB).
- Reproduce GIFT's Seasonal_Naive on all 97 configurations, using the
  notebook's code path, and compare it with the published CSV.
- Reproduce TIME's seasonal_naive. It differs between runs because of the
  unseeded noise reported in issue #12 (about 0.7% per task at the median),
  so compare within that margin. Normalize with the hosted results, as
  `compute_local_leaderboard.py` does.
- Replay check: run a numpy seasonal naive in the diff-gpt environment through
  export → forecast → score. It must equal the scorer's own seasonal naive on
  every GIFT configuration. This checks window order, keys, dtypes and
  normalization before any model runs.
- Re-implement the boards' aggregation and check it against the published
  numbers of two existing entries on each board.
- Gate G1: all of these checks pass. This step needs only the CPU.

### M2. Batched sampling and throughput

- Extend `predict_many` to S samples per context. Test it at temperature 0
  against `predict` and check output shapes and NaN contexts.
- Measure generated tokens per second on the RTX 3050 against model size,
  batch size and `block_size` 2,048, and measure the KV-cache memory.
- Choose S from validation CRPS at S = 20, 50 and 100 on the pilot
  configurations, within the compute budget.

### M3. Track A pilot

- Pick a handful of configurations that span frequencies and terms:
  m4_hourly/H/short, ett1/H (short, medium and long), electricity/H/short,
  us_births/D/short and bizitobs_l2c/5T/short.
- Screen one change at a time, scored on the validation windows and compared
  by FLOPs over seeds 11, 17 and 23. The candidates:
  - order: k = 0, k = 1, `Difference(1, m)`;
  - normalization;
  - context length;
  - V;
  - width and depth;
  - learning rate.
- Gate G2: normalized MASE and CRPS below 1 on the pilot's validation windows
  (geometric mean), on all three seeds. If the gate fails, record the failure
  and decide with the user whether Track A continues.

### M4. Track A on all of GIFT-Eval

- Train the 97 models, or fewer shared ones under the train-split rule, with
  the settings from M3.
- Gate G3: the geometric mean over the 97 validation configurations is below
  1 for both MASE and CRPS.
- Then run the test windows once, write `all_results.csv`, run the validator,
  and report against PatchTST and the other deep-learning baselines.
- The rerun command must work from a clean checkout, because a maintainer
  reproduces a subset.

### M5. Track B pretraining

- Download the corpus subset after checking free disk, build the streaming
  loader with per-window normalization, and carve out the hold-out.
- Screen model size and codec on the hold-out, by FLOPs. Confirm the selected
  setting on three seeds where the budget allows.
- Gate G4: on GIFT validation windows, scored once per candidate, normalized
  MASE and CRPS are below 1.

### M6. TIME

- Run the Track B weights zero-shot on all 98 tasks; optionally add Track A
  per-task models.
- Score the validation blocks as a go/no-go check, then the test windows once,
  and compute the local leaderboard.

### M7. Submissions, only after the user's go-ahead

Follow the checklists below.

## Criterion and gates

- Scores are computed as the boards compute them: per-configuration MASE and
  CRPS divided by seasonal naive's, combined by a geometric mean over 97 GIFT
  configurations or 98 TIME tasks.
- The gates in order:
  - G1: the pipeline reproduces seasonal naive;
  - G2: the Track A pilot beats seasonal naive on validation;
  - G3: Track A beats seasonal naive on all 97 validation configurations;
  - G4: the Track B go/no-go check;
  - then the user's go-ahead to submit.
- Keep failed screens and negative results in the record, as
  `grokking_plan.md` does.
- Compare normalized scores, not ranks: ranks shift with every new entry.

## Pitfalls

- **Local copies of GIFT test data.** The sources of GIFT's ett1/ett2,
  electricity, jena_weather and m4_* test sets, test periods included, are
  already on disk:
  - `datasets/long_term_forecast/ETT-small`;
  - `datasets/long_term_forecast/electricity`;
  - `datasets/long_term_forecast/weather`;
  - `datasets/short_term_forecast/m4`.

  Never pretrain Track B on these files, and never submit a checkpoint trained
  on them, such as those from the README benchmarks or the grokking-series
  runs. Track A trains only on GIFT's own splits. Settings tuned on these
  files are only starting points; select the submitted settings on GIFT
  validation.
- **Terms share series.** A model trained on a short-term train split has
  seen the long-term test windows of the same dataset (see ett1/H above).
  Train per configuration, or on the earliest train split of the terms the
  model serves.
- **Related sources on TIME.** fev-bench `epf_*` comes from the same source as
  TIME's `epf_electricity_price/H`; never train on it. The local `illness`
  data (CDC ILINet) belongs to the same family as the cdc_fluview sources that
  are related to TIME's data, so keep it out of Track B.
- **Blank cells.** The GIFT validator accepts blank metric cells, and the
  Space fills them with the column mean over all models. An entry must have
  all 97 rows complete; never exploit the blank-cell rule.
- **Missing keys.**
  - A `QuantileForecast` without "mean" silently makes the [mean] metrics use
    the median.
  - Export 0.025 and 0.975 so that MSIS reads quantiles the model actually
    produced.
- **NaN forecasts** fail the evaluation (`allow_nan_forecast=False`).
- **M4 windows.** The paper lists 2 windows for M4 Hourly, but the code and
  the board use 1.
- **Dataset names.** Names in the code differ from the CSV names (for
  example, saugeenday becomes saugeen). Key the CSV rows by
  `dataset_configurations.csv`.
- **Routers.** A per-configuration choice between DiffGPT and any other model
  is routing. Per-configuration hyperparameters of per-configuration models
  are not.
- **TIME baseline noise** (issue #12). Normalize with the hosted
  seasonal_naive results; do not rerun them.
- **TIME license.** The data is CC BY-NC 4.0. Keep it in the ignored
  `datasets/` folder and never commit or redistribute it.
- **Trust.** TIME trusts submitted metrics. Publish the code and the exact
  command, and submit the `predictions.npz` from the same run as
  `metrics.npz`.
- **Long horizons.** Unless (C + H − k) · F ≤ `block_size`, `predict` raises.
  Long terms need `block_size` 2,048 at C = 512.
- **Over-dispersion.** Samples at temperature 1 are too wide. Calibrate the
  temperature by CRPS on validation.
- **Clipping.** Steps beyond the domain clip. A quantile domain trades
  clipping for resolution, as on M4 Daily. Count the clipped tokens per
  configuration.
- **Ranks drift** with every new entry; compare normalized scores instead.
- **TIME caching.** New TIME results appear only after the Space restarts.

## Submission checklists

### GIFT-Eval

- [ ] All 97 configuration rows are present; every metric cell is finite and
      ≥ 0; no cell is blank.
- [ ] The folder name, the `model` field in `config.json` and every `model`
      cell in the CSV are identical.
- [ ] `config.json` sets:
  - model_type: deep-learning for Track A, zero-shot for Track B;
  - testdata_leakage: "No";
  - replication_code_available: "Yes";
  - model_dtype, model_link, code_link and org.
- [ ] `scripts/validate_results.py` passes.
- [ ] Code and weights are published where the user chooses, and the rerun
      command works from a clean checkout.
- [ ] The user has signed the Salesforce CLA, which is in their name.
- [ ] The user has given the go-ahead.

### TIME

- [ ] All 98 folders are present, each with `predictions.npz`,
      `metrics.npz` and `config.json` written by `save_window_predictions`.
- [ ] `compute_local_leaderboard.py` reproduces the numbers we report.
- [ ] The PR text states the parameter count, the training corpus, zero-shot
      or per-task training, and the code link.
- [ ] The user has given the go-ahead; the upload uses the user's HF token.

## Decisions for the user

- The model and organization names shown on the boards.
- Where the code and weights are published. GIFT requires `model_link` and
  `code_link`, and a maintainer reruns the model.
- Whether to submit Track A, Track B or both.
- Signing the Salesforce CLA.
- Downloading the GiftEvalPretrain subset: tens of GB out of 123 GB free.

## First steps

1. M1: create the evaluation environment, download both test sets (about
   1.8 GB), reproduce seasonal naive on both benchmarks and run the replay
   check. This needs only the CPU, so it does not compete with the GPU queue.
2. M2: build batched sampling and measure throughput on a representative
   model.
3. M3: start the Track A pilot on m4_hourly and ett1/H.

## Sources

All sources were read on 2026-10-21.

- **GIFT-Eval**
  - github.com/SalesforceAIResearch/gift-eval: README,
    `scripts/validate_results.py`, `scripts/dataset_configurations.csv`,
    `src/gift_eval/data.py`, `notebooks/`, issues 46 and 218, pulls 209, 229
    and 234, commit d11d3ab (model-type rules).
  - Space huggingface.co/spaces/Salesforce/GIFT-Eval: `src/utils.py`, `app.py`
    and the `/config` table.
  - Paper: arXiv 2410.10393v2.
  - Data: huggingface.co/datasets/Salesforce/GiftEval and
    huggingface.co/datasets/Salesforce/GiftEvalPretrain.
  - gluonts v0.15.1: `time_feature/seasonality.py` and `model/forecast.py`.
- **TIME**
  - Paper: arXiv 2602.12147v4.
  - github.com/zqiao11/TIME: README, `src/timebench/evaluation/`
    (`data.py`, `metrics.py`, `saver.py`), `experiments/chronos2.py`,
    `scripts/compute_local_leaderboard.py`, issue 12 and PR 4.
  - Space huggingface.co/spaces/Real-TSF/TIME-leaderboard:
    `src/leaderboard.py`, `src/utils.py`, `src/hf_config.py` and the `/config`
    table.
  - Data: huggingface.co/datasets/Real-TSF/TIME.
  - Results: huggingface.co/datasets/Real-TSF/TIME-Output, discussions 36,
    42, 46, 48, 54 and 55.
  - fev-bench: `benchmarks/fev_bench/tasks.yaml` in github.com/autogluon/fev.
