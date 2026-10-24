# Plan: the GIFT-Eval and TIME leaderboards

Started: 2026-10-21. Status on 2026-10-22: M1 is done on GIFT-Eval, where
gate G1 passes, and under way on TIME; Track B's corpus comes from TempoPFN's
generators (`synthetic/`) and is not generated in full yet. The work moved to
another machine on 2026-10-22. See **Status on 2026-10-22** near the end.

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
- Hardware: until 2026-10-22, a laptop with one RTX 3050 Laptop GPU (4 GB),
  29 GiB of RAM, 6 cores (12 threads) and 123 GB of free disk, where jobs
  used at most 4 CPU threads in all (`taskset`, one thread per library).
  Since 2026-10-22, a machine whose resources the user gave over in full:
  two Xeon E5-2680 v4 (Broadwell: AVX2, no AVX-512, no native bfloat16), two
  NUMA nodes of 14 cores and 28 threads (node 0: CPUs 0-13 and 28-41, node 1:
  14-27 and 42-55), 121 GiB of RAM, about 220 GB free on `/`, and a GTX 1050
  with 2 GB (sm_61, which torch 2.14's sm_60 build runs). Matrix products
  measured there on 2026-10-22: the GPU does 1.54 TFLOPS in float32, 1.85 in
  float16 and 1.06 in bfloat16, which it emulates, so training there stays
  float32. One process pinned to one socket does about 1.0 TFLOPS in float32
  on the CPU, two such processes about 2.0 in all, and one process across
  both sockets 0.16-0.22: CPU jobs run one process per NUMA node
  (`diff_gpt.numa.pin_to_node`). Numbers measured on the RTX 3050 are the
  laptop's; every estimate is checked against the machine it runs on.
- Python versions: diff-gpt requires Python ≥ 3.14. gift-eval and timebench
  pin gluonts ~0.15.1 and numpy ~1.26, which do not run on 3.14, so the
  evaluation, `leaderboards/`, runs copies of the code it uses from them, from
  gluonts and from statsforecast; diff-gpt imports none of it, and the two
  sides exchange files. The synthetic corpus is made by `synthetic/` with
  TempoPFN's code from its clone; it hands diff-gpt .npy files. The three
  share one uv workspace, one lock and one environment, the root's `.venv`.
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
- Commit after each step that makes progress (the user's rule since
  2026-10-22), and stage explicit paths, because other sessions share the
  working tree. Push only when the user asks.

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

- Pretrain one DiffGPT on synthetic series from TempoPFN's generators, and
  never on a GIFT-Eval or TIME split. The GIFT label is zero-shot, leakage No.
  The same weights then run on TIME, also zero-shot. The user chose this on
  2026-10-21: TempoPFN's model is dropped; its generators and its GIFT-Eval
  path are kept. TempoPFN itself, 38M parameters pretrained on its synthetic
  series, scores 0.797 / 0.537 on GIFT-Eval, zero-shot.
- The corpus is made in `synthetic/` (see the docstrings of `src/__init__.py`
  and `tempopfn_corpus/`):
  - TempoPFN's code runs from a clone next to this repository,
    `../TempoPFN` at commit 5969ec6; a run refuses another commit or a
    changed clone. Four of its modules are replaced by stand-ins in
    `synthetic/src`. Among them, KernelSynth samples its GP prior by a
    pivoted Cholesky factorization of a kernel matrix built from lags, and
    CauKer draws its roots independently on the CPU, as the original CauKer
    does, where TempoPFN's port gave all roots of a series one noise.
  - 16 sources, 14 generators and 2 offline augmentations, of series of
    2,048 steps. The full corpus has 65,536 train and 4,096 holdout series
    per source: about 20 CPU hours, 61% of them in gp, or at least 7 hours
    on three cores of the laptop.
  - Series j of a source gets the seed run_id + j in whichever process makes
    it, with PYTHONHASHSEED=0, so the corpus can be built in parts.
  - `tempopfn_corpus.export` writes `npy/<split>/<source>.npy`, float32 of
    shape (N, 2048), and `manifest.json`.
  - A pilot of 5,120 series per source, made by earlier stand-ins, is set
    aside in `datasets/tempopfn-synthetic/set-aside/pilot`.
- Never read TempoPFN's `data/nan_stats.json`: it is computed from the
  GIFT-Eval train splits, and `generate` refuses to open it. TempoPFN draws
  each series' frequency from GIFT_EVAL_FREQUENCY_WEIGHTS, the benchmark's mix
  of frequencies; on 2026-10-22 the user kept it as TempoPFN has it, as
  metadata rather than data.
- Training: `benchmarks/synthetic_pretrain.py`, in diff-gpt's environment.
  `DiffCorpusDataLoader` takes windows of the train split, each in the units
  of a prefix of itself, so that one vocabulary serves series of every
  scale. `--train-dtype bfloat16` runs bfloat16 forward passes on float32
  master weights, gradients and optimizer state.
- Selection: checkpoints, settings and the temperature are chosen on the
  holdout split alone, by the MASE and CRPS of forecasts relative to the
  naive forecast, geometric mean over the sources; `best.bin` holds the
  lowest holdout CRPS. GIFT validation windows are scored once per candidate
  as a go/no-go check, never for tuning, so the zero-shot label stays honest.
- `benchmarks/gift_eval/diff_gpt_forecast.py` forecasts the exported GIFT
  windows with a `synthetic_pretrain` run.
- Model size: start at 1–10M parameters. The nearest precedents:
  - goia-forecast-nano-v0: 4.73M parameters, trained from scratch on a single
    consumer GPU, 0.815;
  - tinycast: 146,505 parameters, about 7.8 h on eight RTX 3090s, 0.774;
  - Timestral-5M: 0.705;
  - TTM-R3: 1.41M parameters, pretrained, 0.724.
- GiftEvalPretrain (Apache-2.0) stays an option to add later. It holds
  975 GB, of which cmip6_* and era5_* take 856 GB and buildings_900k 63 GB;
  a subset without them and most of buildings_900k is about 56 GB. For the
  TIME entry, such a subset would also leave out azure_vm_traces_2017,
  bitcoin_with_missing, fred_md and cdc_fluview_*, the sources related to
  TIME's.
- Variant: the same run plus the GIFT train splits would be labeled
  pretrained, still without leakage. Decide on it after the zero-shot result.

## Evaluation pipeline

The pipeline has three stages, with files passed between them:

1. **Export** (`leaderboards/`, with its copies of gift-eval and timebench).
   For every configuration or task, write each window's context, truncated to
   the context limit, together with:
   - the configuration key, item id and window index;
   - the frequency, H and seasonality;
   - the order in which the scorer will iterate the windows.

   For Track A, also export the train and validation series.
2. **Forecast** (diff-gpt). Normalize each window, sample S
   trajectories, compute the quantiles and the mean, invert the
   normalization, and write one array per configuration.
3. **Score** (`leaderboards/`).
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
  `datasets/gift-eval`, `datasets/time`, `datasets/tempopfn-synthetic` and,
  if it is ever downloaded, `datasets/gift-eval-pretrain`. All of these are
  ignored by git.
- **Code.** The adapter lives in `leaderboards/` in this repository, with its
  own `pyproject.toml` in the root's uv workspace and copies of the code it
  uses from gift-eval at commit 3c7be6d and timebench at commit c633e23. Its
  ruff config extends the repository's with target py314, so the whole
  repository stays ruff clean.
  Keeping it here lets `code_link` point at the diff-gpt repository. The
  GIFT-Eval side is built: export, score, aggregate and the official
  Seasonal_Naive (`leaderboards/README.md`). The export cuts contexts to
  their last 4,096 steps.
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

Measured on the laptop's RTX 3050 on 2026-10-21: training runs 1.6–1.7 times
as fast in bfloat16 as in float32, and decoding 2.8–3.4 times as fast in
float16. An untrained DiffGPT of 0.92M parameters forecast GIFT windows at
55–60k points per second in float16, against 16k in float32: about 1 hour
for validation and 4 hours for test, by that smoke run's estimate.

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

- Extend `predict_many` to S samples per context. Done:
  `DiffGPT.predict_samples_many` prefills each context once and copies it
  into the rows of its samples.
- Measure generated tokens per second against model size, batch size and
  `block_size` 2,048, and measure the KV-cache memory. The first numbers,
  from the laptop's RTX 3050, are under **Compute**; measure again on the
  new machine.
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

- Generate the full synthetic corpus on one machine and export it (see
  Track B). The loader with per-window units and the holdout split exist.
- Pilot on the GPU: float32 against bfloat16 training, by FLOPs and holdout
  scores, and the number of samples S.
- Screen model size and codec on the holdout, by FLOPs. Confirm the selected
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
- Whether to add a GiftEvalPretrain subset to the synthetic corpus: tens of
  GB.

Decided with the user on 2026-10-21 and 22:

- Track B pretrains on TempoPFN's generators; TempoPFN's model is dropped.
- Master weights stay float32; bfloat16 serves the forward passes only.
- CauKer's roots are independent, as in the original CauKer, and the full
  corpus is generated from scratch after G1.
- GIFT_EVAL_FREQUENCY_WEIGHTS stay as TempoPFN has them.

## Status on 2026-10-22

- GitHub has master at 32e1bf1; the later commits are local.
- M1 on GIFT-Eval is done (`leaderboards/README.md`), rerun from the export
  on the new machine on 2026-10-22: the export took 74 s, the numpy
  Seasonal_Naive's forecasts 46 s (test) and 16 s (val), and six scoring
  runs of `--jobs 8` ran at once with at least 90 GiB of RAM left. Gate G1
  passes on GIFT-Eval:
  - statsforecast's Seasonal_Naive with full contexts
    (`leaderboards.gift.official`) equals the published Seasonal_Naive on
    all 97 configurations, in all 11 metrics: the largest relative
    difference is 0;
  - the numpy Seasonal_Naive, through export, forecast and score, equals
    statsforecast's with contexts of 4,096 steps on all 97 configurations
    of test and of val, again with a largest difference of 0;
  - the aggregation reproduces the published PatchTST (0.849, 0.587) and
    TiRex (0.716, 0.488), a test in `leaderboards/tests/test_gift.py`.

  Cutting the contexts to 4,096 steps changes only the intervals: on test
  the cut Seasonal_Naive scores MASE 1.0000 and CRPS 0.9648 of the
  published one, and on val the full contexts score CRPS 1.0512 of the cut.
- M1 on TIME: under way. The data (Real-TSF/TIME at 83e3d0b, 212 MB) is in
  `datasets/time`, and the hosted results of seasonal_naive, Chronos-2 and
  TiRex (Real-TSF/TIME-Output at 4c93dea) in `datasets/time-output`.
- M2: `predict_samples_many` is done; throughput is measured on the laptop
  only; S is not chosen.
- Track B: `synthetic/` is written and tested (14 tests pass) and the
  speeds of its sources are measured; the full corpus is not generated, and
  `synthetic_pretrain.py` has not trained on a corpus yet.
- M3, M4 and M6: not started.

## Next steps

In the order agreed with the user:

1. Finish G1 with TIME's part of M1 (GIFT-Eval's passed on 2026-10-22).
2. Generate the full corpus, from `synthetic/`:

   ```sh
   PYTHONHASHSEED=0 taskset -c <cores> uv run python -m tempopfn_corpus.schedule \
       --series train=65536 --series holdout=4096
   uv run python -m tempopfn_corpus.export
   ```

3. Pilot `synthetic_pretrain.py` on the GPU: float32 against bfloat16
   training, and S.
4. Train Track B and check G4.

## Moving to another machine

Git carries the code only. On the new machine:

- Clone next to the repository:
  - TempoPFN at the commit the corpus checks:
    `git clone https://github.com/automl/TempoPFN ../TempoPFN`, then
    `git -C ../TempoPFN checkout 5969ec6ddbded5a8d2976d42504c1ae436a2f1b3`;
  - gift-eval at 3c7be6d, as `leaderboards/README.md` says.
- Run `uv sync` in the root (Python 3.14): one environment, `.venv`, for
  diff-gpt, `leaderboards/` and `synthetic/`. pyo's wheel bundles PortAudio
  and its other libraries, so the audio generators need no system packages.
- Data, ignored by git: download GIFT-Eval again (`leaderboards/README.md`)
  or copy `datasets/gift-eval`. `datasets/long_term_forecast` and
  `datasets/short_term_forecast` serve only the older benchmarks, never
  Track B (see **Pitfalls**). The set-aside pilot,
  `datasets/tempopfn-synthetic/set-aside`, is not needed.
- Outputs, ignored by git: the laptop's `runs/leaderboards/gift-eval` holds
  5.5 GB of partial G1 outputs. Rerun G1 on the new machine from the export
  on (the export takes about 50 s) rather than compare scores made on two
  machines.
- Generate the whole corpus on one machine. Its run records hold the
  TempoPFN commit, the digests of the stand-ins, PYTHONHASHSEED and the
  versions of Python and the libraries, but not the CPU, whose BLAS kernels
  may change the last bits of a series.
- Measure again before scheduling: `SECONDS_PER_SERIES` in
  `synthetic/tempopfn_corpus/schedule.py`, the GPU's throughput (M2), and
  the CPU threads and memory that jobs may use.

## Sources

All sources were read on 2026-10-21, TempoPFN's also on 2026-10-22.

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
- **TempoPFN**
  - github.com/automl/TempoPFN at commit 5969ec6 (Apache-2.0): the synthetic
    generators, `continuous_generation.py`, the offline augmentations and
    `src/data/frequency.py`.
  - Paper: arXiv 2510.25502.
  - The original CauKer: github.com/ShifengXIE/CauKer, `CauKer.py` at
    ed00670.
