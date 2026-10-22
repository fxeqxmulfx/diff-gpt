# leaderboards

The GIFT-Eval and TIME evaluation environment for diff-gpt. It runs on
Python 3.11 with gift-eval pinned to commit 3c7be6d, TIME's timebench to
commit c633e23 and gluonts 0.15.1. diff-gpt runs on Python 3.14 in its own
environment, so the two exchange files and gluonts never enters diff-gpt:

1. **export** (this environment) writes the windows of each configuration;
2. a **forecaster** (the root environment, `benchmarks/gift_eval/files.py`)
   reads them and writes quantile forecasts;
3. **score** (this environment) replays the forecasts through gluonts'
   `evaluate_model` with the leaderboard's metrics;
4. **aggregate** divides each configuration's MASE and CRPS by
   Seasonal_Naive's and takes the geometric mean over the 97 configurations.

TIME (`leaderboards.time`, below) takes the same steps over its 98 tasks,
with timebench's `save_window_predictions` as the scorer.

## Setup

```sh
cd leaderboards && uv sync    # Python 3.11; gift-eval comes from git
git clone https://github.com/SalesforceAIResearch/gift-eval ../../gift-eval
git -C ../../gift-eval checkout 3c7be6deffb0046e1c08f9103db8bc84b3cf7ebd
uvx --from 'huggingface_hub[hf_xet]' hf download Salesforce/GiftEval \
    --repo-type=dataset --local-dir ../datasets/gift-eval    # 1.5 GB
```

The clone provides the configuration list (`scripts/dataset_configurations.csv`,
`notebooks/dataset_properties.json`) and the published results. If the root
venv is active, run `unset VIRTUAL_ENV` first, or uv warns on every command.

## Commands

From `leaderboards/`, unless noted. Everything is written under
`runs/leaderboards/gift-eval/`.

```sh
uv run python -m leaderboards.gift.export                  # windows/{test,val}/, 50 s
# From the repository root, in diff-gpt's environment:
uv run python -m benchmarks.gift_eval.seasonal_naive --split test   # forecasts/seasonal_naive/test/
uv run python -m leaderboards.gift.score --model seasonal_naive --split test
uv run python -m leaderboards.gift.aggregate \
    ../runs/leaderboards/gift-eval/results/seasonal_naive/test/all_results.csv
uv run python -m leaderboards.gift.official --split test --jobs 2   # statsforecast reference
```

Every command takes `--only KEY ...` (e.g. `m4_hourly/H/short`), and `--help`
describes the file formats. `aggregate --split test` divides by the published
Seasonal_Naive and `--split val` by ours
(`results/seasonal_naive/val/all_results.csv`); `--baseline CSV` overrides.

## Splits

- **test**: gift-eval's `Dataset.test_data`, the leaderboard's windows.
- **val**: one window per series, the horizon right before the test split,
  with everything before it as context. No labels of either split are
  exported.

Contexts are cut to their last 4096 steps (`export --max-context`). The
seasonal naive's interval width depends on the history, so a forecaster that
sees 4096 steps is scored on the same windows as the board but with less
history than the board's Seasonal_Naive had.

## Tests

```sh
uv run pytest                                   # here: names, windows, replay
cd .. && uv run pytest tests/gift_eval          # the root side
```

The replay test runs the root environment's `benchmarks.gift_eval.seasonal_naive`
in a subprocess (`../.venv/bin/python`) on tiny datasets written to disk in
the download's format, and requires the scored result to equal the
statsforecast reference. Nothing is downloaded. The tests of the task list
and of the aggregation read the gift-eval clone and `../datasets/time-output`.

## TIME

TIME's leaderboard (the Hugging Face Space Real-TSF/TIME-leaderboard) scores
98 tasks, a dataset at one term each. timebench at c633e23 provides the
datasets, their lengths (`config/datasets.yaml`) and the scorer, which writes
`predictions.npz`, `metrics.npz` and `config.json` per task: for test, the
folder a submission uploads.

```sh
uvx --from 'huggingface_hub[hf_xet]' hf download Real-TSF/TIME \
    --repo-type=dataset --local-dir ../datasets/time    # 212 MB
```

The hosted results come from the dataset Real-TSF/TIME-Output at revision
4c93dea (`huggingface_hub.snapshot_download` with
`allow_patterns=["results/<model>/**/metrics.npz", "results/<model>/**/config.json"]`
for seasonal_naive, Chronos-2 and TiRex, into `../datasets/time-output`).
TIME's data is CC BY-NC 4.0: it stays in the ignored `datasets/` and is never
committed.

From `leaderboards/`, unless noted; everything is written under
`runs/leaderboards/time/`.

```sh
uv run python -m leaderboards.time.export                  # windows/{test,val}/
# From the repository root, in diff-gpt's environment:
uv run python -m benchmarks.gift_eval.seasonal_naive --split test \
    --windows runs/leaderboards/time/windows --forecasts runs/leaderboards/time/forecasts
uv run python -m leaderboards.time.score --model seasonal_naive --split test --jobs 8
uv run python -m leaderboards.time.aggregate \
    ../runs/leaderboards/time/results/seasonal_naive/test
uv run python -m leaderboards.time.official --split test --jobs 8   # TIME's seasonal_naive
uv run python -m leaderboards.time.compare A B             # two results folders
```

`official` runs timebench's SeasonalNaivePredictor as TIME's
`experiments/seasonal_naive.py` does: 100 paths drawn between statsforecast's
quantiles, unseeded (TIME's issue #12), so two runs differ by about 0.7% per
task; `--seed` makes a run repeatable, and `--samples 0` scores
statsforecast's quantiles themselves, which the numpy seasonal naive
reproduces. `aggregate` takes the nanmean of each task's metrics, divides by
seasonal_naive's (the hosted one for test, ours for val) and takes the
geometric mean, as the Space does. Keep a local seasonal naive out of folders
named `seasonal_naive` in a results directory that TIME's
`compute_local_leaderboard.py` reads: that script prefers it to the hosted one.

- **test**: timebench's `Dataset.test_data`, the last `test_length` steps of
  each series cut into windows of the task's horizon.
- **val**: `Dataset.val_data`, the `val_length` steps right before them, cut
  the same way. Each window has its whole history as context, cut to the
  last 4096 steps by `export`; multivariate series are split into one
  series per variate.
