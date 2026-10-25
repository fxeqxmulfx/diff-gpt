# leaderboards

The GIFT-Eval and TIME evaluation for diff-gpt: the adapters in
`diff_gpt.infrastructure.leaderboards`, run by the command lines in
`diff_gpt.interfaces.leaderboards`, in the repository's one environment on
Python 3.14. It runs copies of the code it uses from gift-eval at commit
3c7be6d, TIME's timebench at commit c633e23, gluonts 0.15.1 and
statsforecast 2.1.1 (`vendored/`; each file names its source, its license
and its changes, and the licenses are the `LICENSE-*` files here). The rest
of diff-gpt imports none of it, so the copies never enter the model or its
forecasters; the two exchange files:

1. **export** (leaderboards) writes the windows of each configuration;
2. a **forecaster** (diff-gpt, `diff_gpt.application.gift_eval.files`) reads
   them and writes quantile forecasts;
3. **score** (leaderboards) replays the forecasts through gluonts'
   `evaluate_model` with the leaderboard's metrics;
4. **aggregate** divides each configuration's MASE and CRPS by
   Seasonal_Naive's and takes the geometric mean over the 97 configurations.

TIME (`diff_gpt.infrastructure.leaderboards.time`, below) takes the same
steps over its 98 tasks, with timebench's `save_window_predictions` as the
scorer.

## Setup

`uv sync` in the repository root installs the leaderboards' dependencies,
the `leaderboards` group of the root's `pyproject.toml` (part of `dev`), into
the root's `.venv`. Then, from the repository root:

```sh
git clone https://github.com/SalesforceAIResearch/gift-eval ../gift-eval
git -C ../gift-eval checkout 3c7be6deffb0046e1c08f9103db8bc84b3cf7ebd
uvx --from 'huggingface_hub[hf_xet]' hf download Salesforce/GiftEval \
    --repo-type=dataset --local-dir datasets/gift-eval    # 1.5 GB
```

The clone provides the configuration list (`scripts/dataset_configurations.csv`,
`notebooks/dataset_properties.json`) and the published results.

## Commands

From the repository root. Everything is written under
`runs/leaderboards/gift-eval/`.

```sh
uv run python -m diff_gpt.interfaces.leaderboards.gift.export   # windows/{test,val}/, 50 s
uv run python -m diff_gpt.interfaces.gift_eval.seasonal_naive --split test   # forecasts/seasonal_naive/test/
uv run python -m diff_gpt.interfaces.leaderboards.gift.score --model seasonal_naive --split test
uv run python -m diff_gpt.interfaces.leaderboards.gift.aggregate \
    runs/leaderboards/gift-eval/results/seasonal_naive/test/all_results.csv
uv run python -m diff_gpt.interfaces.leaderboards.gift.official --split test --jobs 2   # statsforecast reference
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
uv run pytest tests/diff_gpt/infrastructure/leaderboards   # names, windows, replay
uv run pytest tests/diff_gpt/application/gift_eval         # the forecasters' side
```

The replay test runs diff-gpt's `diff_gpt.interfaces.gift_eval.seasonal_naive`
in a subprocess (`.venv/bin/python`) on tiny datasets written to disk in
the download's format, and requires the scored result to equal the
statsforecast reference. Nothing is downloaded. The tests of the task list
and of the aggregation read the gift-eval clone and `datasets/time-output`.

## TIME

TIME's leaderboard (the Hugging Face Space Real-TSF/TIME-leaderboard) scores
98 tasks, a dataset at one term each. timebench at c633e23, copied, provides
the datasets, their lengths (`config/datasets.yaml`) and the scorer, which writes
`predictions.npz`, `metrics.npz` and `config.json` per task: for test, the
folder a submission uploads.

```sh
uvx --from 'huggingface_hub[hf_xet]' hf download Real-TSF/TIME --repo-type=dataset \
    --revision 83e3d0b3be28d11c7182bffcc1892d19b36c4da1 \
    --local-dir datasets/time    # 212 MB
uvx --from 'huggingface_hub[hf_xet]' hf download Real-TSF/TIME-Output --repo-type=dataset \
    --revision 4c93deaac0aa02e3be3ae385be7a8be6dde7aa91 \
    --include 'results/seasonal_naive/**' --include 'results/Chronos-2/**' \
    --include 'results/TiRex/**' --exclude '**/predictions.npz' \
    --local-dir datasets/time-output    # 588 files, 20 MB
```

The first is TIME's data; the second, the hosted `metrics.npz` and
`config.json` of seasonal_naive (the baseline `aggregate` divides test by),
Chronos-2 and TiRex (whose leaderboard scores
`tests/diff_gpt/infrastructure/leaderboards/test_time.py` reproduces). Gate G1
ran on these revisions, both repositories' `main` on 2026-10-22. With
`--dry-run` either command counts the files it would fetch: none for a copy at
its revision. TIME's data is CC BY-NC 4.0: it stays in the ignored `datasets/`
and is never committed.

From the repository root; everything is written under
`runs/leaderboards/time/`.

```sh
uv run python -m diff_gpt.interfaces.leaderboards.time.export   # windows/{test,val}/
uv run python -m diff_gpt.interfaces.gift_eval.seasonal_naive --split test \
    --model numpy_seasonal_naive \
    --windows runs/leaderboards/time/windows --forecasts runs/leaderboards/time/forecasts
uv run python -m diff_gpt.interfaces.leaderboards.time.score --model numpy_seasonal_naive --split test --jobs 8
uv run python -m diff_gpt.interfaces.leaderboards.time.aggregate \
    runs/leaderboards/time/results/numpy_seasonal_naive/test
uv run python -m diff_gpt.interfaces.leaderboards.time.official --split test --jobs 8   # TIME's seasonal_naive
uv run python -m diff_gpt.interfaces.leaderboards.time.compare A B   # two results folders
```

`official` runs timebench's SeasonalNaivePredictor as TIME's
`experiments/seasonal_naive.py` does: 100 paths drawn between statsforecast's
quantiles, unseeded (TIME's issue #12), so two runs differ by about 0.6% per
task. `--seed` makes a run repeatable: each task's draws are seeded with the
CRC-32 of `<seed>:<task key>`, so the tasks of a run do not repeat each
other's draws. `--samples 0` scores statsforecast's quantiles themselves,
which the numpy seasonal naive reproduces. `aggregate` takes the nanmean of
each task's metrics, divides by a seasonal naive's (the hosted seasonal_naive
for test, `results/numpy_seasonal_naive/val` for val) and takes the geometric
mean, as the Space does. Keep a local seasonal naive out of folders named
`seasonal_naive` in a results directory that TIME's
`compute_local_leaderboard.py` reads: that script prefers it to the hosted one.

The hosted seasonal_naive drew differently: its `predictions.npz` show the
paths of a window drawn once for all the variates of a multivariate series,
whereas TIME's code today, like `official`, splits a series into its
variates and draws for each. The expectation is the same, but a task's
scores vary more from one such run to the next. Ten runs of `official` on
test (100 draws, full contexts) measured:

| | MASE | CRPS |
|---|---|---|
| one run against another, median relative difference per task | 0.45-0.79% | 0.44-0.75% |
| a run against the hosted one, the same | 0.56-0.93% | 0.51-0.82% |
| a run's `aggregate` over the hosted one | 0.991-1.002 | 0.996-1.004 |

Statsforecast's own quantiles (`--samples 0 --max-length 4096`, which the
numpy seasonal naive reproduces) aggregate to MASE 0.9660 and CRPS 0.9808
over the hosted one.

- **test**: timebench's `Dataset.test_data`, the last `test_length` steps of
  each series cut into windows of the task's horizon.
- **val**: `Dataset.val_data`, the `val_length` steps right before them, cut
  the same way. Each window has its whole history as context, cut to the
  last 4096 steps by `export`; multivariate series are split into one
  series per variate.
