# leaderboards

The GIFT-Eval evaluation environment for diff-gpt. It runs on Python 3.11 with
gift-eval pinned to commit 3c7be6d and gluonts 0.15.1. diff-gpt runs on Python
3.14 in its own environment, so the two exchange files and gluonts never
enters diff-gpt:

1. **export** (this environment) writes the windows of each configuration;
2. a **forecaster** (the root environment, `benchmarks/gift_eval/files.py`)
   reads them and writes quantile forecasts;
3. **score** (this environment) replays the forecasts through gluonts'
   `evaluate_model` with the leaderboard's metrics;
4. **aggregate** divides each configuration's MASE and CRPS by
   Seasonal_Naive's and takes the geometric mean over the 97 configurations.

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
statsforecast reference. Nothing is downloaded.
