"""
Probabilistic-forecasting variant of m4_bench.py on M4 Hourly.

Same training setup (one global diff-gpt over 414 series, σ=2 soft-CE)
but at inference we generate N=100 Monte-Carlo trajectories per series
at temperature 1 and report probabilistic metrics alongside the point
metrics:

  - sMAPE, MASE — using the per-step median as the point forecast
  - CRPS        — proper scoring rule; mean over series
  - MSIS        — M4's mean scaled interval score (α=0.05 → 95% interval)
  - coverage@80 — empirical frequency of true values inside the P10-P90 band
  - pinball@0.5 — median-level quantile loss

Baseline: seasonal-naive + residual empirical quantiles (bootstrap). A
calibrated model should at minimum match this on MSIS/CRPS.

Usage:
    uv run --no-sync python -m benchmarks.prob_m4_bench
    M4_NUM_SAMPLES=200 M4_N_SERIES=50 uv run --no-sync python -m benchmarks.prob_m4_bench
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diff_gpt.data_loader import DiffDataFrameDataLoader
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import get_domain_of_definition
from diff_gpt.metrics import (
    crps_empirical,
    empirical_coverage,
    msis,
    pinball_loss,
)
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.nucleus import NucleusSampler
from diff_gpt.sampler.temperature import TemperatureSampler


M4_DIR = (
    Path(__file__).resolve().parents[1] / "datasets" / "short_term_forecast" / "m4"
)

HORIZONS = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Weekly": 13, "Daily": 14, "Hourly": 48}
SEASONALITY = {"Yearly": 1, "Quarterly": 4, "Monthly": 12, "Weekly": 1, "Daily": 7, "Hourly": 24}


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    num = np.abs(true - pred)
    den = np.abs(true) + np.abs(pred)
    mask = den > 0
    return float(200.0 * np.mean(num[mask] / den[mask])) if mask.any() else 0.0


def mase(pred: np.ndarray, true: np.ndarray, insample: np.ndarray, season: int) -> float:
    if len(insample) <= season:
        return float("nan")
    naive = float(np.mean(np.abs(insample[season:] - insample[:-season])))
    if naive == 0.0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / naive)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    freq = os.environ.get("M4_FREQ", "Hourly")
    n_series = int(os.environ.get("M4_N_SERIES", "50"))
    num_samples = int(os.environ.get("M4_NUM_SAMPLES", "100"))
    temperature = float(os.environ.get("M4_TEMPERATURE", "1.0"))
    horizon = HORIZONS[freq]
    season = SEASONALITY[freq]

    info = pd.read_csv(M4_DIR / "M4-info.csv")
    train = np.load(M4_DIR / "training.npz", allow_pickle=True)
    test = np.load(M4_DIR / "test.npz", allow_pickle=True)

    idx = info.index[info["SP"] == freq].to_numpy()
    rng = np.random.default_rng(0)
    picked = rng.choice(idx, size=min(n_series, len(idx)), replace=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | freq={freq} | n_series={len(picked)} | horizon={horizon}")
    print(f"inference: N={num_samples} temperature={temperature}")

    series_raw = [train[i].astype(np.float64) for i in picked]
    stats = [(float(s.mean()), float(s.std() or 1.0)) for s in series_raw]
    series_norm = [(s - mu) / sd for s, (mu, sd) in zip(series_raw, stats)]
    dfs = [pd.DataFrame({"x": s}) for s in series_norm]

    train_part = float(os.environ.get("M4_TRAIN_PART", "0.8"))
    min_series_len = min(len(s) for s in series_norm)
    min_val_tokens = int(min_series_len * (1 - train_part))
    target_bs = int(os.environ.get("M4_BLOCK_SIZE", "256"))
    block_size = min(target_bs, min_val_tokens - 1)
    block_size = max(block_size, horizon + 16)
    assert block_size > horizon

    all_tokens = np.concatenate(series_norm, axis=0).reshape(-1, 1)
    domain_of_definition = get_domain_of_definition(
        all_tokens, order_of_derivative=1, use_decimal=False
    )

    label_smoothing_sigma = float(os.environ.get("M4_LABEL_SMOOTHING_SIGMA", "2.0"))
    vocab_size = int(os.environ.get("M4_VOCAB_SIZE", "256"))
    base = GPT(
        vocab_size=vocab_size,
        n_embd=int(os.environ.get("M4_N_EMBD", "64")),
        block_size=block_size,
        n_head=int(os.environ.get("M4_N_HEAD", "4")),
        n_layer=int(os.environ.get("M4_N_LAYER", "2")),
        use_checkpoint=False,
        label_smoothing_sigma=label_smoothing_sigma,
    ).to(device=device)
    if device == "cuda":
        base.compile(mode="max-autotune-no-cudagraphs")
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=domain_of_definition,
        use_decimal=False,
    )

    max_iters = int(os.environ.get("M4_MAX_ITERS", "20000"))
    loader = DiffDataFrameDataLoader(
        dfs=dfs,
        block_size=block_size,
        batch_size=32,
        vocab_size=vocab_size,
        order_of_derivative=1,
        domain_of_definition=domain_of_definition,
        use_decimal=False,
        device=device,
        train_part=train_part,
    )
    val_loss, train_time_s = model.train(
        loader=loader,
        learning_rate=1e-2,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        max_iters=max_iters,
        eval_interval=int(os.environ.get("M4_EVAL_INTERVAL", "500")),
        eval_iters=40,
        use_tqdm=False,
        use_early_stopping=False,
    )
    print(f"train: val_loss={val_loss:.4f} train_time_s={train_time_s}")

    base.eval()
    top_p_env = os.environ.get("M4_TOP_P")
    if top_p_env is not None:
        top_p = float(top_p_env)
        assert 0.0 < top_p <= 1.0
        sampler = NucleusSampler(
            p=top_p, sampler=TemperatureSampler(temperature=temperature)
        )
        print(f"sampler: nucleus top-p={top_p} + temperature={temperature}")
    else:
        sampler = TemperatureSampler(temperature=temperature)
        print(f"sampler: temperature={temperature}")
    smapes, mases = [], []
    crpses, msises, covs, pinballs = [], [], [], []
    for i_global, series_idx in enumerate(picked):
        insample_raw = series_raw[i_global]
        mu, sd = stats[i_global]
        truth_raw = test[series_idx].astype(np.float64)
        ctx_len = min(len(dfs[i_global]), block_size - horizon)
        context = dfs[i_global].iloc[-ctx_len:]
        # Monte-Carlo trajectories, shape (N, horizon, 1).
        samples_norm = model.predict_samples(
            df=context,
            max_new_points=horizon,
            num_samples=num_samples,
            sampler=sampler,
        )[..., 0]  # drop the trailing column axis
        samples_raw = samples_norm * sd + mu  # (N, H)
        median_pred = np.median(samples_raw, axis=0)
        # Point metrics.
        smapes.append(smape(median_pred, truth_raw))
        mases.append(mase(median_pred, truth_raw, insample_raw, season))
        # Probabilistic metrics.
        crpses.append(crps_empirical(samples_raw, truth_raw))
        msises.append(msis(samples_raw, truth_raw, insample_raw, season, alpha=0.05))
        covs.append(empirical_coverage(samples_raw, truth_raw, level=0.8))
        pinballs.append(pinball_loss(median_pred, truth_raw, alpha=0.5))

    print("=" * 72)
    print(
        f"M4 {freq} probabilistic | n={len(picked)} | horizon={horizon} "
        f"| samples={num_samples}"
    )
    print(f"  sMAPE      (median point) = {np.mean(smapes):.3f}")
    print(f"  MASE       (median point) = {np.nanmean(mases):.3f}")
    print(f"  pinball@.5 (= 0.5·MAE)    = {np.mean(pinballs):.4f}")
    print(f"  CRPS                      = {np.mean(crpses):.4f}")
    print(f"  MSIS       (α=0.05)       = {np.nanmean(msises):.3f}")
    print(f"  coverage@80 (nominal 0.80) = {np.mean(covs):.3f}")


if __name__ == "__main__":
    main()
