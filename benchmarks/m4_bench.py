"""
Short-term forecasting on M4 with ONE global diff-gpt per frequency group.

For each frequency (Hourly, Daily, etc.):
  1. z-score normalize each series using its own training mean/std
  2. Train a single diff-gpt on the collection of normalized training series
     via DiffGPT.train_multi (windows sampled across all series)
  3. For each held-out series, condition on its training tail and forecast
     the known horizon; un-normalize before scoring
  4. Report sMAPE, MASE averaged over series

This is the "global" regime used by the M4 leaderboard (N-BEATS, TimesNet,
Smyl). Per-series training is handled by the old m4_bench; this variant is
the competitive setup.

Usage:
    uv run --no-sync python -m benchmarks.m4_bench
    M4_FREQ=Daily  M4_N_SERIES=200 uv run --no-sync python -m benchmarks.m4_bench
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
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.temperature import TemperatureSampler


M4_DIR = Path(__file__).resolve().parents[1] / "all_datasets" / "m4"

HORIZONS = {
    "Yearly": 6,
    "Quarterly": 8,
    "Monthly": 18,
    "Weekly": 13,
    "Daily": 14,
    "Hourly": 48,
}
SEASONALITY = {
    "Yearly": 1, "Quarterly": 4, "Monthly": 12,
    "Weekly": 1, "Daily": 7, "Hourly": 24,
}


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    num = np.abs(true - pred)
    den = np.abs(true) + np.abs(pred)
    mask = den > 0
    return float(200.0 * np.mean(num[mask] / den[mask])) if mask.any() else 0.0


def mase(pred: np.ndarray, true: np.ndarray, insample: np.ndarray, season: int) -> float:
    if len(insample) <= season:
        return float("nan")
    naive = np.mean(np.abs(insample[season:] - insample[:-season]))
    if naive == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / naive)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    freq = os.environ.get("M4_FREQ", "Hourly")
    n_series = int(os.environ.get("M4_N_SERIES", "50"))
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

    # Per-series normalization stats computed on the training history.
    # We store them so we can invert before scoring.
    series_raw = [train[i].astype(np.float64) for i in picked]
    stats = [(float(s.mean()), float(s.std() or 1.0)) for s in series_raw]
    series_norm = [(s - mu) / sd for s, (mu, sd) in zip(series_raw, stats)]
    dfs = [pd.DataFrame({"x": s}) for s in series_norm]

    # Block size: every series's VAL portion must be larger than block_size
    # for val-loss estimation. With train_part=0.8 on Hourly's shortest
    # series (700 tokens), val = 140 → block_size caps at 139.
    train_part = 0.8
    min_series_len = min(len(s) for s in series_norm)
    min_val_tokens = int(min_series_len * (1 - train_part))
    # Target block_size (overridable via env); clamp down if val is too short.
    target_bs = int(os.environ.get("M4_BLOCK_SIZE", "256"))
    block_size = min(target_bs, min_val_tokens - 1)
    block_size = max(block_size, horizon + 16)
    assert block_size > horizon, (
        f"block_size {block_size} <= horizon {horizon}; min series too short"
    )
    print(
        f"block_size={block_size} | min_val_tokens={min_val_tokens} "
        f"| train_part={train_part}"
    )

    # Global domain_of_definition over the union of normalized training data.
    all_tokens = np.concatenate(series_norm, axis=0).reshape(-1, 1)
    domain_of_definition = get_domain_of_definition(
        all_tokens, order_of_derivative=1, use_decimal=False
    )

    # Small model: ~130k params. At ~280k training tokens (414 series * ~850 *
    # train_part=0.8), this gives ~2 tokens/param — a compromise between
    # Chinchilla-optimal (~20) and capacity needed for Hourly's daily +
    # weekly patterns. Tiny (n_embd=32) underfits; big (n_embd=128) overfits.
    base = GPT(
        vocab_size=256,
        n_embd=64,
        block_size=block_size,
        n_head=4,
        n_layer=2,
        use_checkpoint=False,
    ).to(device=device)
    if device == "cuda":
        base.compile(mode="max-autotune-no-cudagraphs")
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=domain_of_definition,
        use_decimal=False,
    )

    # Global training: one model over all series.
    max_iters = int(os.environ.get("M4_MAX_ITERS", "20000"))
    eval_interval = int(os.environ.get("M4_EVAL_INTERVAL", "500"))
    loader = DiffDataFrameDataLoader(
        dfs=dfs,
        block_size=block_size,
        batch_size=32,
        vocab_size=256,
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
        eval_interval=eval_interval,
        eval_iters=40,
        use_tqdm=False,
        use_early_stopping=False,
    )
    print(f"train: val_loss={val_loss:.4f}  train_time_s={train_time_s}")

    # Forecast each series by conditioning on its tail.
    base.eval()
    sampler = TemperatureSampler(temperature=0.0)
    smapes, mases = [], []
    for i_global, series_idx in enumerate(picked):
        insample_raw = series_raw[i_global]
        mu, sd = stats[i_global]
        truth_raw = test[series_idx].astype(np.float64)
        # Use the tail of the normalized training series as context.
        ctx_len = min(len(dfs[i_global]), block_size - horizon)
        context = dfs[i_global].iloc[-ctx_len:]
        pred_df = model.predict(df=context, max_new_points=horizon, sampler=sampler)
        pred_norm = pred_df.iloc[-horizon:].to_numpy().reshape(-1)
        pred_raw = pred_norm * sd + mu
        smapes.append(smape(pred_raw, truth_raw))
        mases.append(mase(pred_raw, truth_raw, insample_raw, season))

    print("=" * 60)
    print(
        f"M4 {freq} (global) | n={len(picked)} series | horizon={horizon} | "
        f"sMAPE={np.mean(smapes):.2f} MASE={np.nanmean(mases):.3f}"
    )


if __name__ == "__main__":
    main()
