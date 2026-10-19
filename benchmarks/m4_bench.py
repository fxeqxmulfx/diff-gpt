r"""
Short-term forecasting on M4 with ONE global diff-gpt per frequency group.

For each frequency (Hourly, Daily, etc.):
  1. z-score normalize each series using its own training mean/std
  2. Train a single diff-gpt on the collection of normalized training series
     (a DiffDataFrameDataLoader samples windows across all series)
  3. For each held-out series, condition on its training tail and forecast
     the known horizon; un-normalize before scoring
  4. Report sMAPE, MASE averaged over series

This is the "global" regime used by the M4 leaderboard (N-BEATS, TimesNet,
Smyl). Per-series training is handled by the old m4_bench; this variant is
the competitive setup.

Usage:
    uv run --no-sync python -m benchmarks.m4_bench
    M4_FREQ=Daily  M4_N_SERIES=200 uv run --no-sync python -m benchmarks.m4_bench
    M4_SEED=1 M4_SAMPLERS=argmax,median M4_SAVE=runs/m4.pt \
        uv run --no-sync python -m benchmarks.m4_bench
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from diff_gpt.data_loader import DiffDataFrameDataLoader
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import (
    DEFAULT_TOKENIZER,
    TokenClippingWarning,
    get_domain_of_definition,
)
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.median import MedianSampler
from diff_gpt.sampler.temperature import TemperatureSampler
from diff_gpt.train import train as train_loop

M4_DIR = Path(__file__).resolve().parents[1] / "datasets" / "short_term_forecast" / "m4"

HORIZONS = {
    "Yearly": 6,
    "Quarterly": 8,
    "Monthly": 18,
    "Weekly": 13,
    "Daily": 14,
    "Hourly": 48,
}
# MASE seasonality per M4's own definition (Makridakis et al. 2020): the daily
# series have a weekly period of 7, but M4 scores Daily with seasonality 1, and
# using 7 understates MASE by ~2.7x on this data.
SEASONALITY = {
    "Yearly": 1,
    "Quarterly": 4,
    "Monthly": 12,
    "Weekly": 1,
    "Daily": 1,
    "Hourly": 24,
}


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Return the M4 sMAPE (in percent, range 0-200) of `pred` against `true`.

    Points where both values are 0 are skipped; 0.0 if no point is left.
    """
    num = np.abs(true - pred)
    den = np.abs(true) + np.abs(pred)
    mask = den > 0
    return float(200.0 * np.mean(num[mask] / den[mask])) if mask.any() else 0.0


def mase(
    pred: np.ndarray,
    true: np.ndarray,
    insample: np.ndarray,
    season: int,
) -> float:
    """
    Return the MASE of `pred` against `true`.

    The scale is the in-sample MAE of the seasonal naive forecast with lag
    `season`; NaN when `insample` is not longer than `season` or that MAE is 0.
    """
    if len(insample) <= season:
        return float("nan")
    naive = np.mean(np.abs(insample[season:] - insample[:-season]))
    if naive == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / naive)


def prepare() -> dict[str, Any]:
    """
    Load the picked M4 series and preprocess them for training and scoring.

    Reads M4_FREQ, M4_N_SERIES, M4_MIN_LEN, M4_TRAIN_PART, M4_BLOCK_SIZE,
    M4_ORDER and M4_DOMAIN_QUANTILE from the environment. Returns the picked
    series indices, their raw values, per-series (mean, std), z-scored
    DataFrames, the test split, horizon, seasonality, train_part,
    min_val_tokens, block_size, derivative order, domain of definition and
    its quantile (None for the max). m4_eval shares it, so that a checkpoint
    is scored with the block size and domain it was trained with.
    """
    freq = os.environ.get("M4_FREQ", "Hourly")
    n_series = int(os.environ.get("M4_N_SERIES", "50"))
    info = pd.read_csv(M4_DIR / "M4-info.csv")
    train = np.load(M4_DIR / "training.npz", allow_pickle=True)
    test = np.load(M4_DIR / "test.npz", allow_pickle=True)
    idx = info.index[info["SP"] == freq].to_numpy()
    rng = np.random.default_rng(0)
    picked = rng.choice(idx, size=min(n_series, len(idx)), replace=False)

    # Drop very short series before sizing the context: block_size is capped by
    # the SHORTEST series' val portion, so one 93-point series would shrink the
    # context for all 4227 of them (M4_MIN_LEN=0 keeps every series).
    min_len = int(os.environ.get("M4_MIN_LEN", "0"))
    if min_len > 0:
        keep = [i for i in picked if len(train[i]) >= min_len]
        print(f"M4_MIN_LEN={min_len}: kept {len(keep)} of {len(picked)} series")
        picked = np.array(keep)
    # Per-series normalization stats computed on the training history.
    # We store them so we can invert before scoring.
    series_raw = [train[i].astype(np.float64) for i in picked]
    stats = [(float(s.mean()), float(s.std() or 1.0)) for s in series_raw]
    series_norm = [(s - mu) / sd for s, (mu, sd) in zip(series_raw, stats, strict=True)]
    dfs = [pd.DataFrame({"x": s}) for s in series_norm]

    # Block size: every series's VAL portion must be larger than block_size
    # for val-loss estimation. With train_part=0.8 on Hourly's shortest
    # series (700 tokens), val = 140 → block_size caps at 139.
    horizon = HORIZONS[freq]
    train_part = float(os.environ.get("M4_TRAIN_PART", "0.8"))
    min_val_tokens = int(min(len(s) for s in series_norm) * (1 - train_part))
    # Target block_size (overridable via env); clamp down if val is too short.
    target_bs = int(os.environ.get("M4_BLOCK_SIZE", "256"))
    block_size = max(min(target_bs, min_val_tokens - 1), horizon + 16)

    order = int(os.environ.get("M4_ORDER", "1"))
    # Global domain_of_definition over the union of normalized training data.
    dq = os.environ.get("M4_DOMAIN_QUANTILE")
    quantile = float(dq) if dq else None
    domain = get_domain_of_definition(
        np.concatenate(series_norm, axis=0).reshape(-1, 1),
        order_of_derivative=order,
        use_decimal=False,
        quantile=quantile,
    )
    return {
        "freq": freq,
        "picked": picked,
        "series_raw": series_raw,
        "stats": stats,
        "dfs": dfs,
        "test": test,
        "horizon": horizon,
        "season": SEASONALITY[freq],
        "train_part": train_part,
        "min_val_tokens": min_val_tokens,
        "block_size": block_size,
        "order": order,
        "domain": domain,
        "quantile": quantile,
    }


def main() -> None:
    """
    Train one global diff-gpt on the picked M4 series and print its scores.

    Settings come from the M4_* environment variables (see the module
    docstring); sMAPE and MASE are printed for each readout in M4_SAMPLERS.
    """
    torch.set_float32_matmul_precision("high")
    seed = int(os.environ.get("M4_SEED", "0"))
    torch.manual_seed(seed)

    data = prepare()
    freq = data["freq"]
    picked = data["picked"]
    series_raw = data["series_raw"]
    stats = data["stats"]
    dfs = data["dfs"]
    test = data["test"]
    horizon = data["horizon"]
    season = data["season"]
    train_part = data["train_part"]
    block_size = data["block_size"]
    order_of_derivative = data["order"]
    domain_of_definition = data["domain"]
    if data["quantile"] is not None:
        # A quantile domain clips the steps beyond it on purpose.
        warnings.filterwarnings("ignore", category=TokenClippingWarning)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | freq={freq} | n_series={len(picked)} | horizon={horizon}")
    train_module = (
        None  # set when a CUDA-graph compile mode wraps the model for training
    )
    print(
        f"block_size={block_size} | min_val_tokens={data['min_val_tokens']} "
        f"| train_part={train_part}",
    )
    print(
        f"domain={domain_of_definition.round(4)} "
        f"(quantile={data['quantile'] or 'max'})",
    )

    # Small model: ~130k params. At ~280k training tokens (414 series * ~850 *
    # train_part=0.8), this gives ~2 tokens/param — a compromise between
    # Chinchilla-optimal (~20) and capacity needed for Hourly's daily +
    # weekly patterns. Tiny (n_embd=32) underfits; big (n_embd=128) overfits.
    vocab_size = int(os.environ.get("M4_VOCAB_SIZE", "256"))
    n_embd = int(os.environ.get("M4_N_EMBD", "64"))
    n_layer = int(os.environ.get("M4_N_LAYER", "2"))
    n_head = int(os.environ.get("M4_N_HEAD", "4"))
    print(f"tokenizer={DEFAULT_TOKENIZER} | V={vocab_size}")
    base = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        use_checkpoint=False,
    ).to(device=device)
    if device == "cuda":
        # M4_COMPILE_MODE: "max-autotune-no-cudagraphs" (default), "max-autotune"
        # (adds CUDA graphs; training shapes are static), "reduce-overhead", "none".
        compile_mode = os.environ.get("M4_COMPILE_MODE", "max-autotune-no-cudagraphs")
        if compile_mode in ("reduce-overhead", "max-autotune"):
            # CUDA graphs need static shapes: compile a separate wrapper for the
            # training loop only; inference (KV cache, growing T) stays on `base`.
            train_module = torch.compile(base, mode=compile_mode)
        elif compile_mode != "none":
            base.compile(mode=compile_mode)
    model = DiffGPT(
        model=base,
        order_of_derivative=order_of_derivative,
        domain_of_definition=domain_of_definition,
        use_decimal=False,
    )

    # Global training: one model over all series.
    max_iters = int(os.environ.get("M4_MAX_ITERS", "20000"))
    eval_interval = int(os.environ.get("M4_EVAL_INTERVAL", "500"))
    # Data-scaling knob: train on a prefix of the series, always score all of them,
    # so the test set is identical across training-set sizes.
    n_train_series = int(os.environ.get("M4_TRAIN_SERIES", "0")) or len(dfs)
    if n_train_series < len(dfs):
        print(
            f"M4_TRAIN_SERIES={n_train_series}: "
            f"training on {n_train_series} of {len(dfs)} series",
        )
    loader = DiffDataFrameDataLoader(
        dfs=dfs[:n_train_series],
        block_size=block_size,
        batch_size=32,
        vocab_size=vocab_size,
        order_of_derivative=order_of_derivative,
        domain_of_definition=domain_of_definition,
        use_decimal=False,
        device=device,
        train_part=train_part,
        rng=torch.Generator().manual_seed(42 + seed),
    )
    train_kwargs = {
        "loader": loader,
        "learning_rate": 1e-2,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "eval_iters": 40,
        "use_tqdm": False,
        "use_early_stopping": False,
    }
    if train_module is not None:
        val_loss, train_time_s = train_loop(mut_model=train_module, **train_kwargs)
    else:
        val_loss, train_time_s = model.train(**train_kwargs)
    print(f"train: val_loss={val_loss:.4f}  train_time_s={train_time_s}")

    save_path = os.environ.get("M4_SAVE")
    if save_path:
        torch.save(base.state_dict(), save_path)

    # Forecast each series by conditioning on its tail, once per requested
    # point readout (M4_SAMPLERS="argmax" | "median" | "argmax,median").
    base.eval()
    sampler_names = os.environ.get("M4_SAMPLERS", "argmax").split(",")
    samplers = {
        "argmax": TemperatureSampler(temperature=0.0),
        "median": MedianSampler(),
    }
    for name in sampler_names:
        sampler = samplers[name]
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
            f"seed={seed} | readout={name} | "
            f"sMAPE={np.mean(smapes):.2f} MASE={np.nanmean(mases):.3f}",
        )


if __name__ == "__main__":
    main()
