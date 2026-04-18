"""
Long-term forecasting on ETT (multivariate) — iTransformer-style protocol.

Matches arXiv 2310.06625v4 evaluation:
  - All 7 variates (features = 'M')
  - seq_len = 96
  - pred_len ∈ {96, 192, 336, 720}
  - Chronological split 12 / 4 / 4 months for ETTh{1,2}
    (or 48 / 16 / 16 months-equivalent in 15-min for ETTm{1,2})
  - StandardScaler fit on the TRAIN split, applied to train/val/test
  - Metrics: MSE, MAE on z-score-normalised test data, averaged across all
    test windows and all 7 channels
  - Test windows: sliding with stride = pred_len so each point is covered
    exactly once (iTransformer uses stride=1, i.e. 2 689-ish windows per
    horizon; this script defaults to non-overlapping windows for speed but
    can be overridden via ETT_STRIDE env var)

Usage:
    uv run --no-sync python -m benchmarks.ett_bench
    ETT_DATASET=ETTh2 uv run --no-sync python -m benchmarks.ett_bench
    ETT_PRED_LENS=96,192 ETT_STRIDE=1 uv run --no-sync python -m benchmarks.ett_bench
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


ETT_DIR = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "long_term_forecast"
    / "ETT-small"
)

# iTransformer / TSLib split boundaries in ROWS per subset.
# ETTh is hourly, ETTm is 15-min. Convention (from Dataset_ETT_hour /
# Dataset_ETT_minute in TSLib): the first 12 months are train, next 4 val,
# next 4 test.
BOUNDARIES: dict[str, tuple[int, int, int]] = {
    # (train_end, val_end, test_end) in row indices
    "ETTh1": (12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24),
    "ETTh2": (12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24),
    "ETTm1": (12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4),
    "ETTm2": (12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4),
}

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def mse_mae(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    err = pred - true
    return float(np.mean(err**2)), float(np.mean(np.abs(err)))


def _evaluate(
    model: DiffGPT,
    test_df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    stride: int,
    sampler: TemperatureSampler,
) -> tuple[float, float, int]:
    """Slide a (seq_len + pred_len) window over test_df; report MSE / MAE."""
    total = len(test_df)
    max_start = total - seq_len - pred_len
    if max_start <= 0:
        raise RuntimeError(
            f"test split too small: need >={seq_len + pred_len}, got {total}"
        )
    starts = list(range(0, max_start + 1, stride))
    mses: list[float] = []
    maes: list[float] = []
    for s in starts:
        ctx = test_df.iloc[s : s + seq_len]
        truth = test_df.iloc[s + seq_len : s + seq_len + pred_len].to_numpy()
        pred_df = model.predict(df=ctx, max_new_points=pred_len, sampler=sampler)
        pred = pred_df.iloc[-pred_len:].to_numpy()
        ms, ma = mse_mae(pred, truth)
        mses.append(ms)
        maes.append(ma)
    return float(np.mean(mses)), float(np.mean(maes)), len(starts)


def run_horizon(
    dataset: str,
    pred_len: int,
    seq_len: int,
    stride: int,
    max_iters: int,
    n_embd: int,
    n_layer: int,
    n_head: int,
    batch_size: int,
    device: str,
    label_smoothing_sigma: float,
    use_checkpoint: bool,
) -> tuple[float, float, int, float]:
    """Train+evaluate one (dataset, pred_len) cell. Returns (mse, mae, windows, train_time_s)."""
    csv = ETT_DIR / f"{dataset}.csv"
    assert csv.exists(), f"missing {csv}"
    train_end, val_end, test_end = BOUNDARIES[dataset]
    df = pd.read_csv(csv)[FEATURES].iloc[:test_end].reset_index(drop=True)

    # z-score via train-split stats (iTransformer standard).
    train_raw = df.iloc[:train_end]
    mu = train_raw.mean().to_numpy()
    sd = train_raw.std().to_numpy()
    sd = np.where(sd > 0, sd, 1.0)
    norm = (df.to_numpy() - mu) / sd
    norm_df = pd.DataFrame(norm, columns=FEATURES)

    # Train + val portion for the loader; test held out for evaluation.
    train_plus_val = norm_df.iloc[:val_end]
    test_df = norm_df.iloc[val_end:test_end].reset_index(drop=True)
    train_part = train_end / val_end  # e.g. 12/16 = 0.75 for ETTh

    # Block size must fit (seq_len + pred_len - 1) * n_features tokens.
    n_features = len(FEATURES)
    block_size = (seq_len + pred_len - 1) * n_features
    # Round up to a small multiple for RoPE cache friendliness.
    block_size = ((block_size + 63) // 64) * 64

    domain = get_domain_of_definition(
        train_plus_val.iloc[:train_end].to_numpy(dtype=np.float64),
        order_of_derivative=1,
        use_decimal=False,
    )
    base = GPT(
        vocab_size=64,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        use_checkpoint=use_checkpoint,
        label_smoothing_sigma=label_smoothing_sigma,
    ).to(device=device)
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=domain,
        use_decimal=False,
    )
    loader = DiffDataFrameDataLoader(
        dfs=[train_plus_val],
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=64,
        order_of_derivative=1,
        domain_of_definition=domain,
        use_decimal=False,
        device=device,
        train_part=train_part,
    )
    val_loss, train_time_s = model.train(
        loader=loader,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        max_iters=max_iters,
        eval_interval=max(500, max_iters // 10),
        eval_iters=20,
        use_tqdm=False,
        use_early_stopping=False,
        save_best=True,
    )
    base.eval()
    sampler = TemperatureSampler(temperature=0.0)  # argmax → deterministic
    mse, mae, n_windows = _evaluate(
        model=model,
        test_df=test_df,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        sampler=sampler,
    )
    print(
        f"{dataset:5s} pred={pred_len:4d} seq={seq_len:3d} "
        f"MSE={mse:.4f} MAE={mae:.4f} "
        f"windows={n_windows} val_loss={val_loss:.3f} train_s={train_time_s}"
    )
    return mse, mae, n_windows, train_time_s


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    dataset = os.environ.get("ETT_DATASET", "ETTh1")
    seq_len = int(os.environ.get("ETT_SEQ_LEN", "96"))
    pred_lens = [int(x) for x in os.environ.get("ETT_PRED_LENS", "96,192,336,720").split(",")]
    stride_env = os.environ.get("ETT_STRIDE")
    max_iters = int(os.environ.get("ETT_MAX_ITERS", "2000"))
    n_embd = int(os.environ.get("ETT_N_EMBD", "128"))
    n_layer = int(os.environ.get("ETT_N_LAYER", "2"))
    n_head = int(os.environ.get("ETT_N_HEAD", "4"))
    label_smoothing_sigma = float(os.environ.get("ETT_LABEL_SMOOTHING_SIGMA", "0.0"))
    use_checkpoint = os.environ.get("ETT_USE_CHECKPOINT", "0") == "1"
    batch_size_override = os.environ.get("ETT_BATCH_SIZE")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} dataset={dataset} pred_lens={pred_lens} max_iters={max_iters}")

    # Summary table.
    rows = []
    for pred_len in pred_lens:
        # Default stride = pred_len (non-overlapping). Override with ETT_STRIDE=1
        # for the exact iTransformer stride-1 protocol (much slower).
        stride = int(stride_env) if stride_env else pred_len
        # Scale batch_size down for longer horizons so block_size * batch_size
        # stays within GPU memory.
        bs_per_horizon = {96: 32, 192: 16, 336: 8, 720: 4}
        batch_size = (
            int(batch_size_override)
            if batch_size_override
            else bs_per_horizon.get(pred_len, 4)
        )
        mse, mae, n, t = run_horizon(
            dataset=dataset,
            pred_len=pred_len,
            seq_len=seq_len,
            stride=stride,
            max_iters=max_iters,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            batch_size=batch_size,
            device=device,
            label_smoothing_sigma=label_smoothing_sigma,
            use_checkpoint=use_checkpoint,
        )
        rows.append((pred_len, mse, mae, n, t))

    print("=" * 60)
    print(f"{dataset} (multivariate, 7 channels) | seq_len={seq_len}")
    print(f"{'pred_len':>8}  {'MSE':>7}  {'MAE':>7}  {'windows':>8}  {'train_s':>8}")
    for pred_len, mse, mae, n, t in rows:
        print(f"{pred_len:>8}  {mse:7.4f}  {mae:7.4f}  {n:>8}  {t:>8}")
    mses = [r[1] for r in rows]
    maes = [r[2] for r in rows]
    print(f"{'avg':>8}  {np.mean(mses):7.4f}  {np.mean(maes):7.4f}")


if __name__ == "__main__":
    main()
