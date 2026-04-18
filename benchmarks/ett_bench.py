"""
Long-term forecasting on ETTh1 (univariate OT column).

Protocol mirrors TSLib's default long-term forecasting:
  - Split by row index: 12 months train / 4 months val / 4 months test
  - seq_len = 96, pred_len ∈ {96, 192, 336, 720}
  - Metrics: MSE and MAE on z-score-normalized values (train-set stats)

Usage (from repo root):
    uv run --no-sync python -m benchmarks.ett_bench
Override the default pred_len / rows-scanned via env:
    ETT_PRED_LEN=192 ETT_TEST_STEPS=200 uv run --no-sync python -m benchmarks.ett_bench

Deterministic forecast: TemperatureSampler(0) → argmax on every token.
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


ETT_CSV = Path(__file__).resolve().parents[1] / "all_datasets" / "ETT-small" / "ETTh1.csv"

# TSLib ETTh fixed boundaries (hourly, 30 days/month).
MONTH = 30 * 24
TRAIN_END = 12 * MONTH    # 8 640
VAL_END = 16 * MONTH      # 11 520
TEST_END = 20 * MONTH     # 14 400


def mse_mae(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    err = pred - true
    return float(np.mean(err**2)), float(np.mean(np.abs(err)))


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    pred_len = int(os.environ.get("ETT_PRED_LEN", "96"))
    seq_len = int(os.environ.get("ETT_SEQ_LEN", "96"))
    test_steps = int(os.environ.get("ETT_TEST_STEPS", "300"))  # # windows to evaluate

    # Block size must fit seq_len + pred_len (univariate, 1 feature).
    block_size = 256
    assert seq_len + pred_len - 1 <= block_size, (
        f"{seq_len}+{pred_len}-1 > block_size={block_size}"
    )

    # --- Load ---
    assert ETT_CSV.exists(), f"Missing {ETT_CSV}"
    df = pd.read_csv(ETT_CSV)
    ot = df[["OT"]].iloc[:TEST_END].reset_index(drop=True)

    train_raw = ot.iloc[:TRAIN_END]
    # z-score based on train stats (as TSLib does)
    mu = float(train_raw["OT"].mean())
    sd = float(train_raw["OT"].std())
    ot_norm = (ot["OT"] - mu) / sd
    train_df = ot_norm.iloc[:TRAIN_END].to_frame()
    test_df = ot_norm.iloc[VAL_END:TEST_END].to_frame()

    # --- Model ---
    domain_of_definition = get_domain_of_definition(
        train_df.to_numpy(dtype=np.float64),
        order_of_derivative=1,
        use_decimal=False,
    )
    base = GPT(
        vocab_size=256,
        n_embd=128,
        block_size=block_size,
        n_head=4,
        n_layer=4,
        use_checkpoint=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.to(device=device)
    # One model, trained + inferred many times — pay the compile cost once.
    if device == "cuda":
        base.compile(mode="max-autotune-no-cudagraphs")
    print(f"device={device}")
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=domain_of_definition,
        use_decimal=False,
    )

    # --- Train on the train split ---
    loader = DiffDataFrameDataLoader(
        dfs=[train_df],
        block_size=base.block_size,
        batch_size=32,
        vocab_size=256,
        order_of_derivative=1,
        domain_of_definition=domain_of_definition,
        use_decimal=False,
        device=device,
        train_part=0.9,
    )
    val_loss, train_time = model.train(
        loader=loader,
        learning_rate=1e-2,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        max_iters=5000,
        eval_interval=5000,
        eval_iters=50,
        use_tqdm=False,
    )
    print(f"train: val_loss={val_loss:.4f}  train_time_s={train_time}")

    # --- Forecast over the test split with a sliding window ---
    base.eval()
    sampler = TemperatureSampler(temperature=0.0)  # argmax = deterministic
    total_rows = len(test_df)
    # Valid windows start at 0..(total_rows - seq_len - pred_len).
    max_start = total_rows - seq_len - pred_len
    if max_start <= 0:
        raise RuntimeError("test split too small for requested seq_len/pred_len")
    # Uniformly subsample `test_steps` windows.
    starts = np.linspace(0, max_start, num=min(test_steps, max_start + 1), dtype=np.int64)

    mses, maes = [], []
    for s in starts:
        context = test_df.iloc[s : s + seq_len]
        truth = test_df.iloc[s + seq_len : s + seq_len + pred_len].to_numpy().reshape(-1)
        pred_df = model.predict(df=context, max_new_points=pred_len, sampler=sampler)
        # DiffGPT.predict returns seq_len + pred_len rows; the tail is the forecast.
        pred = pred_df.iloc[-pred_len:].to_numpy().reshape(-1)
        ms, ma = mse_mae(pred, truth)
        mses.append(ms)
        maes.append(ma)

    mse = float(np.mean(mses))
    mae = float(np.mean(maes))
    print(
        f"ETTh1 OT | seq={seq_len} pred={pred_len} | "
        f"MSE={mse:.4f} MAE={mae:.4f} | windows={len(starts)}"
    )


if __name__ == "__main__":
    main()
