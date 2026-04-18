"""
Anomaly detection demo on ETTh1 oil-temperature.

Demonstrates the `DiffGPT.anomaly_scores(df)` API on a real hourly signal.
Procedure:
  1. Load ETTh1 OT column (hourly electricity-transformer oil temperature).
  2. Take a clean training window, train a small diff-gpt on it.
  3. Inject a handful of synthetic spikes at known positions in a
     held-out test window.
  4. Score the test window; report the top-K most anomalous timestamps.
  5. Compare against the seeded anomaly positions to verify detection.

Usage:
    uv run --no-sync python -m benchmarks.anomaly_demo
    ANOMALY_N_INJECT=5 ANOMALY_MAX_ITERS=3000 uv run --no-sync python -m benchmarks.anomaly_demo
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


ETT_CSV = (
    Path(__file__).resolve().parents[1]
    / "datasets" / "long_term_forecast" / "ETT-small" / "ETTh1.csv"
)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    n_inject = int(os.environ.get("ANOMALY_N_INJECT", "5"))
    max_iters = int(os.environ.get("ANOMALY_MAX_ITERS", "2000"))
    top_k = int(os.environ.get("ANOMALY_TOP_K", "10"))

    # Load ETTh1 and take OT channel only. Keep first 4 months for train,
    # next month for scoring. Hourly → 24*30 ≈ 720 rows per month.
    df = pd.read_csv(ETT_CSV)
    ot = df[["OT"]].copy()
    train_end = 24 * 30 * 4        # 2880 rows
    test_end = train_end + 24 * 14  # 336 rows for the test window
    train_df = ot.iloc[:train_end].reset_index(drop=True)
    test_df_clean = ot.iloc[train_end:test_end].reset_index(drop=True)

    # Per-series z-score using train stats (standard RevIN-style).
    mu = float(train_df["OT"].mean())
    sd = float(train_df["OT"].std() or 1.0)
    train_norm = pd.DataFrame({"OT": (train_df["OT"].to_numpy() - mu) / sd})
    test_norm = pd.DataFrame({"OT": (test_df_clean["OT"].to_numpy() - mu) / sd})

    # Inject synthetic spikes into the test window at known positions.
    test_len = len(test_norm)
    # Space injections across the middle of the window.
    injection_idx = rng.choice(
        np.arange(50, test_len - 50),
        size=n_inject,
        replace=False,
    )
    injection_idx = np.sort(injection_idx)
    spike_magnitude = 5.0  # in normalized units, ~5σ of the training signal
    test_spiked = test_norm.copy()
    sign = rng.choice([-1.0, 1.0], size=n_inject)
    for i, idx in enumerate(injection_idx):
        test_spiked.iloc[int(idx), 0] += sign[i] * spike_magnitude

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"train rows: {len(train_norm)}, test rows: {test_len}")
    print(f"injected {n_inject} spikes at test-local indices: "
          f"{injection_idx.tolist()}")

    # Domain from training data only (treat injected spikes as unknown).
    domain = get_domain_of_definition(
        train_norm.to_numpy(dtype=np.float64),
        order_of_derivative=1,
        use_decimal=False,
    )
    base = GPT(
        vocab_size=256,
        n_embd=64,
        block_size=256,
        n_head=4,
        n_layer=2,
        use_checkpoint=False,
        label_smoothing_sigma=1.0,
    ).to(device=device)
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=domain,
        use_decimal=False,
    )
    loader = DiffDataFrameDataLoader(
        dfs=[train_norm],
        block_size=base.block_size,
        batch_size=16,
        vocab_size=base.vocab_size,
        order_of_derivative=1,
        domain_of_definition=domain,
        use_decimal=False,
        device=device,
        train_part=0.9,
    )
    val_loss, train_time_s = model.train(
        loader=loader,
        max_iters=max_iters,
        eval_interval=max(500, max_iters // 5),
        eval_iters=20,
        use_tqdm=False,
        use_early_stopping=False,
    )
    print(f"trained: val_loss={val_loss:.4f} time_s={train_time_s}")

    # Score the spiked test window.
    # The test window can be longer than block_size; chunk into sub-windows.
    B = base.block_size
    # Score a sub-window that fits in one forward pass.
    scored_window_len = B // 1 - 1  # 1 feature
    window = test_spiked.iloc[:scored_window_len].copy()
    scores = model.anomaly_scores(window)
    score_vec = scores["OT"].to_numpy()

    # Report top-K anomalous positions (ignore NaN prefix).
    valid = ~np.isnan(score_vec)
    # argpartition to find top-K efficiently.
    idx_valid = np.where(valid)[0]
    scores_valid = score_vec[idx_valid]
    k = min(top_k, len(idx_valid))
    top_local = idx_valid[np.argsort(scores_valid)[-k:][::-1]]

    print("\n" + "=" * 60)
    print(f"Top-{k} anomalies by -log p(token|ctx):")
    print(f"{'rank':>4} {'idx':>5} {'score':>8} {'value':>8} {'injected?':>12}")
    injected_set = set(injection_idx.tolist())
    caught = 0
    for rank, pos in enumerate(top_local, 1):
        is_injected = pos in injected_set or (pos + 1) in injected_set
        if is_injected:
            caught += 1
        val = float(test_spiked.iloc[int(pos), 0])
        s = float(score_vec[pos])
        marker = "✓" if is_injected else ""
        print(f"{rank:>4} {pos:>5} {s:>8.3f} {val:>8.3f} {marker:>12}")

    recall = caught / n_inject if n_inject > 0 else 0.0
    print(f"\nRecall of injected spikes in top-{k}: {caught}/{n_inject} = {recall:.0%}")
    # A passing demo: at least half the injected spikes should be in top-K.
    if recall >= 0.5:
        print("✓ model detects injected anomalies above noise floor.")
    else:
        print("(!) detection weak — try ANOMALY_MAX_ITERS > 3000 or larger model.")


if __name__ == "__main__":
    main()
