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
    uv run --no-sync python -m diff_gpt.interfaces.ett_bench
    ETT_DATASET=ETTh2 uv run --no-sync python -m diff_gpt.interfaces.ett_bench
    ETT_PRED_LENS=96,192 ETT_STRIDE=1 \
        uv run --no-sync python -m diff_gpt.interfaces.ett_bench
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from diff_gpt.application.data_loader import DiffDataFrameDataLoader
from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.application.paths import ROOT
from diff_gpt.domain.encoder_decoder import DEFAULT_TOKENIZER, get_domain_of_definition
from diff_gpt.domain.metrics import bayes_point
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.sampler.median import MedianSampler
from diff_gpt.domain.sampler.temperature import TemperatureSampler

ETT_DIR = ROOT / "datasets" / "long_term_forecast" / "ETT-small"

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
    """Return (MSE, MAE) of `pred` against `true`, averaged over all elements."""
    err = pred - true
    return float(np.mean(err**2)), float(np.mean(np.abs(err)))


def _evaluate(
    model: DiffGPT,
    test_df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    stride: int,
    readouts: list[str],
    num_samples: int,
) -> tuple[dict[str, tuple[float, float]], int]:
    """
    Slide a (seq_len + pred_len) window over test_df; MSE / MAE per readout.

    Readouts: "argmax" and "median" feed one token back per step; "mc" draws
    `num_samples` trajectories once and reports the Bayes points for MSE
    ("mc-mean") and MAE ("mc-median").
    """
    total = len(test_df)
    max_start = total - seq_len - pred_len
    if max_start <= 0:
        msg = f"test split too small: need >={seq_len + pred_len}, got {total}"
        raise RuntimeError(msg)
    starts = list(range(0, max_start + 1, stride))
    acc: dict[str, list[tuple[float, float]]] = {}
    for s in starts:
        ctx = test_df.iloc[s : s + seq_len]
        truth = test_df.iloc[s + seq_len : s + seq_len + pred_len].to_numpy()
        preds: dict[str, np.ndarray] = {}
        for r in readouts:
            if r == "argmax":
                preds[r] = (
                    model.predict(
                        df=ctx,
                        max_new_points=pred_len,
                        sampler=TemperatureSampler(0.0),
                    )
                    .iloc[-pred_len:]
                    .to_numpy()
                )
            elif r == "median":
                preds[r] = (
                    model.predict(
                        df=ctx,
                        max_new_points=pred_len,
                        sampler=MedianSampler(),
                    )
                    .iloc[-pred_len:]
                    .to_numpy()
                )
            elif r == "mc":
                samp = model.predict_samples(
                    df=ctx,
                    max_new_points=pred_len,
                    num_samples=num_samples,
                    sampler=TemperatureSampler(1.0),
                )
                preds["mc-mean"] = bayes_point(samp, "mse")
                preds["mc-median"] = bayes_point(samp, "mae")
            else:
                msg = f"unknown readout {r!r}"
                raise ValueError(msg)
        for name, pred in preds.items():
            acc.setdefault(name, []).append(mse_mae(pred, truth))
    out = {
        name: (float(np.mean([m for m, _ in v])), float(np.mean([a for _, a in v])))
        for name, v in acc.items()
    }
    return out, len(starts)


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
    *,
    use_checkpoint: bool,
    compile_model: bool,
    seed: int = 0,
    vocab_size: int = 64,
    readouts: list[str] | None = None,
    num_samples: int = 50,
    save_path: str | None = None,
    align_windows: bool = False,
) -> tuple[dict[str, tuple[float, float]], int, float]:
    """
    Train and evaluate one (dataset, pred_len) cell.

    Returns ({readout: (mse, mae)}, windows, train_time_s).
    """
    readouts = readouts or ["argmax"]
    torch.manual_seed(seed)
    csv = ETT_DIR / f"{dataset}.csv"
    if not csv.exists():
        msg = f"missing {csv}"
        raise FileNotFoundError(msg)
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
    # Round up to a multiple of 64.
    block_size = ((block_size + 63) // 64) * 64

    domain = get_domain_of_definition(
        train_plus_val.iloc[:train_end].to_numpy(dtype=np.float64),
        order_of_derivative=1,
    )
    base = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        use_checkpoint=use_checkpoint,
    ).to(device=device)
    if device == "cuda" and compile_model:
        base.compile(mode="max-autotune")
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=domain,
    )
    loader = DiffDataFrameDataLoader(
        dfs=[train_plus_val],
        block_size=block_size,
        batch_size=batch_size,
        vocab_size=vocab_size,
        order_of_derivative=1,
        domain_of_definition=domain,
        device=device,
        train_part=train_part,
        rng=torch.Generator().manual_seed(42 + seed),
        align_windows=align_windows,
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
    if save_path:
        torch.save(base.state_dict(), save_path)
    results, n_windows = _evaluate(
        model=model,
        test_df=test_df,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        readouts=readouts,
        num_samples=num_samples,
    )
    for name, (mse, mae) in results.items():
        print(
            f"{dataset:5s} pred={pred_len:4d} seq={seq_len:3d} seed={seed} "
            f"readout={name:9s} MSE={mse:.4f} MAE={mae:.4f} "
            f"windows={n_windows} val_loss={val_loss:.3f} train_s={train_time_s}",
        )
    return results, n_windows, train_time_s


@dataclass(frozen=True)
class Settings:
    """The settings of bench, read from the ETT_* environment variables."""

    compile_model: bool
    dataset: str
    seq_len: int
    pred_lens: list[int]
    stride_env: str | None
    max_iters: int
    n_embd: int
    n_layer: int
    n_head: int
    use_checkpoint: bool
    batch_size_override: str | None
    seed: int
    vocab_size: int
    readouts: list[str]
    num_samples: int
    save_prefix: str | None
    align_windows: bool


def bench(settings: Settings) -> None:
    """Train and score one model per horizon, then print the summary table."""
    torch.manual_seed(0)

    dataset = settings.dataset
    seq_len = settings.seq_len
    pred_lens = settings.pred_lens
    stride_env = settings.stride_env
    max_iters = settings.max_iters
    n_embd = settings.n_embd
    n_layer = settings.n_layer
    n_head = settings.n_head
    use_checkpoint = settings.use_checkpoint
    batch_size_override = settings.batch_size_override
    seed = settings.seed
    vocab_size = settings.vocab_size
    readouts = settings.readouts
    num_samples = settings.num_samples
    save_prefix = (
        settings.save_prefix
    )  # e.g. runs/ett_step6 -> runs/ett_step6_p96_seed0.pt
    align_windows = settings.align_windows  # step 7

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} dataset={dataset} pred_lens={pred_lens} "
        f"max_iters={max_iters}",
    )
    print(
        f"tokenizer={DEFAULT_TOKENIZER} | V={vocab_size} | seed={seed} "
        f"| readouts={readouts} | align_windows={align_windows}",
    )

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
        results, n, t = run_horizon(
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
            use_checkpoint=use_checkpoint,
            compile_model=settings.compile_model,
            seed=seed,
            vocab_size=vocab_size,
            readouts=readouts,
            num_samples=num_samples,
            save_path=f"{save_prefix}_p{pred_len}_seed{seed}.pt"
            if save_prefix
            else None,
            align_windows=align_windows,
        )
        for name, (mse, mae) in results.items():
            rows.append((pred_len, name, mse, mae, n, t))

    print("=" * 60)
    print(f"{dataset} (multivariate, 7 channels) | seq_len={seq_len} | seed={seed}")
    print(
        f"{'pred_len':>8}  {'readout':>9}  {'MSE':>7}  {'MAE':>7}  "
        f"{'windows':>8}  {'train_s':>8}",
    )
    for pred_len, name, mse, mae, n, t in rows:
        print(f"{pred_len:>8}  {name:>9}  {mse:7.4f}  {mae:7.4f}  {n:>8}  {t:>8}")
