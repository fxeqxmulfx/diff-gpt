"""
Per-series diagnostics for saved M4 Hourly checkpoints (see m4_bench.py).

Usage:
    uv run --no-sync python -m benchmarks.m4_eval runs/a.pt:softmax runs/b.pt:mixture
Each argument is <state_dict path>:<head>. Data preparation mirrors m4_bench
exactly (same normalisation, block_size, domain), the tokenizer comes from
DIFF_GPT_TOKENIZER. Prints paired per-series comparisons between the first
checkpoint and every other one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmarks.m4_bench import HORIZONS, M4_DIR, SEASONALITY, mase, smape
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import DEFAULT_TOKENIZER, get_domain_of_definition
from diff_gpt.metrics import bayes_point, crps_empirical, empirical_coverage, msis
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.median import MedianSampler
from diff_gpt.sampler.temperature import TemperatureSampler

# M4 Hourly Naive2 reference values (Makridakis et al. 2020), used for OWA.
SMAPE_REF, MASE_REF = 18.383, 2.395


def prepare():
    freq = os.environ.get("M4_FREQ", "Hourly")
    n_series = int(os.environ.get("M4_N_SERIES", "414"))
    info = pd.read_csv(M4_DIR / "M4-info.csv")
    train = np.load(M4_DIR / "training.npz", allow_pickle=True)
    test = np.load(M4_DIR / "test.npz", allow_pickle=True)
    idx = info.index[info["SP"] == freq].to_numpy()
    picked = np.random.default_rng(0).choice(idx, size=min(n_series, len(idx)), replace=False)
    series_raw = [train[i].astype(np.float64) for i in picked]
    stats = [(float(s.mean()), float(s.std() or 1.0)) for s in series_raw]
    series_norm = [(s - mu) / sd for s, (mu, sd) in zip(series_raw, stats)]
    dfs = [pd.DataFrame({"x": s}) for s in series_norm]
    train_part = float(os.environ.get("M4_TRAIN_PART", "0.8"))
    min_val_tokens = int(min(len(s) for s in series_norm) * (1 - train_part))
    horizon = HORIZONS[freq]
    block_size = max(min(int(os.environ.get("M4_BLOCK_SIZE", "256")), min_val_tokens - 1), horizon + 16)
    order = int(os.environ.get("M4_ORDER", "1"))
    domain = get_domain_of_definition(
        np.concatenate(series_norm, axis=0).reshape(-1, 1), order_of_derivative=order, use_decimal=False
    )
    return dict(freq=freq, picked=picked, series_raw=series_raw, stats=stats, dfs=dfs, test=test,
                horizon=horizon, season=SEASONALITY[freq], block_size=block_size, order=order, domain=domain)


def load_model(path: str, head: str, data: dict, device: str) -> DiffGPT:
    base = GPT(
        vocab_size=int(os.environ.get("M4_VOCAB_SIZE", "256")),
        n_embd=int(os.environ.get("M4_N_EMBD", "64")),
        block_size=data["block_size"],
        n_head=int(os.environ.get("M4_N_HEAD", "4")),
        n_layer=int(os.environ.get("M4_N_LAYER", "2")),
        use_checkpoint=False,
        head=head,
        mixture_components=int(os.environ.get("M4_MIX_K", "8")),
        feature_lags=tuple(int(x) for x in os.environ.get("M4_FEATURE_LAGS", "").split(",") if x.strip()),
    )
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    if "feature_proj.weight" in sd:
        want = sd["feature_proj.weight"].shape[1]
        assert base.n_extra_features == want, (
            f"{path} was trained with {want} token features but M4_FEATURE_LAGS gives "
            f"{base.n_extra_features}; the lag values are not stored in the checkpoint, "
            f"set M4_FEATURE_LAGS (e.g. 24) to match the training run"
        )
    base.load_state_dict(sd)
    base.to(device).eval()
    return DiffGPT(model=base, order_of_derivative=data["order"],
                   domain_of_definition=data["domain"], use_decimal=False)


@torch.inference_mode()
def forecast(model: DiffGPT, data: dict, sampler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns per-series sMAPE, MASE and signed relative bias mean((pred-true)/|true|)."""
    sm, ms, bias = [], [], []
    for i, series_idx in enumerate(data["picked"]):
        mu, sd = data["stats"][i]
        truth = data["test"][series_idx].astype(np.float64)
        ctx_len = min(len(data["dfs"][i]), data["block_size"] - data["horizon"])
        ctx = data["dfs"][i].iloc[-ctx_len:]
        pred = model.predict(df=ctx, max_new_points=data["horizon"], sampler=sampler)
        pred = pred.iloc[-data["horizon"]:].to_numpy().reshape(-1) * sd + mu
        sm.append(smape(pred, truth))
        ms.append(mase(pred, truth, data["series_raw"][i], data["season"]))
        bias.append(float(np.mean((pred - truth) / np.maximum(np.abs(truth), 1e-9))))
    return np.array(sm), np.array(ms), np.array(bias)


@torch.inference_mode()
def readouts_and_scores(model: DiffGPT, data: dict, num_samples: int) -> dict[str, np.ndarray]:
    """Bayes readouts from Monte-Carlo trajectories plus proper scores, per series.

    Returns per-series arrays: sMAPE/MASE for the median path (MedianSampler
    fed back), for the MC marginal median, the MC sMAPE-Bayes point and the MC
    OWA-Bayes point; CRPS, MSIS(α=0.05), coverage@80 of the trajectories; and
    the mean token NLL of the true future (teacher-forced log score).
    """
    keys = ["path_smape", "path_mase", "med_smape", "med_mase", "sm_smape", "sm_mase",
            "owa_smape", "owa_mase", "crps", "msis", "cov80", "nll"]
    out = {k: [] for k in keys}
    H, season = data["horizon"], data["season"]
    for i, series_idx in enumerate(data["picked"]):
        mu, sd = data["stats"][i]
        truth = data["test"][series_idx].astype(np.float64)
        insample = data["series_raw"][i]
        naive = float(np.mean(np.abs(insample[season:] - insample[:-season])))
        ctx_len = min(len(data["dfs"][i]), data["block_size"] - H)
        ctx = data["dfs"][i].iloc[-ctx_len:]
        # 1) median path (current readout)
        p = model.predict(df=ctx, max_new_points=H, sampler=MedianSampler())
        p = p.iloc[-H:].to_numpy().reshape(-1) * sd + mu
        out["path_smape"].append(smape(p, truth)); out["path_mase"].append(mase(p, truth, insample, season))
        # 2) Monte-Carlo trajectories at temperature 1
        samp = model.predict_samples(df=ctx, max_new_points=H, num_samples=num_samples,
                                     sampler=TemperatureSampler(temperature=1.0))[..., 0] * sd + mu  # (N, H)
        for name, metric, kw in (("med", "mae", {}), ("sm", "smape", {}),
                                 ("owa", "owa", dict(naive_scale=max(naive, 1e-9), smape_ref=SMAPE_REF, mase_ref=MASE_REF))):
            pt = bayes_point(samp[:, :, None], metric, **kw)[:, 0]
            out[f"{name}_smape"].append(smape(pt, truth)); out[f"{name}_mase"].append(mase(pt, truth, insample, season))
        out["crps"].append(crps_empirical(samp, truth))
        out["msis"].append(msis(samp, truth, insample, season, alpha=0.05))
        out["cov80"].append(empirical_coverage(samp, truth, level=0.8))
        # 3) log score: NLL of the true future tokens given the context
        full = pd.concat([ctx, pd.DataFrame({"x": (truth - mu) / sd})], ignore_index=True)
        sc = model.anomaly_scores(full).to_numpy()[-H:, 0]
        out["nll"].append(float(np.nanmean(sc)))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    specs = [a.split(":") for a in sys.argv[1:]]
    # A checkpoint carrying feature_proj was trained with token features; infer the
    # lag from the projection width (1 + 2·n_lags) so the caller need not repeat it.
    assert len(specs) >= 1, "need at least one <path>:<head> argument"
    device = os.environ.get("M4_EVAL_DEVICE", "cpu")
    data = prepare()
    print(f"tokenizer={DEFAULT_TOKENIZER} n={len(data['picked'])} block_size={data['block_size']} device={device}")
    results = {}
    for path, head in specs:
        model = load_model(path, head, data, device)
        results[path] = forecast(model, data, MedianSampler())
        sm, ms, b = results[path]
        keep = ms <= np.nanquantile(ms, 0.98)
        print(f"{path}: sMAPE={sm.mean():.2f} MASE={np.nanmean(ms):.3f} | "
              f"median-series sMAPE={np.median(sm):.2f} MASE={np.nanmedian(ms):.3f} | "
              f"trimmed2% sMAPE={sm[keep].mean():.2f} MASE={ms[keep].mean():.3f} | bias={b.mean():+.3f}")
    if os.environ.get("M4_EVAL_READOUTS"):
        n_samples = int(os.environ.get("M4_NUM_SAMPLES", "100"))
        print(f"\nBayes readouts from {n_samples} trajectories + proper scores (OWA = ½(sMAPE/{SMAPE_REF} + MASE/{MASE_REF}))")
        print(f"{'checkpoint':28s} {'readout':12s} {'sMAPE':>7} {'MASE':>7} {'OWA':>7}")
        for path, head in specs:
            r = readouts_and_scores(load_model(path, head, data, device), data, n_samples)
            for name, label in (("path", "median path"), ("med", "MC median"), ("sm", "MC sMAPE-B"), ("owa", "MC OWA-B")):
                sm_, ms_ = r[f"{name}_smape"].mean(), np.nanmean(r[f"{name}_mase"])
                print(f"{path:28s} {label:12s} {sm_:7.2f} {ms_:7.3f} {0.5 * (sm_ / SMAPE_REF + ms_ / MASE_REF):7.3f}")
            print(f"{path:28s} {'scores':12s} CRPS={r['crps'].mean():.1f} MSIS={np.nanmean(r['msis']):.2f} "
                  f"cov80={r['cov80'].mean():.3f} NLL(test tokens)={r['nll'].mean():.3f}")
        return
    if os.environ.get("M4_EVAL_SUMMARY_ONLY"):
        return
    ref_path = specs[0][0]
    sm0, ms0, b0 = results[ref_path]
    for path, _ in specs[1:]:
        sm1, ms1, b1 = results[path]
        d_sm, d_ms = sm1 - sm0, ms1 - ms0
        print(f"\n{path} vs {ref_path} (per-series paired differences, negative = better)")
        print(f"  sMAPE: mean {d_sm.mean():+.2f}, median {np.median(d_sm):+.2f}, "
              f"series improved {np.mean(d_sm < 0):.0%}, worst 5 series contribute {np.sort(d_sm)[-5:].sum()/len(d_sm):+.2f} to the mean")
        print(f"  MASE : mean {np.nanmean(d_ms):+.3f}, median {np.nanmedian(d_ms):+.3f}, "
              f"series improved {np.mean(d_ms < 0):.0%}, best 5 series contribute {np.sort(d_ms)[:5].sum()/len(d_ms):+.3f} to the mean")
        # Trimmed means: drop the 2% of series with the largest MASE in the reference model.
        keep = ms0 <= np.nanquantile(ms0, 0.98)
        print(f"  trimmed (drop top 2% MASE series): sMAPE {sm0[keep].mean():.2f} -> {sm1[keep].mean():.2f}, "
              f"MASE {ms0[keep].mean():.3f} -> {ms1[keep].mean():.3f}")
        print(f"  signed relative bias: {b0.mean():+.3f} -> {b1.mean():+.3f}  (median series {np.median(b0):+.3f} -> {np.median(b1):+.3f})")
        lvl = np.array([abs(data['stats'][i][0]) / (data['stats'][i][1] + 1e-9) for i in range(len(sm0))])
        lo = lvl <= np.median(lvl)
        print(f"  by |mean|/std of series: low half ΔsMAPE {d_sm[lo].mean():+.2f}, high half {d_sm[~lo].mean():+.2f}")


if __name__ == "__main__":
    main()
