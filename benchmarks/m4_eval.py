"""
Per-series diagnostics for saved M4 Hourly checkpoints (see m4_bench.py).

Usage:
    uv run --no-sync python -m benchmarks.m4_eval runs/a.pt runs/b.pt
Each argument is a state_dict path. Data preparation is m4_bench's `prepare`
(same normalisation, block_size, domain), so run with the M4_* settings of the
training run; the tokenizer comes from DIFF_GPT_TOKENIZER. Prints paired
per-series comparisons between the first checkpoint and every other one.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from benchmarks.m4_bench import mase, prepare, smape
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import DEFAULT_TOKENIZER, TokenClippingWarning
from diff_gpt.metrics import bayes_point, crps_empirical, empirical_coverage, msis
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.median import MedianSampler
from diff_gpt.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from diff_gpt.sampler.sampler import Sampler

# M4 Hourly Naive2 reference values (Makridakis et al. 2020), used for OWA.
SMAPE_REF, MASE_REF = 18.383, 2.395


def load_model(path: str, data: dict, device: str) -> DiffGPT:
    """
    Load a saved GPT state_dict from `path` and wrap it in a DiffGPT.

    The architecture comes from M4_VOCAB_SIZE, M4_N_EMBD, M4_N_HEAD and
    M4_N_LAYER (m4_bench's defaults); the `_orig_mod.` key prefix added by
    torch.compile is stripped. The model is moved to `device` in eval mode.
    """
    base = GPT(
        vocab_size=int(os.environ.get("M4_VOCAB_SIZE", "256")),
        n_embd=int(os.environ.get("M4_N_EMBD", "64")),
        block_size=data["block_size"],
        n_head=int(os.environ.get("M4_N_HEAD", "4")),
        n_layer=int(os.environ.get("M4_N_LAYER", "2")),
        use_checkpoint=False,
    )
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    base.load_state_dict(sd)
    base.to(device).eval()
    return DiffGPT(
        model=base,
        order_of_derivative=data["order"],
        domain_of_definition=data["domain"],
    )


@torch.inference_mode()
def forecast(
    model: DiffGPT,
    data: dict,
    sampler: Sampler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-series sMAPE, MASE and signed relative bias.

    The bias of a series is mean((pred-true)/|true|).
    """
    sm, ms, bias = [], [], []
    for i, series_idx in enumerate(data["picked"]):
        mu, sd = data["stats"][i]
        truth = data["test"][series_idx].astype(np.float64)
        ctx_len = min(len(data["dfs"][i]), data["block_size"] - data["horizon"])
        ctx = data["dfs"][i].iloc[-ctx_len:]
        pred = model.predict(df=ctx, max_new_points=data["horizon"], sampler=sampler)
        pred = pred.iloc[-data["horizon"] :].to_numpy().reshape(-1) * sd + mu
        sm.append(smape(pred, truth))
        ms.append(mase(pred, truth, data["series_raw"][i], data["season"]))
        bias.append(float(np.mean((pred - truth) / np.maximum(np.abs(truth), 1e-9))))
    return np.array(sm), np.array(ms), np.array(bias)


@torch.inference_mode()
def readouts_and_scores(
    model: DiffGPT,
    data: dict,
    num_samples: int,
) -> dict[str, np.ndarray]:
    """
    Bayes readouts from Monte-Carlo trajectories plus proper scores, per series.

    Returns per-series arrays: sMAPE/MASE for the median path (MedianSampler
    fed back), for the MC marginal median, the MC sMAPE-Bayes point and the MC
    OWA-Bayes point; CRPS, MSIS(alpha=0.05), coverage@80 of the trajectories; and
    the mean token NLL of the true future (teacher-forced log score).
    """
    keys = [
        "path_smape",
        "path_mase",
        "med_smape",
        "med_mase",
        "sm_smape",
        "sm_mase",
        "owa_smape",
        "owa_mase",
        "crps",
        "msis",
        "cov80",
        "nll",
    ]
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
        out["path_smape"].append(smape(p, truth))
        out["path_mase"].append(mase(p, truth, insample, season))
        # 2) Monte-Carlo trajectories at temperature 1
        samp = (
            model.predict_samples(
                df=ctx,
                max_new_points=H,
                num_samples=num_samples,
                sampler=TemperatureSampler(temperature=1.0),
            )[..., 0]
            * sd
            + mu
        )  # (N, H)
        for name, metric, kw in (
            ("med", "mae", {}),
            ("sm", "smape", {}),
            (
                "owa",
                "owa",
                {
                    "naive_scale": max(naive, 1e-9),
                    "smape_ref": SMAPE_REF,
                    "mase_ref": MASE_REF,
                },
            ),
        ):
            pt = bayes_point(samp[:, :, None], metric, **kw)[:, 0]
            out[f"{name}_smape"].append(smape(pt, truth))
            out[f"{name}_mase"].append(mase(pt, truth, insample, season))
        out["crps"].append(crps_empirical(samp, truth))
        out["msis"].append(msis(samp, truth, insample, season, alpha=0.05))
        out["cov80"].append(empirical_coverage(samp, truth, level=0.8))
        # 3) log score: NLL of the true future tokens given the context
        full = pd.concat(
            [ctx, pd.DataFrame({"x": (truth - mu) / sd})],
            ignore_index=True,
        )
        sc = model.anomaly_scores(full).to_numpy()[-H:, 0]
        out["nll"].append(float(np.nanmean(sc)))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    """
    Evaluate the checkpoints given as arguments and print diagnostics.

    Prints a median-path summary per checkpoint. Then, if M4_EVAL_READOUTS is
    set, prints Bayes readouts and proper scores from M4_NUM_SAMPLES
    trajectories; else, unless M4_EVAL_SUMMARY_ONLY is set, prints paired
    per-series comparisons of every checkpoint against the first one.
    M4_EVAL_DEVICE selects the device (default cpu).
    """
    paths = sys.argv[1:]
    if not paths:
        msg = "need at least one checkpoint path"
        raise SystemExit(msg)
    device = os.environ.get("M4_EVAL_DEVICE", "cpu")
    data = prepare()
    if data["quantile"] is not None:
        # A quantile domain clips the steps beyond it on purpose.
        warnings.filterwarnings("ignore", category=TokenClippingWarning)
    print(
        f"tokenizer={DEFAULT_TOKENIZER} n={len(data['picked'])} "
        f"block_size={data['block_size']} device={device}",
    )
    results = {}
    for path in paths:
        model = load_model(path, data, device)
        results[path] = forecast(model, data, MedianSampler())
        sm, ms, b = results[path]
        keep = ms <= np.nanquantile(ms, 0.98)
        print(
            f"{path}: sMAPE={sm.mean():.2f} MASE={np.nanmean(ms):.3f} | "
            f"median-series sMAPE={np.median(sm):.2f} MASE={np.nanmedian(ms):.3f} | "
            f"trimmed2% sMAPE={sm[keep].mean():.2f} MASE={ms[keep].mean():.3f} | "
            f"bias={b.mean():+.3f}",
        )
    if os.environ.get("M4_EVAL_READOUTS"):
        n_samples = int(os.environ.get("M4_NUM_SAMPLES", "100"))
        print(
            f"\nBayes readouts from {n_samples} trajectories + proper scores "
            f"(OWA = ½(sMAPE/{SMAPE_REF} + MASE/{MASE_REF}))",
        )
        print(f"{'checkpoint':28s} {'readout':12s} {'sMAPE':>7} {'MASE':>7} {'OWA':>7}")
        for path in paths:
            r = readouts_and_scores(load_model(path, data, device), data, n_samples)
            for name, label in (
                ("path", "median path"),
                ("med", "MC median"),
                ("sm", "MC sMAPE-B"),
                ("owa", "MC OWA-B"),
            ):
                sm_, ms_ = r[f"{name}_smape"].mean(), np.nanmean(r[f"{name}_mase"])
                print(
                    f"{path:28s} {label:12s} {sm_:7.2f} {ms_:7.3f} "
                    f"{0.5 * (sm_ / SMAPE_REF + ms_ / MASE_REF):7.3f}",
                )
            print(
                f"{path:28s} {'scores':12s} CRPS={r['crps'].mean():.1f} "
                f"MSIS={np.nanmean(r['msis']):.2f} "
                f"cov80={r['cov80'].mean():.3f} NLL(test tokens)={r['nll'].mean():.3f}",
            )
        return
    if os.environ.get("M4_EVAL_SUMMARY_ONLY"):
        return
    ref_path = paths[0]
    sm0, ms0, b0 = results[ref_path]
    for path in paths[1:]:
        sm1, ms1, b1 = results[path]
        d_sm, d_ms = sm1 - sm0, ms1 - ms0
        print(
            f"\n{path} vs {ref_path} "
            "(per-series paired differences, negative = better)",
        )
        print(
            f"  sMAPE: mean {d_sm.mean():+.2f}, median {np.median(d_sm):+.2f}, "
            f"series improved {np.mean(d_sm < 0):.0%}, worst 5 series contribute "
            f"{np.sort(d_sm)[-5:].sum() / len(d_sm):+.2f} to the mean",
        )
        print(
            f"  MASE : mean {np.nanmean(d_ms):+.3f}, median {np.nanmedian(d_ms):+.3f}, "
            f"series improved {np.mean(d_ms < 0):.0%}, best 5 series contribute "
            f"{np.sort(d_ms)[:5].sum() / len(d_ms):+.3f} to the mean",
        )
        # Trimmed means: drop the 2% of series with the largest MASE in the
        # reference model.
        keep = ms0 <= np.nanquantile(ms0, 0.98)
        print(
            "  trimmed (drop top 2% MASE series): "
            f"sMAPE {sm0[keep].mean():.2f} -> {sm1[keep].mean():.2f}, "
            f"MASE {ms0[keep].mean():.3f} -> {ms1[keep].mean():.3f}",
        )
        print(
            f"  signed relative bias: {b0.mean():+.3f} -> {b1.mean():+.3f}  "
            f"(median series {np.median(b0):+.3f} -> {np.median(b1):+.3f})",
        )
        lvl = np.array(
            [
                abs(data["stats"][i][0]) / (data["stats"][i][1] + 1e-9)
                for i in range(len(sm0))
            ],
        )
        lo = lvl <= np.median(lvl)
        print(
            f"  by |mean|/std of series: low half ΔsMAPE {d_sm[lo].mean():+.2f}, "
            f"high half {d_sm[~lo].mean():+.2f}",
        )


if __name__ == "__main__":
    main()
