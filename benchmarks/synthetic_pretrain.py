r"""
Pretrain one DiffGPT on the synthetic corpus: Track B of leaderboard_plan.md.

The corpus is what synthetic/ writes from TempoPFN's generators: one file
<corpus>/<split>/<source>.npy per generator, float32 of shape (N, L). The
model trains on windows of the "train" split (`DiffCorpusDataLoader`, each
window in the units of a prefix of itself) and is scored on the "holdout"
split, other series of the same generators:
- the loss of its windows at every evaluation (val_loss);
- every --score-every evaluations, the forecasts of fixed holdout windows,
  --score-windows per source: --horizon points after a context of the rest
  of the block, --num-samples samples at --temperature. The MASE of the
  median and the weighted quantile loss of the quantiles 0.1, ..., 0.9
  (GIFT-Eval's CRPS) are divided per source by those of the naive forecast,
  the last value, and combined over the sources by a geometric mean, as
  GIFT-Eval combines its configurations.

--train-dtype bfloat16 trains in mixed precision, bfloat16 forward passes on
float32 master weights; --score-dtype draws the holdout forecasts from a
float16 or bfloat16 copy of the weights.

Settings are chosen on the holdout alone: no GIFT-Eval or TIME data enters
this script. --output, which must not exist yet, gets config.json,
history.csv (one row per evaluation: updates, training FLOPs =
GPT.estimate_flops() x trained tokens, losses, scores, seconds),
scores.jsonl (the scores of every source) and the checkpoints last.bin and
best.bin (the lowest holdout CRPS), written by DiffGPT.save.

Usage:
    uv run --no-sync python -m benchmarks.synthetic_pretrain \
        --output runs/synthetic-pretrain/pilot --max-iters 2000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from diff_gpt.data_loader import DiffCorpusDataLoader, corpus_domain
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import ENCODER_VERSION, prefix_rows
from diff_gpt.model.gpt import GPT
from diff_gpt.normalization import (
    NORMALIZATIONS,
    predict_samples_normalized,
    window_scale,
)
from diff_gpt.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from diff_gpt.train import Evaluation

ROOT = Path(__file__).resolve().parents[1]
# The library files a run executes; config.json records their hashes, since
# other sessions edit the working tree.
LIBRARY_SOURCES = (
    ROOT / "src/diff_gpt/data_loader.py",
    ROOT / "src/diff_gpt/diff_gpt.py",
    ROOT / "src/diff_gpt/encoder_decoder.py",
    ROOT / "src/diff_gpt/normalization.py",
    ROOT / "src/diff_gpt/train.py",
    ROOT / "src/diff_gpt/model/gpt.py",
    ROOT / "src/diff_gpt/model/block.py",
    ROOT / "src/diff_gpt/model/mha.py",
    ROOT / "src/diff_gpt/model/feed_forward.py",
    ROOT / "src/diff_gpt/model/rope.py",
    ROOT / "src/diff_gpt/model/engine.py",
    ROOT / "src/diff_gpt/model/kv_cache.py",
    ROOT / "src/diff_gpt/optimizer/schedule_free.py",
)
# The generator proportions of TempoPFN's configs/example.yaml; a source
# listed neither here nor in --weights weighs 1.
TEMPOPFN_WEIGHTS = {
    "forecast_pfn": 1.0,
    "gp": 1.0,
    "kernel": 1.0,
    "sawtooth": 1.0,
    "sinewave": 1.0,
    "step": 1.0,
    "anomaly": 1.0,
    "spike": 1.0,
    "cauker_univariate": 1.0,
    "ou_process": 3.0,
    "audio_financial_volatility": 0.1,
    "audio_multi_scale_fractal": 0.1,
    "audio_network_topology": 0.5,
    "audio_stochastic_rhythm": 0.5,
    "augmented_per_sample_2048": 2.0,
    "augmented_temp_batch_2048": 2.0,
}
QUANTILES = np.arange(1, 10) / 10
MEDIAN = 4  # the index of the quantile 0.5 in QUANTILES
HISTORY_FIELDS = (
    "step",
    "train_flops",
    "train_loss",
    "val_loss",
    "mase",
    "crps",
    "seconds",
)


def load_split(corpus: Path, split: str) -> dict[str, np.ndarray]:
    """Map each source of `split` to its (N, L) series, memory-mapped."""
    paths = sorted((corpus / split).glob("*.npy"))
    if not paths:
        msg = f"no .npy sources in {corpus / split}"
        raise FileNotFoundError(msg)
    return {path.stem: np.load(path, mmap_mode="r") for path in paths}


class Windows(NamedTuple):
    """The fixed holdout windows: each one's source, context and future."""

    sources: np.ndarray  # int64 (N,): the index of the window's source
    contexts: np.ndarray  # float64 (N, C)
    futures: np.ndarray  # float64 (N, H)


def holdout_windows(
    series: Mapping[str, np.ndarray],
    per_source: int,
    context: int,
    horizon: int,
    seed: int,
) -> Windows:
    """Draw `per_source` windows of every source: a series and an offset, uniformly."""
    rng = np.random.default_rng(seed)
    rows = context + horizon
    sources: list[int] = []
    windows: list[np.ndarray] = []
    for k, (name, data) in enumerate(series.items()):
        n, length = data.shape
        if length < rows:
            msg = f"holdout source {name!r} has series of {length} < {rows} rows"
            raise ValueError(msg)
        picks = rng.integers(n, size=per_source)
        offsets = rng.integers(length - rows + 1, size=per_source)
        windows.extend(
            np.asarray(data[i, o : o + rows], dtype=np.float64)
            for i, o in zip(picks, offsets, strict=True)
        )
        sources.extend([k] * per_source)
    stacked = np.stack(windows)
    return Windows(
        sources=np.array(sources, dtype=np.int64),
        contexts=stacked[:, :context],
        futures=stacked[:, context:],
    )


def _geometric_mean(ratios: np.ndarray) -> float:
    """Return the geometric mean of the finite positive `ratios`; NaN if none."""
    kept = ratios[np.isfinite(ratios) & (ratios > 0)]
    return float(np.exp(np.log(kept).mean())) if kept.size else float("nan")


def forecast_scores(
    samples: np.ndarray,
    windows: Windows,
    names: Sequence[str],
) -> dict[str, Any]:
    """
    Score the (N, S, H) samples of the holdout windows against the naive forecast.

    Per source, MASE is the mean over its windows of the mean |y - median|
    divided by the window's mean |step| (`window_scale`, "mean_abs_step"),
    and WQL the pinball losses 2·max(q·e, (q - 1)·e) of the errors e at the
    levels q, summed over its points, divided by the sum of |y| and averaged
    over the levels (GIFT-Eval's CRPS, MeanWeightedSumQuantileLoss). The
    naive forecast repeats the last observed value, so every quantile of it
    is that value and its WQL is sum |y - ŷ| / sum |y|. A missing future
    value counts for nothing.

    Returns the geometric means over the sources of the ratios to the naive
    forecast, "mase" and "crps" (a source whose naive score is 0 is left
    out), and per source the four scores.
    """
    count = len(names)
    truth = windows.futures
    observed = ~np.isnan(truth)
    quantiles = np.quantile(samples, QUANTILES, axis=1)  # (Q, N, H)
    seen = ~np.isnan(windows.contexts)
    last = windows.contexts[np.arange(len(seen)), seen.cumsum(axis=1).argmax(axis=1)]
    naive = np.broadcast_to(last[:, None], truth.shape)
    step = window_scale(windows.contexts.T, "mean_abs_step").scale
    points = observed.sum(axis=1)
    scored = points > 0

    def mase(point: np.ndarray) -> np.ndarray:
        error = np.where(observed, np.abs(truth - point), 0.0).sum(axis=1)
        per_window = error[scored] / points[scored] / step[scored]
        total = np.bincount(windows.sources[scored], per_window, minlength=count)
        return total / np.bincount(windows.sources[scored], minlength=count)

    def wql(levels: np.ndarray) -> np.ndarray:
        error = truth - levels
        q = QUANTILES[:, None, None]
        loss = 2 * np.maximum(q * error, (q - 1) * error).mean(axis=0)
        total = np.where(observed, loss, 0.0).sum(axis=1)
        size = np.where(observed, np.abs(truth), 0.0).sum(axis=1)
        return np.bincount(windows.sources, total, minlength=count) / np.bincount(
            windows.sources,
            size,
            minlength=count,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        model_mase, naive_mase = mase(quantiles[MEDIAN]), mase(naive)
        model_wql = wql(quantiles)
        naive_wql = wql(np.broadcast_to(naive, quantiles.shape))
        relative_mase = model_mase / naive_mase
        relative_wql = model_wql / naive_wql
    return {
        "mase": _geometric_mean(relative_mase),
        "crps": _geometric_mean(relative_wql),
        "sources": {
            name: {
                "mase": float(model_mase[k]),
                "naive_mase": float(naive_mase[k]),
                "crps": float(model_wql[k]),
                "naive_crps": float(naive_wql[k]),
            }
            for k, name in enumerate(names)
        },
    }


def _git_head() -> str:
    return subprocess.check_output(  # noqa: S603 - fixed git argv, no shell.
        [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dtype(name: str) -> torch.dtype | None:
    """Return the torch dtype called `name`; None for float32, the weights' own."""
    return None if name == "float32" else getattr(torch, name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "datasets/tempopfn-synthetic/npy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=json.loads,
        default={},
        help="JSON map of source to weight, over TempoPFN's proportions",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--tokenizer", choices=["lgt", "mash", "mash-half"])
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATIONS,
        default="mean_abs_step",
    )
    parser.add_argument("--domain-quantile", type=float, default=0.999)
    parser.add_argument("--domain-windows", type=int, default=4096)
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument("--min-context", type=int, default=16)
    parser.add_argument("--missing-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument(
        "--train-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--score-every", type=int, default=1)
    parser.add_argument("--score-windows", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-batch", type=int, default=256)
    parser.add_argument("--score-seed", type=int, default=0)
    parser.add_argument(
        "--score-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Pretrain one model as the arguments say and record its curve."""
    args = _parser().parse_args(argv)
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    train_series = load_split(args.corpus, "train")
    holdout_series = load_split(args.corpus, "holdout")
    names = sorted(train_series.keys() | holdout_series.keys())
    weights = {
        name: {**TEMPOPFN_WEIGHTS, **args.weights}.get(name, 1.0) for name in names
    }
    # The domain is the corpus's, whatever the seed: corpus_domain's own rng.
    domain = corpus_domain(
        train_series,
        args.block_size,
        args.domain_windows,
        args.order,
        quantile=args.domain_quantile,
        symmetric=not args.asymmetric,
        weights=weights,
        normalization=args.normalization,
        min_context=args.min_context,
    )
    gpt = GPT(
        vocab_size=args.vocab,
        n_embd=args.width,
        block_size=args.block_size,
        n_head=args.heads,
        n_layer=args.layers,
    ).to(args.device)
    model = DiffGPT(
        model=gpt,
        order_of_derivative=args.order,
        domain_of_definition=domain,
        tokenizer=args.tokenizer,
    )
    loader = DiffCorpusDataLoader(
        train=train_series,
        block_size=args.block_size,
        batch_size=args.batch_size,
        vocab_size=args.vocab,
        order_of_derivative=args.order,
        domain_of_definition=domain,
        device=args.device,
        val=holdout_series,
        weights=weights,
        normalization=args.normalization,
        min_context=args.min_context,
        missing_prob=args.missing_prob,
        rng=np.random.default_rng(args.seed),
        tokenizer=model.tokenizer,
    )
    # The context and the horizon fill the block: C - p + H tokens.
    context = args.block_size + prefix_rows(args.order) - args.horizon
    holdout_names = list(holdout_series)
    windows = holdout_windows(
        holdout_series,
        args.score_windows,
        context,
        args.horizon,
        args.score_seed,
    )
    flops_per_token = gpt.estimate_flops()
    tokens_per_update = args.batch_size * args.block_size * args.grad_accum
    manifest = args.corpus / "manifest.json"
    config = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git_head": _git_head(),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (Path(__file__), *LIBRARY_SOURCES)
        },
        "manifest_sha256": _sha256(manifest) if manifest.exists() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device_name": torch.cuda.get_device_name()
        if args.device.startswith("cuda")
        else platform.processor(),
        "encoder_version": ENCODER_VERSION,
        "tokenizer": model.tokenizer,
        "domain_of_definition": domain.tolist(),
        "parameters": sum(p.numel() for p in gpt.parameters()),
        "flops_per_token": flops_per_token,
        "tokens_per_update": tokens_per_update,
        "holdout": {"context": context, "windows": len(windows.sources)},
        "sources": {
            name: {
                "weight": weights[name],
                "train": list(train_series[name].shape)
                if name in train_series
                else None,
                "holdout": list(holdout_series[name].shape)
                if name in holdout_series
                else None,
            }
            for name in names
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history = args.output / "history.csv"
    with history.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=HISTORY_FIELDS).writeheader()
    sampler = TemperatureSampler(temperature=args.temperature)
    start = time.monotonic()
    state = {"evaluations": 0, "best": float("inf")}

    def on_eval(evaluation: Evaluation) -> None:
        row: dict[str, Any] = {
            "step": evaluation.step,
            "train_flops": flops_per_token * tokens_per_update * evaluation.step,
            "train_loss": evaluation.train_loss,
            "val_loss": evaluation.val_loss,
        }
        last = evaluation.step == args.max_iters
        if state["evaluations"] % args.score_every == 0 or last:
            samples = predict_samples_normalized(
                model,
                [c[:, None] for c in windows.contexts],
                max_new_points=args.horizon,
                num_samples=args.num_samples,
                method=args.normalization,
                sampler=sampler,
                batch_size=args.score_batch,
                seed=args.score_seed,
                dtype=_dtype(args.score_dtype),
            )
            scores = forecast_scores(np.stack(samples)[..., 0], windows, holdout_names)
            row |= {"mase": scores["mase"], "crps": scores["crps"]}
            with (args.output / "scores.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": evaluation.step, **scores}) + "\n")
            model.save(str(args.output / "last.bin"))
            if scores["crps"] < state["best"]:
                state["best"] = scores["crps"]
                model.save(str(args.output / "best.bin"))
        state["evaluations"] += 1
        row["seconds"] = round(time.monotonic() - start, 1)
        with history.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=HISTORY_FIELDS).writerow(row)
        print(", ".join(f"{key}={value}" for key, value in row.items()))

    model.train(
        loader=loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        warmup=args.warmup,
        use_tqdm=False,
        grad_accum_steps=args.grad_accum,
        grad_clip_norm=args.grad_clip,
        on_eval=on_eval,
        autocast=_dtype(args.train_dtype),
    )


if __name__ == "__main__":
    main()
