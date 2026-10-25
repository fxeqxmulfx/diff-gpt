r"""
Pretrain one DiffGPT on the synthetic corpus: Track B of leaderboard_plan.md.

The corpus is what synthetic/ writes from TempoPFN's generators: one file
<corpus>/<split>/<source>.npy per generator, float32 of shape (N, L), and
its manifest <corpus>/../manifest.json, whose SHA-256 config.json records. The
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
  GIFT-Eval combines its configurations. --score-samples also scores
  forecasts of fewer samples, from disjoint groups of the ones drawn, so
  that one draw compares the numbers of samples S.

--train-dtype bfloat16 trains in mixed precision, bfloat16 forward passes on
float32 master weights; --score-dtype draws the holdout forecasts from a
float16 or bfloat16 copy of the weights. --gradient-checkpointing auto (the
default) recomputes each block's activations in the backward pass on a GPU,
whose memory needs it, and keeps them on a CPU, which is faster there and
trains the same weights, bit for bit.

Settings are chosen on the holdout alone: no GIFT-Eval or TIME data enters
this script. --output, which must not exist yet, gets config.json,
history.csv (one row per evaluation: updates, training FLOPs =
GPT.estimate_flops() x trained tokens, losses, scores, seconds in all and
spent scoring, then the scores of each --score-samples), scores.jsonl (the
scores of every source) and the checkpoints last.bin and best.bin (the
lowest holdout CRPS), written by DiffGPT.save. --keep-every FLOPS also keeps
the weights of the first evaluation at or past each multiple of FLOPS
training FLOPs, as step_<updates>_flops_<FLOPs>.bin, to compare models at
equal compute.

Every evaluation also writes state.pt, all that the run goes on from: the
weights, the optimizer's state (z, v, the step count that places the warmup,
and the weighting of the average), the loader's generator, the global RNGs
of torch, CUDA, numpy and python, and the counts of evaluations and kept
copies, the best CRPS and the seconds so far. --resume goes on from it to
--max-iters (the saved one, or more updates), with the settings of
config.json: the updates and evaluations that follow are those of an
unbroken run, bit for bit on the same device and threads. It drops what
later evaluations wrote to history.csv and scores.jsonl before a crash and
adds the resume to config.json's "resumes". A finished run trains on as a
longer unbroken one would when its updates are a multiple of
--eval-interval; otherwise its last evaluation drew batches that the longer
run does not. With --score-every above 1, that evaluation was also scored,
as the last.

Usage:
    uv run --no-sync python -m diff_gpt.interfaces.synthetic_pretrain \
        --output runs/synthetic-pretrain/pilot --max-iters 2000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from diff_gpt.application.data_loader import DiffCorpusDataLoader, corpus_domain
from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.application.normalization import predict_samples_normalized
from diff_gpt.application.paths import ROOT
from diff_gpt.application.ports import UsageError
from diff_gpt.application.train import train
from diff_gpt.domain.encoder_decoder import ENCODER_VERSION, prefix_rows
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.normalization import (
    window_scale,
)
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW
from diff_gpt.domain.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from diff_gpt.application.ports import GitHead
    from diff_gpt.application.train import Evaluation

# The library files a run executes; config.json records their hashes, since
# other sessions edit the working tree.
LIBRARY_SOURCES = (
    ROOT / "src/diff_gpt/application/data_loader.py",
    ROOT / "src/diff_gpt/application/diff_gpt.py",
    ROOT / "src/diff_gpt/domain/encoder_decoder.py",
    ROOT / "src/diff_gpt/domain/normalization.py",
    ROOT / "src/diff_gpt/application/normalization.py",
    ROOT / "src/diff_gpt/application/train.py",
    ROOT / "src/diff_gpt/domain/model/gpt.py",
    ROOT / "src/diff_gpt/domain/model/block.py",
    ROOT / "src/diff_gpt/domain/model/mha.py",
    ROOT / "src/diff_gpt/domain/model/feed_forward.py",
    ROOT / "src/diff_gpt/domain/model/rope.py",
    ROOT / "src/diff_gpt/domain/model/engine.py",
    ROOT / "src/diff_gpt/domain/model/kv_cache.py",
    ROOT / "src/diff_gpt/domain/optimizer/schedule_free.py",
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
    "score_seconds",
)
# What --resume goes on from, written at every evaluation.
RESUME_STATE = "state.pt"
STATE_VERSION = 1
# The options --resume takes; the rest are the run's, from its config.json.
RESUME_OPTIONS = frozenset({"output", "resume", "max_iters"})
# What config.json says of the corpus and the model, which a resumed run
# must find the same.
SAME_RUN = (
    "manifest_sha256",
    "encoder_version",
    "tokenizer",
    "domain_of_definition",
    "parameters",
    "flops_per_token",
    "tokens_per_update",
    "holdout",
    "sources",
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
    group: int | None = None,
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

    `group` scores forecasts of fewer samples: the S samples of each window
    split into S // `group` disjoint groups of `group` consecutive samples,
    leaving out the rest; each group's quantiles make one forecast, and a
    source's MASE and WQL are their means over the groups. The samples of a
    window are independent draws, so the means estimate the scores of
    `group` samples with less noise than one group would. None scores all
    S samples as one forecast; a `group` outside [1, S] raises ValueError.

    Returns the geometric means over the sources of the ratios to the naive
    forecast, "mase" and "crps" (a source whose naive score is 0 is left
    out), the samples per forecast and the number of groups, and per source
    the four scores.
    """
    size = samples.shape[1] if group is None else group
    if not 1 <= size <= samples.shape[1]:
        msg = f"group must lie in [1, {samples.shape[1]}], got {group}"
        raise ValueError(msg)
    starts = range(0, samples.shape[1] - size + 1, size)
    count = len(names)
    truth = windows.futures
    observed = ~np.isnan(truth)
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
        naive_mase = mase(naive)
        naive_wql = wql(np.broadcast_to(naive, (len(QUANTILES), *truth.shape)))
        model_mase = np.zeros(count)
        model_wql = np.zeros(count)
        for start in starts:
            # (Q, N, H): the quantiles of the group's samples.
            quantiles = np.quantile(
                samples[:, start : start + size],
                QUANTILES,
                axis=1,
            )
            model_mase += mase(quantiles[MEDIAN]) / len(starts)
            model_wql += wql(quantiles) / len(starts)
        relative_mase = model_mase / naive_mase
        relative_wql = model_wql / naive_wql
    return {
        "mase": _geometric_mean(relative_mase),
        "crps": _geometric_mean(relative_wql),
        "samples": size,
        "groups": len(starts),
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dtype(name: str) -> torch.dtype | None:
    """Return the torch dtype called `name`; None for float32, the weights' own."""
    return None if name == "float32" else getattr(torch, name)


def _rng_states(device: str) -> dict[str, Any]:
    """
    Return the global RNG states: torch's, CUDA's on a CUDA device, numpy's, python's.

    The numpy keys are a list, which torch.load(weights_only=True) reads.
    """
    # The global numpy state, for code that draws from it.
    name, keys, position, has_gauss, gauss = np.random.get_state()  # noqa: NPY002
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
        "numpy": (name, keys.tolist(), position, has_gauss, gauss),
        "python": random.getstate(),
    }


def _set_rng_states(states: Mapping[str, Any]) -> None:
    """Restore the global RNG states that `_rng_states` returned."""
    torch.set_rng_state(states["torch"])
    if states["cuda"] is not None:
        torch.cuda.set_rng_state_all(states["cuda"])
    name, keys, position, has_gauss, gauss = states["numpy"]
    np.random.set_state(  # noqa: NPY002
        (name, np.array(keys, dtype=np.uint32), position, has_gauss, gauss),
    )
    random.setstate(states["python"])


def _save_atomically(save: Callable[[str], None], path: Path) -> None:
    """
    Write `path` with `save`, so that a crash leaves the old file or the new one.

    `save` writes <stem>.partial beside it, which then replaces `path`.
    torch.save names the archive inside a file after its stem, so the bytes
    are those that `save` would write to `path`.
    """
    partial_path = path.with_suffix(".partial")
    save(str(partial_path))
    partial_path.replace(path)


def _drop_after(path: Path, step: int, *, header: bool) -> None:
    """
    Keep the lines of `path` that evaluations up to `step` wrote, and its header.

    A line of history.csv starts with its step, one of scores.jsonl holds it
    under "step"; a line cut short by a crash has no newline and goes too.
    """
    lines = path.read_bytes().splitlines(keepends=True)
    head, body = (lines[:1], lines[1:]) if header else ([], lines)

    def step_of(line: bytes) -> int:
        return int(line.split(b",", 1)[0]) if header else json.loads(line)["step"]

    kept = [line for line in body if line.endswith(b"\n") and step_of(line) <= step]
    path.write_bytes(b"".join([*head, *kept]))


def resumed(
    args: argparse.Namespace,
    given: set[str],
) -> tuple[argparse.Namespace, dict[str, Any], dict[str, Any]]:
    """
    Return the settings, config.json and saved state of the run in --output.

    The settings are those of config.json, with --max-iters if `given` holds
    it. A run without state.pt, or a --max-iters not above the updates made,
    raises UsageError.
    """
    path = args.output / RESUME_STATE
    if not path.is_file():
        msg = f"--resume goes on from {path}, which does not exist"
        raise UsageError(msg)
    config = json.loads((args.output / "config.json").read_text())
    settings = {**vars(args), **config["args"], "output": args.output, "resume": True}
    settings["corpus"] = Path(settings["corpus"])
    if "max_iters" in given:
        settings["max_iters"] = args.max_iters
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["version"] != STATE_VERSION:
        msg = f"{path} is of version {saved['version']}, not {STATE_VERSION}"
        raise UsageError(msg)
    if not saved["step"] < settings["max_iters"]:
        msg = f"the run has made {saved['step']} updates: give --max-iters above them"
        raise UsageError(msg)
    return argparse.Namespace(**settings), config, saved


def gradient_checkpointing(setting: str, device: str) -> bool:
    """
    Return whether the model recomputes its activations in the backward pass.

    `setting` is --gradient-checkpointing: "on", "off", or "auto", which
    recomputes them on a GPU, whose memory needs it, and keeps them on a CPU.
    """
    if setting == "auto":
        return torch.device(device).type != "cpu"
    return setting == "on"


def pretrain(
    args: argparse.Namespace,
    config: dict[str, Any],
    saved: dict[str, Any] | None,
    *,
    git_head: GitHead,
) -> None:
    """Pretrain one model as `args` say and record its curve."""
    if saved is None:
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
        use_checkpoint=gradient_checkpointing(args.gradient_checkpointing, args.device),
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
    manifest = args.corpus.parent / "manifest.json"  # export writes it next to npy/
    run = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git_head": git_head(),
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
        "gradient_checkpointing": gpt.use_checkpoint,
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
    history = args.output / "history.csv"
    scores_path = args.output / "scores.jsonl"
    fields = [
        *HISTORY_FIELDS,
        *(
            f"{score}_s{size}"
            for size in args.score_samples
            for score in ("mase", "crps")
        ),
    ]
    if saved is None:
        config = run
        with history.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    else:
        differing = [key for key in SAME_RUN if run[key] != config[key]]
        if differing:
            msg = (
                f"the corpus and the code now give other {', '.join(differing)} "
                f"than {args.output / 'config.json'}"
            )
            raise UsageError(msg)
        config["args"]["max_iters"] = args.max_iters
        config.setdefault("resumes", []).append(
            {
                "step": saved["step"],
                "max_iters": args.max_iters,
                "git_head": run["git_head"],
                "source_sha256": run["source_sha256"],
                "started": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            },
        )
        _drop_after(history, saved["step"], header=True)
        if scores_path.exists():
            _drop_after(scores_path, saved["step"], header=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sampler = TemperatureSampler(temperature=args.temperature)
    # The optimizer train() would build, built here to save and restore it.
    optimizer = ScheduleFreeAdamW(
        gpt.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        warmup=args.warmup,
    )
    state = {"evaluations": 0, "best": float("inf"), "kept": 0}
    offset = 0.0  # the seconds of the run before this process
    if saved is not None:
        gpt.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        loader.rng.bit_generator.state = saved["loader_rng"]
        _set_rng_states(saved["rng"])
        state = {key: saved[key] for key in state}
        offset = saved["seconds"]
    start = time.monotonic()

    def on_eval(evaluation: Evaluation) -> None:
        flops = flops_per_token * tokens_per_update * evaluation.step
        row: dict[str, Any] = {
            "step": evaluation.step,
            "train_flops": flops,
            "train_loss": evaluation.train_loss,
            "val_loss": evaluation.val_loss,
        }
        last = evaluation.step == args.max_iters
        if state["evaluations"] % args.score_every == 0 or last:
            scoring = time.monotonic()
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
            drawn = np.stack(samples)[..., 0]
            scores = forecast_scores(drawn, windows, holdout_names)
            row |= {"mase": scores["mase"], "crps": scores["crps"]}
            fewer = {
                size: forecast_scores(drawn, windows, holdout_names, group=size)
                for size in args.score_samples
            }
            for size, scored in fewer.items():
                row |= {
                    f"mase_s{size}": scored["mase"],
                    f"crps_s{size}": scored["crps"],
                }
            row["score_seconds"] = round(time.monotonic() - scoring, 1)
            record = {
                "step": evaluation.step,
                **scores,
                "fewer_samples": {str(size): scored for size, scored in fewer.items()},
            }
            with scores_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            _save_atomically(model.save, args.output / "last.bin")
            if scores["crps"] < state["best"]:
                state["best"] = scores["crps"]
                _save_atomically(model.save, args.output / "best.bin")
        state["evaluations"] += 1
        row["seconds"] = round(offset + time.monotonic() - start, 1)
        with history.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(", ".join(f"{key}={value}" for key, value in row.items()))
        if args.keep_every is not None and flops // args.keep_every > state["kept"]:
            state["kept"] = int(flops // args.keep_every)
            name = f"step_{evaluation.step}_flops_{flops:.4e}.bin"
            _save_atomically(model.save, args.output / name)
        resume = {
            "version": STATE_VERSION,
            "step": evaluation.step,
            "model": gpt.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loader_rng": loader.rng.bit_generator.state,
            "rng": _rng_states(args.device),
            **state,
            "seconds": row["seconds"],
        }
        _save_atomically(partial(torch.save, resume), args.output / RESUME_STATE)

    train(
        gpt,
        loader,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        use_tqdm=False,
        grad_accum_steps=args.grad_accum,
        grad_clip_norm=args.grad_clip,
        optimizer=optimizer,
        on_eval=on_eval,
        autocast=_dtype(args.train_dtype),
        start_step=0 if saved is None else saved["step"],
    )
