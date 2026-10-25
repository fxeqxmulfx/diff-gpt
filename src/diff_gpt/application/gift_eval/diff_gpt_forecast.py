r"""
Forecast the exported GIFT-Eval windows with a DiffGPT that synthetic_pretrain trained.

Run from the repository root:

    uv run --no-sync python -m diff_gpt.interfaces.gift_eval.diff_gpt_forecast \
        --run runs/synthetic-pretrain/pilot --split val

It forecasts every exported window of the split into
RUNS/forecasts/<model>/<split>/, for leaderboards' score, and records the
settings, and each configuration's counts, batch size and device, in
forecast_config.json there. A configuration whose forecasts exist is
skipped, so a stopped run resumes; --overwrite redraws them. Forecasts of
other settings go to another --model. --batch-size may differ between
runs into one folder, and runs may share one at once (say one process per
device, each with its own configurations): a run writes a configuration's
forecasts and its record entry under the folder's lock, onto the record as
it then stands.

- The model is the run's: its config.json gives the sizes and the encoder
  settings, --checkpoint (best.bin, the lowest holdout CRPS) the weights,
  read straight onto --device, so that a process that sees no GPU forecasts
  with the weights of a GPU run.
  --dtype float16 or bfloat16 casts this loaded copy for the forward passes;
  the checkpoint keeps its float32 weights.
- A context is forecast in its own units: predict_samples_normalized divides
  it by its window_scale with the normalization the model trained with.
- A pass from C rows of context, p of them the difference prefix, drawing h
  points holds C - p + h tokens in block_size. A horizon of at most
  block_size + p - --min-context points is one pass from the last
  block_size + p - H rows. A longer one is drawn in passes of that many
  points from --min-context rows: each sample path goes on from the last
  rows of its context and of the points it drew, one sample per path.
- The --num-samples paths give the quantiles at QUANTILE_LEVELS and the mean.
- Infinities count as missing. A context without an observed value in the
  rows of its first pass, or without a row past the prefix, gets a constant
  forecast: its last observed value, or 0 without one, as gluonts'
  LastValueImputation fills such a context for Seasonal_Naive.
"""

import fnmatch
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from diff_gpt.application.gift_eval.files import (
    QUANTILE_LEVELS,
    forecast_path,
    read_windows,
    windows_paths,
    write_forecasts,
)
from diff_gpt.application.normalization import predict_samples_normalized
from diff_gpt.application.paths import ROOT
from diff_gpt.application.synthetic_pretrain import LIBRARY_SOURCES
from diff_gpt.application.synthetic_rescore import load_checkpoint
from diff_gpt.domain.encoder_decoder import prefix_rows
from diff_gpt.domain.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

    from diff_gpt.application.diff_gpt import DiffGPT
    from diff_gpt.application.gift_eval.files import Windows
    from diff_gpt.application.ports import FileLock, GitHead
    from diff_gpt.domain.sampler.sampler import Sampler

TERMS = ("short", "medium", "long")
# predict_samples_normalized seeds its b-th batch with seed + b. Each pass
# of each call gets its own range of seeds, this many wide.
SEED_STRIDE = 2**16
RECORD = "forecast_config.json"
LOCK = "forecast_config.lock"


def chunk_plan(
    horizon: int,
    block_size: int,
    prefix: int,
    min_context: int,
) -> list[tuple[int, int]]:
    """
    Return the passes that draw `horizon` points: (context rows, points) each.

    A pass of C context rows and h points holds C - prefix + h tokens, at
    most block_size. A horizon of at most block_size + prefix - min_context
    points is one pass from the longest context that fits; a longer one is
    drawn in passes of that many points from `min_context` rows, the last
    pass from the longest context that fits its points.
    """
    if not prefix < min_context < block_size + prefix:
        msg = (
            f"min_context must lie in ({prefix}, {block_size + prefix}), for a "
            f"row past the prefix and a point per pass; got {min_context}"
        )
        raise ValueError(msg)
    longest = block_size + prefix - min_context
    plan = []
    left = horizon
    while left > 0:
        points = min(left, longest)
        plan.append((block_size + prefix - points, points))
        left -= points
    return plan


def _clean(context: np.ndarray) -> np.ndarray:
    """Return `context` as float64, an infinity as NaN: both are missing."""
    values = context.astype(np.float64)
    values[np.isinf(values)] = np.nan
    return values


def _last_observed(context: np.ndarray) -> float:
    """Return the last observed value of `context`, or 0 without one."""
    seen = np.flatnonzero(~np.isnan(context))
    return float(context[seen[-1]]) if seen.size else 0.0


def sample_paths(
    model: DiffGPT,
    contexts: Sequence[np.ndarray],
    plan: Sequence[tuple[int, int]],
    num_samples: int,
    *,
    normalization: str,
    sampler: Sampler,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """
    Return (len(contexts), num_samples, H) sample paths, pass by pass as `plan` says.

    The first pass draws `num_samples` futures of the last rows of each
    float64 context; every later pass one more stretch of each path, from
    the last rows of its context and of the points it drew. Pass k seeds its
    batches from seed + k·SEED_STRIDE.
    """
    rows, points = plan[0]
    first = predict_samples_normalized(
        model,
        [context[-rows:, None] for context in contexts],
        max_new_points=points,
        num_samples=num_samples,
        method=normalization,
        sampler=sampler,
        batch_size=batch_size,
        seed=seed,
    )
    paths = np.stack(first)[..., 0]  # (n, S, points)
    for k, (rows, points) in enumerate(plan[1:], start=1):
        drawn = paths.shape[2]
        need = max(0, rows - drawn)  # rows of the context in front of the path
        start = max(0, drawn - rows)
        tails = [
            np.concatenate(
                [context[max(0, len(context) - need) :], path[start:]],
            )[:, None]
            for context, context_paths in zip(contexts, paths, strict=True)
            for path in context_paths
        ]
        more = predict_samples_normalized(
            model,
            tails,
            max_new_points=points,
            num_samples=1,
            method=normalization,
            sampler=sampler,
            batch_size=batch_size,
            seed=seed + k * SEED_STRIDE,
        )
        stretch = np.stack(more)[:, 0, :, 0].reshape(len(contexts), num_samples, -1)
        paths = np.concatenate([paths, stretch], axis=2)
    return paths


def forecast(
    model: DiffGPT,
    windows: Windows,
    *,
    normalization: str,
    num_samples: int,
    min_context: int,
    sampler: Sampler,
    batch_size: int,
    windows_per_call: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return the quantiles (N, 11, H) at QUANTILE_LEVELS, the means (N, H) and a count.

    The count is that of the constant forecasts. The windows are drawn
    `windows_per_call` at a time, call c seeding its passes from
    seed + c·len(plan)·SEED_STRIDE. RuntimeError if a sample is not finite.
    """
    prefix = prefix_rows(model.order_of_derivative)
    horizon = windows.prediction_length
    plan = chunk_plan(horizon, model.model.block_size, prefix, min_context)
    count = len(windows)
    quantiles = np.empty((count, len(QUANTILE_LEVELS), horizon), dtype=np.float32)
    mean = np.empty((count, horizon), dtype=np.float32)
    constant = 0
    for call, start in enumerate(range(0, count, windows_per_call)):
        indices = np.arange(start, min(start + windows_per_call, count))
        contexts = [_clean(windows.context(i)) for i in indices]
        drawn = np.array(
            [
                len(context) > prefix and not np.isnan(context[-plan[0][0] :]).all()
                for context in contexts
            ],
        )
        for i in np.flatnonzero(~drawn):
            quantiles[indices[i]] = mean[indices[i]] = _last_observed(contexts[i])
        constant += int((~drawn).sum())
        if not drawn.any():
            continue
        paths = sample_paths(
            model,
            [contexts[i] for i in np.flatnonzero(drawn)],
            plan,
            num_samples,
            normalization=normalization,
            sampler=sampler,
            batch_size=batch_size,
            seed=seed + call * len(plan) * SEED_STRIDE,
        )
        if not np.isfinite(paths).all():
            msg = f"{windows.config}: a sample path is not finite"
            raise RuntimeError(msg)
        levels = np.quantile(paths, QUANTILE_LEVELS, axis=1)  # (11, n, H)
        quantiles[indices[drawn]] = levels.transpose(1, 0, 2)
        mean[indices[drawn]] = paths.mean(axis=1)
    return quantiles, mean, constant


def load_model(
    run: Path,
    checkpoint: str,
    device: str,
    dtype: torch.dtype | None,
) -> tuple[DiffGPT, dict[str, Any]]:
    """
    Return the run's DiffGPT with the weights of its file `checkpoint`, and its config.

    The weights are read straight onto `device`, whichever device saved them
    (synthetic_rescore.load_checkpoint): a process that sees no GPU forecasts
    with the weights of a GPU run.
    """
    model, config = load_checkpoint(run, checkpoint, device)
    if dtype is not None:
        # These weights only forecast: casting them leaves the checkpoint's be.
        model.model.to(dtype=dtype)
    return model, config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_record(folder: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """
    Return the folder's record of `settings`, a new one if it has none.

    ValueError if it holds forecasts of other settings. A record from
    before the batch size and the device were each configuration's gets
    them into its entries.
    """
    path = folder / RECORD
    if not path.exists():
        return {"settings": settings, "configs": {}}
    record = json.loads(path.read_text())
    moved = {"batch_size": record["settings"].pop("batch_size", None)}
    moved["device_name"] = record.pop("device_name", None)
    for key, value in moved.items():
        if value is not None:
            for entry in record["configs"].values():
                entry.setdefault(key, value)
    if record["settings"] != settings:
        msg = (
            f"{folder} holds forecasts of other settings, "
            f"{record['settings']}; draw these into another --model"
        )
        raise ValueError(msg)
    return record


def forecast_split(
    args: argparse.Namespace, *, git_head: GitHead, lock: FileLock
) -> None:
    """Forecast every window of the split as `args` say, one file per window."""
    checkpoint = args.run / args.checkpoint
    dtype = None if args.dtype == "float32" else getattr(torch, args.dtype)
    model, run_config = load_model(args.run, args.checkpoint, args.device, dtype)
    name = args.model or f"diff_gpt-{args.run.name}"
    min_context = args.min_context or model.model.block_size // 2
    paths = [
        path
        for path in windows_paths(args.split, args.windows)
        if path.stem.rsplit("__", 1)[-1] in args.terms
        and any(fnmatch.fnmatch(path.stem, pattern) for pattern in args.configs)
    ]
    if not paths:
        msg = f"no windows of {args.terms} matching {args.configs} in {args.windows}"
        raise FileNotFoundError(msg)
    normalization = run_config["args"]["normalization"]
    settings = {
        "run": str(args.run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "normalization": normalization,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "min_context": min_context,
        "dtype": args.dtype,
        "windows_per_call": args.windows_per_call,
        "seed": args.seed,
    }
    folder = args.forecasts / name / args.split
    read_record(folder, settings)  # other settings fail before any work
    code = {
        "git_head": git_head(),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (Path(__file__), *LIBRARY_SOURCES)
        },
        "torch": torch.__version__,
    }
    device_name = (
        torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu"
    )
    folder.mkdir(parents=True, exist_ok=True)
    sampler = TemperatureSampler(temperature=args.temperature)
    for path in paths:
        windows = read_windows(path)
        target = forecast_path(windows, name, args.forecasts)
        if target.exists() and not args.overwrite:
            print(f"{windows.config}: forecasts exist, skipped", flush=True)
            continue
        started = time.perf_counter()
        quantiles, mean, constant = forecast(
            model,
            windows,
            normalization=normalization,
            num_samples=args.num_samples,
            min_context=min_context,
            sampler=sampler,
            batch_size=args.batch_size,
            windows_per_call=args.windows_per_call,
            seed=args.seed,
        )
        seconds = time.perf_counter() - started
        plan = chunk_plan(
            windows.prediction_length,
            model.model.block_size,
            prefix_rows(model.order_of_derivative),
            min_context,
        )
        with lock(folder / LOCK):
            write_forecasts(target, windows, quantiles, mean)
            # The record as it stands now, with the entries of any other run.
            record = read_record(folder, settings) | code
            record["configs"][windows.config] = {
                "windows": len(windows),
                "horizon": windows.prediction_length,
                "plan": plan,
                "constant": constant,
                "batch_size": args.batch_size,
                "device_name": device_name,
                "seconds": round(seconds, 1),
            }
            partial = folder / f"{RECORD}.partial"
            partial.write_text(json.dumps(record, indent=2) + "\n")
            partial.replace(folder / RECORD)
        tokens = len(windows) * windows.prediction_length * args.num_samples
        print(
            f"{windows.config}: {len(windows)} windows, {len(plan)} passes, "
            f"{constant} constant, {seconds:.1f} s, "
            f"{tokens / seconds / 1e3:.1f}k points/s",
            flush=True,
        )
