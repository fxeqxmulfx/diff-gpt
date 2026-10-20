"""
Grokking of row-dependent finite-field time series with the actual DiffGPT.

One column x evolves as x[t+2] = x[t] / x[t+1] mod 97 (or Fibonacci mod 97).
Differences are taken down rows, with no synthetic zero prefix. Complete
trajectory orbits are disjoint between training and test. The optimizer is
the repository's ScheduleFreeAdamW throughout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from benchmarks.grokking.div97 import (
    LIBRARY_SOURCES,
    NORMALIZATION,
    REPLAY_LOSS_TOLERANCE,
    ROOT,
    TARGET_ACCURACY,
    Criterion,
    P,
    SplitMix64,
    digest,
    domain,
    load_checkpoint,
    vocabulary,
    write_json,
)
from benchmarks.grokking.recurrence_field_sampler import CanonicalFieldSampler
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import ENCODER_VERSION, decode, encode, resolve_tokenizer
from diff_gpt.model.gpt import GPT
from diff_gpt.optimizer.schedule_free import ScheduleFreeAdamW
from diff_gpt.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from collections.abc import Sequence

BATCH = 512
DEFAULT_BETA1 = 0.9
RATE_FIELDS = ("learning_rate", "base_rate", "averaging_coefficient")
EVALUATION_BATCH = 1024
CHECKPOINT_EVERY = 250
MIN_CONTEXT = 2
SOURCE_SNAPSHOT = {
    str(path.relative_to(ROOT)): path.read_bytes()
    for path in (
        Path(__file__),
        ROOT / "benchmarks/grokking/div97.py",
        ROOT / "benchmarks/grokking/recurrence_field_sampler.py",
        *LIBRARY_SOURCES,
    )
}


@dataclass(frozen=True)
class Config:
    """One temporal experiment; the optimizer implementation stays fixed."""

    task: str = "division"
    order: int = 1
    context: int = 4
    width: int = 128
    layers: int = 2
    heads: int = 4
    seed: int = 11
    learning_rate: float = 0.01
    weight_decay: float = 1.0
    beta1: float = DEFAULT_BETA1
    beta2: float = 0.98
    warmup: int = 10
    epochs: int = 3000
    after99: int = 100
    tokenizer: str = "lgt"
    grid_domain: float = 0.0
    vocab_size: int = 0
    readout: str = "argmax"
    supervise_from: int = 0
    batching: str = "tail"
    batch_size: int = BATCH
    micro_batch_size: int = 0
    restart_step: int = 0
    restart_warmup: int = 0

    def name(self) -> str:
        """Keep distinct data representations and optimizer settings separate."""
        result = (
            f"{self.task}-k{self.order}-c{self.context}-w{self.width}"
            f"-l{self.layers}-h{self.heads}-sf-lr{self.learning_rate:g}"
            f"-wd{self.weight_decay:g}-b2{self.beta2:g}-wu{self.warmup}-s{self.seed}"
        )
        if self.beta1 != DEFAULT_BETA1:
            result += f"-b1{self.beta1:g}"
        if self.tokenizer != "lgt" or self.grid_domain or self.vocab_size:
            result += (
                f"-{self.tokenizer}-v{grid_vocabulary(self)}-d{grid_domain(self)[0]:g}"
            )
        if self.readout != "argmax":
            result += f"-readout-{self.readout}"
        if self.supervise_from:
            result += f"-sup{self.supervise_from}"
        if self.batching != "tail":
            result += f"-batch-{self.batching}"
        if self.batch_size != BATCH:
            result += f"-bs{self.batch_size}"
        if self.micro_batch_size:
            result += f"-mbs{self.micro_batch_size}"
        if self.restart_step:
            result += f"-restart{self.restart_step}-rw{self.restart_warmup}"
        return result


def grid_vocabulary(config: Config) -> int:
    """Use the configured vocabulary, or the original exact integer grid."""
    return config.vocab_size or vocabulary(config.order)


def batch_slices(size: int, mode: str, batch_size: int = BATCH) -> list[slice]:
    """Keep a fixed update count while optionally spreading a small final batch."""
    if batch_size < 1:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if mode == "tail":
        return [
            slice(first, min(first + batch_size, size))
            for first in range(0, size, batch_size)
        ]
    if mode == "balanced":
        count = math.ceil(size / batch_size)
        return [
            slice(index * size // count, (index + 1) * size // count)
            for index in range(count)
        ]
    msg = f"unknown batching mode: {mode}"
    raise ValueError(msg)


def grid_domain(config: Config) -> np.ndarray:
    """Use the same derivative domain for both tokenizer implementations."""
    return (
        np.array([config.grid_domain], dtype=np.float64)
        if config.grid_domain
        else domain(config.order)
    )


def encode_windows(raw: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Encode independent temporal windows with the real row-wise tokenizer."""
    resolve_tokenizer(config.tokenizer)
    if not config.grid_domain and not config.vocab_size:
        # On this exact unit grid MASH and LGT both have zero anchor fraction.
        tokens = np.diff(raw, n=config.order, axis=1) + grid_vocabulary(config) // 2
        return tokens, np.ones(len(raw))
    _, scale, tokens = encode(
        raw.T.astype(np.float64),
        grid_vocabulary(config),
        np.repeat(grid_domain(config), len(raw)),
        config.order,
        tokenizer=config.tokenizer,
    )
    return tokens.T.copy(), scale


def future_grid(past: np.ndarray, config: Config) -> np.ndarray:
    """Derive one-step decoder bases from observed rows, never the future."""
    prefix = past[:, -config.order :].T.astype(np.float64)
    _, scale, _ = encode(
        prefix,
        grid_vocabulary(config),
        np.repeat(grid_domain(config), len(past)),
        config.order,
        tokenizer=config.tokenizer,
    )
    center = np.full((1, len(past)), grid_vocabulary(config) // 2, dtype=np.int64)
    bases = decode(
        prefix,
        scale,
        center,
        grid_vocabulary(config),
        config.order,
        tokenizer=config.tokenizer,
    )[-1]
    return np.column_stack((bases, 1 / scale))


def transition(state: tuple[int, int], task: str) -> tuple[int, int]:
    """Advance an invertible recurrence over the finite field."""
    a, b = state
    if task == "division":
        return b, a * pow(b, -1, P) % P
    if task == "fibonacci":
        return b, (a + b) % P
    msg = f"unknown recurrence: {task}"
    raise ValueError(msg)


def trajectory_orbits(task: str) -> list[list[tuple[int, int]]]:
    """Enumerate each cycle once, excluding undefined and absorbing states."""
    states = (
        {(a, b) for a in range(1, P) for b in range(1, P)}
        if task == "division"
        else {(a, b) for a in range(P) for b in range(P)} - {(0, 0)}
    )
    remaining = states.copy()
    result = []
    while remaining:
        first = min(remaining)
        state = first
        orbit = []
        while state in remaining:
            remaining.remove(state)
            orbit.append(state)
            state = transition(state, task)
        if state != first:
            msg = "the recurrence must partition the state space into cycles"
            raise ValueError(msg)
        result.append(orbit)
    return result


def windows(states: Sequence[tuple[int, int]], config: Config) -> np.ndarray:
    """Return actual successive rows, including the one held-out future row."""
    length = config.order + config.context + 1
    raw = np.empty((len(states), length), dtype=np.int64)
    for row, state in enumerate(states):
        a, b = state
        raw[row, :2] = (a, b)
        for position in range(2, length):
            a, b = transition((a, b), config.task)
            raw[row, position] = b
    return raw


def dataset(config: Config) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Split entire orbits, then shuffle the states within each split."""
    if config.order < 1 or config.context < MIN_CONTEXT:
        msg = "temporal experiments require order >=1 and context >=2"
        raise ValueError(msg)
    if config.readout not in {"argmax", "field"}:
        msg = f"unknown readout: {config.readout}"
        raise ValueError(msg)
    if config.readout == "field" and (config.grid_domain or config.vocab_size):
        msg = "field readout requires the exact integer grid"
        raise ValueError(msg)
    if not 0 <= config.supervise_from <= config.context:
        msg = "supervise_from must be zero (last only) or a valid context length"
        raise ValueError(msg)
    if config.batching not in {"tail", "balanced"}:
        msg = f"unknown batching mode: {config.batching}"
        raise ValueError(msg)
    if config.batch_size < 1:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if config.micro_batch_size < 0:
        msg = "micro_batch_size must be zero or positive"
        raise ValueError(msg)
    if config.restart_step < 0 or config.restart_warmup < 0:
        msg = "restart_step and restart_warmup must be nonnegative"
        raise ValueError(msg)
    if config.restart_warmup and not config.restart_step:
        msg = "repeat warmup requires a positive restart_step"
        raise ValueError(msg)
    orbits = trajectory_orbits(config.task)
    indices = list(range(len(orbits)))
    SplitMix64(config.seed ^ 0x6F72626974730001).shuffle(indices)
    train, test, train_ids, test_ids = [], [], [], []
    for index in indices:
        if len(train) <= len(test):
            train.extend(orbits[index])
            train_ids.append(index)
        else:
            test.extend(orbits[index])
            test_ids.append(index)
    SplitMix64(config.seed ^ 0x747261696E000002).shuffle(train)
    SplitMix64(config.seed ^ 0x7465737400000003).shuffle(test)
    raw = windows([*train, *test], config)
    _, scale, encoded = encode(
        raw[0, :, None].astype(np.float64),
        grid_vocabulary(config),
        grid_domain(config),
        config.order,
        tokenizer=config.tokenizer,
    )
    tokens, _ = encode_windows(raw, config)
    np.testing.assert_array_equal(encoded[:, 0], tokens[0])
    if tokens.min() < 0 or tokens.max() >= grid_vocabulary(config):
        msg = "the integer derivative exceeds its vocabulary"
        raise ValueError(msg)
    contexts = tokens[:, :-1]
    unique = np.unique(contexts, axis=0).shape[0]
    # Multiple hidden states may have the same derivative history. The true
    # next-token ambiguity must be measured rather than silently ignored.
    groups = {}
    for context, target in zip(contexts, tokens[:, -1], strict=True):
        counts = groups.setdefault(tuple(context), {})
        counts[int(target)] = counts.get(int(target), 0) + 1
    best = sum(max(counts.values()) for counts in groups.values()) / len(raw)
    metadata = {
        "task": config.task,
        "columns": ["x"],
        "difference_axis": "rows",
        "synthetic_prefix": False,
        "orbit_lengths": [len(orbit) for orbit in orbits],
        "train_orbits": train_ids,
        "test_orbits": test_ids,
        "train_rows": len(train),
        "test_rows": len(test),
        "unique_contexts": unique,
        "optimal_full_table_token_accuracy": best,
        "encoder_version": ENCODER_VERSION,
        "tokenizer": config.tokenizer,
        "vocab_size": grid_vocabulary(config),
        "domain": grid_domain(config).tolist(),
        "scale": scale.tolist(),
        "raw_sha256": digest(raw),
    }
    return raw, len(train), metadata


def tensors(
    raw: np.ndarray, config: Config, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply k differences along each trajectory's row axis; supervise its future."""
    tokens, _ = encode_windows(raw, config)
    x = torch.as_tensor(tokens[:, :-1].copy(), device=device)
    y = torch.full_like(x, -100)
    start = config.supervise_from or config.context
    y[:, start - 1 :] = torch.as_tensor(tokens[:, start:].copy(), device=device)
    return x, y


def build(config: Config, device: str) -> GPT:
    """Instantiate the unmodified repository model."""
    return GPT(
        vocab_size=grid_vocabulary(config),
        n_embd=config.width,
        n_head=config.heads,
        n_layer=config.layers,
        block_size=config.context + 1,
        use_checkpoint=False,
    ).to(device)


def update_learning_rate(config: Config, optimizer: ScheduleFreeAdamW) -> None:
    """Ramp a fork's rate without resetting its global step or averaged iterates."""
    group = optimizer.param_groups[0]
    if config.restart_step and group["step"] >= config.restart_step:
        local_step = group["step"] - config.restart_step + 1
        ramp = (
            min(1.0, local_step / config.restart_warmup)
            if config.restart_warmup
            else 1.0
        )
        group["lr"] = config.learning_rate * ramp


def optimizer_progress(optimizer: ScheduleFreeAdamW) -> dict[str, float | int]:
    """Expose actual base-update and averaging rates, rather than a nominal LR."""
    group = optimizer.param_groups[0]
    step = group["step"]
    ramp = min(1.0, step / group["warmup"]) if group["warmup"] else 1.0
    rate = group["lr"] * ramp
    return {
        "step": step,
        "learning_rate": group["lr"],
        "base_rate": rate,
        "averaging_total": group["total"],
        "averaging_coefficient": (
            group["maximum"] ** 2 / group["total"] if group["total"] else 0.0
        ),
    }


def train_batch(
    model: GPT,
    optimizer: ScheduleFreeAdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    micro_batch_size: int = 0,
) -> None:
    """Take one optimizer step over every row of one effective batch."""
    optimizer.zero_grad(set_to_none=True)
    limit = micro_batch_size or len(indices)
    for first in range(0, len(indices), limit):
        part = indices[first : first + limit]
        _, loss = model(x[part], y[part])
        # Every row has the same number of supervised tokens. Weighting by
        # row count makes the accumulated gradient the full-batch mean.
        if len(part) != len(indices):
            loss = loss * (len(part) / len(indices))
        loss.backward()
    optimizer.step()


@torch.no_grad()
def evaluate(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    raw: torch.Tensor,
    config: Config,
    decoder_grid: torch.Tensor | None,
) -> dict[str, float]:
    """Score tokens and invert their finite differences against the true future."""
    totals = torch.zeros(4, device=x.device, dtype=torch.float64)
    for first in range(0, len(x), EVALUATION_BATCH):
        stop = first + EVALUATION_BATCH
        logits, _ = model(x[first:stop])
        logits = logits[:, -1]
        target = y[first:stop, -1]
        native = logits.argmax(-1)
        predicted = (
            CanonicalFieldSampler(
                raw[first:stop, :-1],
                config.order,
                grid_vocabulary(config),
                P,
                lower=int(config.task == "division"),
            )(logits)[:, 0]
            if config.readout == "field"
            else native
        )
        value = predicted - grid_vocabulary(config) // 2
        if decoder_grid is None:
            for lag in range(1, config.order + 1):
                coefficient = (-1) ** lag * math.comb(config.order, lag)
                value = value - coefficient * raw[first:stop, -1 - lag]
        else:
            value = torch.round(
                decoder_grid[first:stop, 0] + value * decoder_grid[first:stop, 1]
            )
        totals[0] += F.cross_entropy(logits, target, reduction="sum")
        totals[1] += (predicted == target).sum()
        totals[2] += (value == raw[first:stop, -1]).sum()
        totals[3] += (native == target).sum()
    # Divide on the host: CUDA reciprocal multiplication may differ by one ULP
    # from NumPy's division even when the integer counts agree exactly.
    loss_sum, correct, decoded_correct, native_correct = totals.tolist()
    result = {
        "loss": loss_sum / len(x),
        "accuracy": correct / len(x),
        "decoded_accuracy": decoded_correct / len(x),
    }
    if config.readout == "field":
        result["argmax_accuracy"] = native_correct / len(x)
    return result


def score(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    raw: torch.Tensor,
    train: int,
    config: Config,
    decoder_grid: torch.Tensor | None,
) -> dict[str, dict[str, float]]:
    """Evaluate both complete, disjoint orbit sets."""
    return {
        name: evaluate(
            model,
            x[part],
            y[part],
            raw[part],
            config,
            decoder_grid[part] if decoder_grid is not None else None,
        )
        for name, part in (("train", slice(None, train)), ("test", slice(train, None)))
    }


def run(
    config: Config, directory: Path, device: str, *, resume: bool = False
) -> dict[str, Any]:
    """Train with ScheduleFreeAdamW, preserving curves and resumable checkpoints."""
    directory.mkdir(parents=True, exist_ok=resume)
    raw_array, train, data_info = dataset(config)
    x, y = tensors(raw_array, config, device)
    raw = torch.as_tensor(raw_array, device=device)
    decoder_grid = (
        torch.as_tensor(future_grid(raw_array[:, :-1], config), device=device)
        if config.grid_domain or config.vocab_size
        else None
    )
    torch.manual_seed(config.seed)
    model = build(config, device)
    optimizer = ScheduleFreeAdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        warmup=config.warmup,
    )
    sampler = SplitMix64(config.seed ^ 0x6261746368657303)
    order = list(range(train))
    sampler.shuffle(order)
    criterion = Criterion()
    first_epoch, elapsed_before, stable99 = 0, 0.0, 0
    slices = batch_slices(train, config.batching, config.batch_size)
    metadata = {
        "config": asdict(config),
        "normalization": NORMALIZATION,
        "data": data_info,
        "optimizer": "ScheduleFreeAdamW",
        "parameters": sum(p.numel() for p in model.parameters()),
        "source_sha256": {
            name: hashlib.sha256(contents).hexdigest()
            for name, contents in SOURCE_SNAPSHOT.items()
        },
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "precision": "float32",
        "tf32": False,
        "updates_per_epoch": math.ceil(train / config.batch_size),
        "training_batch_sizes": [part.stop - part.start for part in slices],
        "micro_batches_per_update": [
            math.ceil(
                (part.stop - part.start)
                / (config.micro_batch_size or config.batch_size)
            )
            for part in slices
        ],
        "supervised_context_lengths": list(
            range(config.supervise_from or config.context, config.context + 1)
        ),
        "evaluation_context_length": config.context,
    }
    fields = [
        "epoch",
        "step",
        "seconds",
        "train_loss",
        "train_accuracy",
        "train_decoded_accuracy",
        "test_loss",
        "test_accuracy",
        "test_decoded_accuracy",
        "query_train_accuracy",
        "query_test_accuracy",
    ]
    if config.readout == "field":
        fields.extend(
            [
                "train_argmax_accuracy",
                "test_argmax_accuracy",
                "query_train_argmax_accuracy",
                "query_test_argmax_accuracy",
            ]
        )
    if config.restart_step:
        fields.extend(RATE_FIELDS)
    if resume:
        saved = load_checkpoint(directory / "checkpoint.pt", device)
        prior = asdict(Config(**saved["config"]))
        requested = asdict(config)
        prior.pop("epochs")
        requested.pop("epochs")
        if prior != requested:
            msg = "resume can extend epochs but cannot change the experiment"
            raise ValueError(msg)
        if saved["epoch"] >= config.epochs:
            msg = f"the run already reached epoch {saved['epoch']}; resume needs more"
            raise ValueError(msg)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.state = saved["sampler_state"]
        order = saved["sampler_order"]
        criterion = Criterion(**saved["criterion"])
        first_epoch = saved["epoch"] + 1
        elapsed_before = saved["seconds"]
        # Truncate observations beyond the last durable checkpoint, if interrupted.
        with (directory / "history.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            old_rows = list(reader)
        if reader.fieldnames != fields:
            msg = "history.csv has other columns; appending would misalign them"
            raise ValueError(msg)
        with (directory / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in old_rows if int(row["epoch"]) < first_epoch)
        for row in reversed(old_rows):
            if int(row["epoch"]) >= first_epoch:
                continue
            if (
                min(
                    float(row[key])
                    for key in ("test_accuracy", "test_decoded_accuracy")
                )
                < TARGET_ACCURACY
            ):
                break
            stable99 += 1
        if (directory / "result.json").exists():
            (directory / "result.json").replace(
                directory / f"result-before-resume-epoch{saved['epoch']}.json"
            )
        write_json(directory / "resume-metadata.json", metadata)
    else:
        write_json(directory / "metadata.json", metadata)
        (directory / "measured_source.py").write_bytes(
            SOURCE_SNAPSHOT["benchmarks/grokking/recurrence.py"]
        )
        (directory / "measured_field_sampler.py").write_bytes(
            SOURCE_SNAPSHOT["benchmarks/grokking/recurrence_field_sampler.py"]
        )
    start_time = time.monotonic()
    status = {}
    with (directory / "history.csv").open("a" if resume else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not resume:
            writer.writeheader()
        for epoch in range(first_epoch, config.epochs + 1):
            if epoch:
                if epoch > 1:
                    sampler.shuffle(order)
                model.train()
                optimizer.train()
                for part in slices:
                    indices = torch.tensor(order[part], device=device)
                    update_learning_rate(config, optimizer)
                    train_batch(
                        model, optimizer, x, y, indices, config.micro_batch_size
                    )
            model.eval()
            query = score(model, x, y, raw, train, config, decoder_grid)
            optimizer.eval()
            metrics = score(model, x, y, raw, train, config, decoder_grid)
            stable99 = (
                stable99 + 1
                if min(metrics["test"][key] for key in ("accuracy", "decoded_accuracy"))
                >= TARGET_ACCURACY
                else 0
            )
            criterion.observe(
                epoch, metrics["train"]["accuracy"], metrics["test"]["accuracy"]
            )
            elapsed = elapsed_before + time.monotonic() - start_time
            row = {
                "epoch": epoch,
                "step": epoch * math.ceil(train / config.batch_size),
                "seconds": elapsed,
                **{
                    f"{split}_{key}": value
                    for split, values in metrics.items()
                    for key, value in values.items()
                },
                **{
                    f"query_{split}_accuracy": values["accuracy"]
                    for split, values in query.items()
                },
                **{
                    f"query_{split}_argmax_accuracy": values["argmax_accuracy"]
                    for split, values in query.items()
                    if "argmax_accuracy" in values
                },
            }
            if config.restart_step:
                progress = optimizer_progress(optimizer)
                row.update({key: progress[key] for key in RATE_FIELDS})
            writer.writerow(row)
            handle.flush()
            status = {
                "config": asdict(config),
                "epoch": epoch,
                "seconds": elapsed,
                "metrics": metrics,
                "criterion": asdict(criterion),
                "consecutive_epochs_at_99": stable99,
            }
            if config.restart_step:
                status["optimizer_progress"] = progress
            write_json(directory / "status.json", status)
            done = stable99 >= config.after99 + 1
            if (
                epoch % 100 == 0
                or criterion.first99 == epoch
                or done
                or epoch == config.epochs
            ):
                print(
                    json.dumps(
                        {"run": config.name(), **row, "criterion": asdict(criterion)}
                    ),
                    flush=True,
                )
            if (
                epoch % CHECKPOINT_EVERY == 0
                or criterion.first99 == epoch
                or done
                or epoch == config.epochs
            ):
                temporary = directory / "checkpoint.tmp"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": asdict(config),
                        "normalization": NORMALIZATION,
                        "raw": torch.from_numpy(raw_array),
                        "train_rows": train,
                        "criterion": asdict(criterion),
                        "epoch": epoch,
                        "seconds": elapsed,
                        "metrics": metrics,
                        "query_metrics": query,
                        "sampler_state": sampler.state,
                        "sampler_order": order,
                        "consecutive_epochs_at_99": stable99,
                    },
                    temporary,
                )
                temporary.replace(directory / "checkpoint.pt")
            if done:
                break
    write_json(directory / "result.json", status)
    return status


@torch.no_grad()
def verify(directory: Path, device: str, *, full: bool = False) -> dict[str, Any]:
    """Reload both optimizer modes and check the public row-wise DiffGPT decoder."""
    saved = load_checkpoint(directory / "checkpoint.pt", device)
    config = Config(**saved["config"])
    model = build(config, device)
    model.load_state_dict(saved["model"])
    model.eval()
    raw_array = saved["raw"].cpu().numpy()
    x, y = tensors(raw_array, config, device)
    raw = torch.as_tensor(raw_array, device=device)
    decoder_grid = (
        torch.as_tensor(future_grid(raw_array[:, :-1], config), device=device)
        if config.grid_domain or config.vocab_size
        else None
    )
    train = saved["train_rows"]
    metrics = score(model, x, y, raw, train, config, decoder_grid)
    for split in ("train", "test"):
        for key, value in metrics[split].items():
            expected = saved["metrics"][split][key]
            rows = train if split == "train" else len(x) - train
            differs = (
                abs(value - expected) > REPLAY_LOSS_TOLERANCE
                if key == "loss"
                else round(value * rows) != round(expected * rows)
            )
            if differs:
                msg = f"checkpoint {split} {key} differs from the measured state"
                raise ValueError(msg)
    wrapper = DiffGPT(
        model, config.order, grid_domain(config), tokenizer=config.tokenizer
    )
    selected = raw_array[train:] if full else raw_array[train : train + 3]
    predictions = []
    for first in range(0, len(selected), 256):
        batch = selected[first : first + 256]
        readout = (
            CanonicalFieldSampler(
                torch.as_tensor(batch[:, :-1], device=device),
                config.order,
                grid_vocabulary(config),
                P,
                lower=int(config.task == "division"),
            )
            if config.readout == "field"
            else TemperatureSampler(temperature=0.0)
        )
        predictions.extend(
            wrapper.predict_many(
                [pd.DataFrame({"x": row[:-1].astype(float)}) for row in batch],
                max_new_points=1,
                sampler=readout,
                batch_size=len(batch),
            )
        )
    predicted = np.rint([frame.iloc[-1, 0] for frame in predictions]).astype(int)
    public_accuracy = float(np.mean(predicted == selected[:, -1]))
    public_correct = int(np.count_nonzero(predicted == selected[:, -1]))
    expected_correct = round(metrics["test"]["decoded_accuracy"] * (len(x) - train))
    if full and public_correct != expected_correct:
        msg = (
            f"public DiffGPT correct count {public_correct} differs from "
            f"the full decoded test count {expected_correct}"
        )
        raise ValueError(msg)
    optimizer = ScheduleFreeAdamW(model.parameters())
    optimizer.load_state_dict(saved["optimizer"])
    optimizer.train()
    query = score(model, x, y, raw, train, config, decoder_grid)
    for split in ("train", "test"):
        rows = train if split == "train" else len(x) - train
        if round(query[split]["accuracy"] * rows) != round(
            saved["query_metrics"][split]["accuracy"] * rows
        ):
            msg = "saved query weights do not replay their measured accuracy"
            raise ValueError(msg)
    result = {
        "epoch": saved["epoch"],
        "readout": config.readout,
        "metrics": metrics,
        "query_metrics": query,
        "public_rows": len(selected),
        "public_accuracy": public_accuracy,
        "public_correct": public_correct,
    }
    write_json(directory / "verification.json", result)
    return result


def main() -> None:
    """Run a fixed-optimizer plan, extend a checkpoint, or verify saved weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/grokking-div97-rows"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threads", type=int, default=2)
    for field, default in asdict(Config()).items():
        parser.add_argument(
            "--" + field.replace("_", "-"), type=type(default), default=default
        )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if args.verify:
        print(json.dumps(verify(args.verify, args.device, full=args.full), indent=2))
        return
    configs = (
        [Config(**item) for item in json.loads(args.plan.read_text())]
        if args.plan
        else [Config(**{key: getattr(args, key) for key in asdict(Config())})]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for config in configs:
        directory = args.output / config.name()
        result = directory / "result.json"
        # A completed run resumes only to extend it to more epochs.
        if result.exists() and (
            not args.resume or json.loads(result.read_text())["epoch"] >= config.epochs
        ):
            print(f"Completed: {directory}", flush=True)
            continue
        run(config, directory, args.device, resume=args.resume)
        verify(directory, args.device, full=args.full)


if __name__ == "__main__":
    main()
