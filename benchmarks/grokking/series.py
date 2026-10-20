"""
Full-split series grokking runner using the unchanged GPT/ScheduleFreeAdamW.

Run ``.venv/bin/python -m benchmarks.grokking.series --help``. A JSON plan is
a list of Config dictionaries; dataset choices live under ``data``. Every epoch
scores both optimizer states. Checkpoints contain the codec/split fingerprints,
source hashes, sampling state, RNG state and the architecture marker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.grokking.div97 import (
    LIBRARY_SOURCES,
    NORMALIZATION,
    ROOT,
    TARGET_ACCURACY,
    Criterion,
    SplitMix64,
    load_checkpoint,
    write_json,
)
from benchmarks.grokking.recurrence import batch_slices, optimizer_progress
from benchmarks.grokking.recurrence_field_sampler import CanonicalFieldSampler
from benchmarks.grokking.series_data import DataConfig, dataset
from diff_gpt.model.gpt import GPT
from diff_gpt.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from collections.abc import Mapping

    from benchmarks.grokking.series_data import Dataset, Split

FORMAT_VERSION = 2
SHORT_CONTEXT_LIMIT = 4
SHORT_CONTEXT_EVALUATION_BATCH = 4096
METRICS = (
    "loss",
    "accuracy",
    "clean_accuracy",
    "argmax_accuracy",
    "mse",
    "mae",
    "skill",
    "bayes_loss",
)


@dataclass(frozen=True)
class Config:
    """One independent run; data, model and optimizer are all explicit."""

    data: DataConfig = field(default_factory=DataConfig)
    width: int = 128
    layers: int = 1
    heads: int = 4
    learning_rate: float = 0.01
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    warmup: int = 10
    batch_size: int = 512
    micro_batch_size: int = 0
    batching: str = "tail"
    epochs: int = 1000
    after99: int = 25
    rope_base: float = 10000.0
    rope_layers: tuple[bool, ...] | None = None
    restart_step: int = 0
    restart_warmup: int = 0
    checkpoint_every: int = 100
    early_stopping_patience: int = 25

    def name(self) -> str:
        """Include a hash of all scientific settings, excluding epoch budget."""
        scientific = asdict(self)
        scientific.pop("epochs")
        key = hashlib.sha256(
            json.dumps(scientific, sort_keys=True).encode()
        ).hexdigest()[:10]
        return f"{self.data.dataset}-k{self.data.order}-s{self.data.seed}-{key}"


def config_from_dict(value: Mapping[str, Any]) -> Config:
    """Restore nested dataclasses and JSON lists without implicit defaults."""
    values = dict(value)
    values["data"] = DataConfig(**values.get("data", {}))
    if values.get("rope_layers") is not None:
        values["rope_layers"] = tuple(values["rope_layers"])
    return Config(**values)


def source_hashes() -> dict[str, str]:
    """Record all library and benchmark code this runner executes."""
    paths = (
        *LIBRARY_SOURCES,
        Path(__file__),
        Path(__file__).with_name("series_data.py"),
        Path(__file__).with_name("series_oracle.py"),
        Path(__file__).with_name("series_gaussian.py"),
        Path(__file__).with_name("div97.py"),
        Path(__file__).with_name("recurrence.py"),
        Path(__file__).with_name("recurrence_field_sampler.py"),
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def skill(loss: float, reference: float, floor: float | None) -> float | None:
    """Keep the raw skill, including negative and below-floor observations."""
    if floor is None or reference <= floor + 1e-12:
        return None
    return (reference - loss) / (reference - floor)


def _tensor(split: Split, device: str) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value.copy(), device=device)
        for key in split.__dataclass_fields__
        if (value := getattr(split, key)) is not None
    }


def build(config: Config, context: int, device: str) -> GPT:
    """Pass only existing GPT parameters; no series logic enters the model."""
    return GPT(
        vocab_size=config.data.vocab_size,
        n_embd=config.width,
        n_head=config.heads,
        n_layer=config.layers,
        block_size=context + 1,
        use_checkpoint=False,
        rope_base=config.rope_base,
        rope_layers=config.rope_layers,
    ).to(device)


@torch.no_grad()
def evaluate(
    model: GPT,
    tensors: dict[str, torch.Tensor],
    data: Dataset,
    name: str,
    config: Config,
    evaluation_batch: int = 256,
) -> dict[str, Any]:
    """Score all frozen examples, per column, with proper and point losses."""
    features = len(data.columns)
    totals = torch.zeros((features, 8), dtype=torch.float64, device=model.device)
    seasonal_totals = torch.zeros(
        (features, 2), dtype=torch.float64, device=model.device
    )
    for first in range(0, len(tensors["x"]), evaluation_batch):
        stop = first + evaluation_batch
        logits = model(tensors["x"][first:stop])[0][:, -features:]
        target = tensors["y"][first:stop, -features:]
        clean = tensors["clean_y"][first:stop, -features:]
        native = logits.argmax(-1)
        predicted = native
        if config.data.dataset == "fibonacci":
            predicted = CanonicalFieldSampler(
                tensors["past"][first:stop],
                config.data.order,
                config.data.vocab_size,
                97,
            )(logits[:, 0])
        log_probability = F.log_softmax(logits, dim=-1)
        loss = -log_probability.gather(-1, target[..., None])[..., 0]
        probability = log_probability.exp()
        grid = (
            torch.arange(
                config.data.vocab_size, device=model.device, dtype=probability.dtype
            )
            - config.data.vocab_size // 2
        )
        expected = torch.sum(probability * grid, dim=-1)
        value = tensors["base"][first:stop] + tensors["step"][first:stop] * expected
        error = value - tensors["truth"][first:stop]
        bayes_loss = torch.zeros_like(loss)
        if "bayes" in tensors:
            bayes_loss = -(tensors["bayes"][first:stop] * log_probability).sum(-1)
        values = (
            loss,
            predicted == target,
            predicted == clean,
            native == target,
            error.square(),
            error.abs(),
            bayes_loss,
            torch.ones_like(loss),
        )
        totals += torch.stack(
            [value.to(torch.float64).sum(0) for value in values], dim=-1
        )
        if "seasonal" in tensors:
            residual = tensors["seasonal"][first:stop] - tensors["truth"][first:stop]
            seasonal_totals += torch.stack(
                (residual.square().sum(0), residual.abs().sum(0)), dim=-1
            )
    result = {}
    for index, column in enumerate(data.columns):
        sums = totals[index].tolist()
        size = sums[-1]
        metrics = {
            key: sums[pos] / size
            for pos, key in enumerate(
                (
                    "loss",
                    "accuracy",
                    "clean_accuracy",
                    "argmax_accuracy",
                    "mse",
                    "mae",
                    "bayes_loss",
                )
            )
        }
        if "bayes" not in tensors:
            metrics["bayes_loss"] = None
        reference = data.references[name][column]
        skill_loss = (
            metrics["bayes_loss"]
            if "bayes" in tensors and name != "train"
            else metrics["loss"]
        )
        metrics["skill"] = (
            None
            if reference.get("input_independent_control")
            else skill(skill_loss, reference["baseline_loss"], reference["floor_loss"])
        )
        if "seasonal" in tensors:
            metrics.update(
                zip(
                    ("seasonal_mse", "seasonal_mae"),
                    (value / size for value in seasonal_totals[index].tolist()),
                    strict=True,
                )
            )
        result[column] = metrics
    aggregate = {
        key: float(np.mean([result[column][key] for column in data.columns]))
        if all(result[column][key] is not None for column in data.columns)
        else None
        for key in METRICS
    }
    aggregate["columns"] = result
    return aggregate


def score(
    model: GPT,
    tensors: dict[str, dict[str, torch.Tensor]],
    data: Dataset,
    config: Config,
) -> dict[str, Any]:
    """Evaluate each complete split, including validation when available."""
    evaluation_batch = max(1, min(256, config.micro_batch_size or 256))
    if (
        not config.micro_batch_size
        and tensors["train"]["x"].shape[1] <= SHORT_CONTEXT_LIMIT
    ):
        evaluation_batch = SHORT_CONTEXT_EVALUATION_BATCH
    return {
        name: evaluate(
            model,
            values,
            data,
            name,
            config,
            evaluation_batch=evaluation_batch,
        )
        for name, values in tensors.items()
    }


@torch.no_grad()
def forecast_one_row(
    model: GPT,
    tensors: dict[str, torch.Tensor],
    columns: list[str],
    vocab: int,
) -> dict[str, Any]:
    """Generate a whole next row without observing any of its true columns."""
    features = len(columns)
    totals = torch.zeros((features, 4), dtype=torch.float64, device=model.device)
    for first in range(0, len(tensors["x"]), 64):
        stop = first + 64
        prefix_length = tensors["x"].shape[1] - features + 1
        prefix = tensors["x"][first:stop, :prefix_length]
        generated = []
        for _ in columns:
            following = model(prefix)[0][:, -1].argmax(-1, keepdim=True)
            generated.append(following)
            prefix = torch.cat((prefix, following), dim=1)
        tokens = torch.cat(generated, dim=1)
        prediction = tensors["base"][first:stop] + tensors["step"][first:stop] * (
            tokens - vocab // 2
        )
        error = prediction - tensors["truth"][first:stop]
        totals += torch.stack(
            (
                error.square().sum(0),
                error.abs().sum(0),
                (tokens == tensors["clean_y"][first:stop, -features:]).sum(0),
                torch.full((features,), len(tokens), device=model.device),
            ),
            dim=-1,
        )
    measured = {
        column: {
            "mse": values[0] / values[3],
            "mae": values[1] / values[3],
            "accuracy": values[2] / values[3],
        }
        for column, values in zip(columns, totals.tolist(), strict=True)
    }
    return {
        "horizon_rows": 1,
        "readout": "autoregressive token argmax, decoded on context grid",
        "teacher_forced_future_columns": False,
        "columns": measured,
        "mse": float(np.mean([value["mse"] for value in measured.values()])),
        "mae": float(np.mean([value["mae"] for value in measured.values()])),
    }


def _validate(config: Config) -> None:
    if config.epochs < 0 or config.checkpoint_every < 1 or config.after99 < 0:
        msg = "epochs/after99 must be nonnegative and checkpoint_every positive"
        raise ValueError(msg)
    if (
        config.restart_step < 0
        or config.restart_warmup < 0
        or (config.restart_warmup and not config.restart_step)
    ):
        msg = "restart warmup requires a positive optimizer restart step"
        raise ValueError(msg)


def _flatten(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    values = {}
    for split, measured in metrics.items():
        for key in METRICS:
            values[f"{prefix}{split}_{key}"] = measured[key]
        for column, column_values in measured["columns"].items():
            for key in METRICS:
                values[f"{prefix}{split}:{column}:{key}"] = column_values[key]
    return values


def _restore_history(directory: Path, epoch: int, fields: list[str]) -> None:
    with (directory / "history.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames != fields or [
            int(row["epoch"]) for row in rows[: epoch + 1]
        ] != list(range(epoch + 1)):
            msg = "history and durable checkpoint disagree"
            raise ValueError(msg)
    temporary = directory / "history.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if int(row["epoch"]) <= epoch)
    temporary.replace(directory / "history.csv")


def run(
    config: Config, directory: Path, device: str, *, resume: bool = False
) -> dict[str, Any]:
    """Train through the late phase, without restoring minimum-validation weights."""
    _validate(config)
    data = dataset(config.data)
    if not all(value["covered"] for value in data.metadata["token_coverage"].values()):
        msg = (
            "evaluation contains unseen tokens; increase training coverage "
            "or coarsen the codec"
        )
        raise ValueError(msg)
    tensors = {name: _tensor(split, device) for name, split in data.splits.items()}
    train = tensors["train"]
    torch.manual_seed(config.data.seed)
    model = build(config, train["x"].shape[1], device)
    optimizer = ScheduleFreeAdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        warmup=config.warmup,
    )
    sampler = SplitMix64(config.data.seed ^ 0x6261746368657303)
    order = list(range(len(train["x"])))
    sampler.shuffle(order)
    criteria = {
        mode: {column: Criterion() for column in data.columns}
        for mode in ("averaged", "query")
    }
    slices = batch_slices(len(order), config.batching, config.batch_size)
    flops_per_epoch = model.estimate_flops() * len(order) * train["x"].shape[1]
    hashes = source_hashes()
    metadata = {
        "format_version": FORMAT_VERSION,
        "config": asdict(config),
        "normalization": NORMALIZATION,
        "source_sha256": hashes,
        "data": data.metadata,
        "references": data.references,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_flops_per_epoch": flops_per_epoch,
        "flops_definition": (
            "GPT.estimate_flops times actual input tokens; evaluation excluded"
        ),
        "device": device,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "precision": "float32",
        "tf32": False,
    }
    before, first_epoch, stable = 0.0, 0, 0
    selector = {
        "best_validation_loss": None,
        "best_epoch": None,
        "best_metrics": None,
        "early_stop_epoch": None,
        "early_stop_metrics": None,
        "early_stop_selected_epoch": None,
    }
    saved = None
    if resume:
        saved = load_checkpoint(directory / "checkpoint.pt", device)
        prior, requested = asdict(config_from_dict(saved["config"])), asdict(config)
        prior.pop("epochs")
        requested.pop("epochs")
        if (
            prior != requested
            or saved["format_version"] != FORMAT_VERSION
            or saved["data_hashes"] != data.fingerprint()
            or saved["source_sha256"] != hashes
        ):
            msg = (
                "resume requires identical configuration, source, codec and frozen "
                "splits (only epochs may extend)"
            )
            raise ValueError(msg)
        if saved["epoch"] >= config.epochs:
            msg = "resume must extend the saved epoch budget"
            raise ValueError(msg)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.state, order = saved["sampler_state"], saved["sampler_order"]
        criteria = {
            mode: {column: Criterion(**value) for column, value in columns.items()}
            for mode, columns in saved["criteria"].items()
        }
        first_epoch, before, stable = (
            saved["epoch"] + 1,
            saved["seconds"],
            saved["stable"],
        )
        selector = saved["validation_selection"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "metadata.json", metadata)
        # Save exact input arrays and source code for auditing without data reload.
        arrays = {
            f"{name}__{key}": value
            for name, split in data.splits.items()
            for key in split.__dataclass_fields__
            if (value := getattr(split, key)) is not None
        }
        np.savez_compressed(directory / "frozen_splits.npz", **arrays)
        for relative in hashes:
            path = directory / "source" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
    # Build the CSV schema without another optimizer mode switch on resume.
    template = {
        name: {
            **dict.fromkeys(METRICS),
            "columns": {column: dict.fromkeys(METRICS) for column in data.columns},
        }
        for name in tensors
    }
    fields = [
        "epoch",
        "step",
        "seconds",
        "training_flops",
        "base_rate",
        "averaging_coefficient",
        *_flatten(template),
        *_flatten(template, "query_"),
    ]
    if resume:
        _restore_history(directory, saved["epoch"], fields)
        write_json(directory / "resume-metadata.json", metadata)
    started = time.monotonic()
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
                    optimizer.zero_grad(set_to_none=True)
                    indices = order[part]
                    micro = config.micro_batch_size or len(indices)
                    for start in range(0, len(indices), micro):
                        batch = torch.tensor(
                            indices[start : start + micro], device=device
                        )
                        _, loss = model(train["x"][batch], train["y"][batch])
                        (loss * (len(batch) / len(indices))).backward()
                    if (
                        config.restart_step
                        and optimizer.param_groups[0]["step"] >= config.restart_step
                    ):
                        local = (
                            optimizer.param_groups[0]["step"] - config.restart_step + 1
                        )
                        ramp = (
                            min(1.0, local / config.restart_warmup)
                            if config.restart_warmup
                            else 1.0
                        )
                        optimizer.param_groups[0]["lr"] = config.learning_rate * ramp
                    optimizer.step()
            model.eval()
            query = score(model, tensors, data, config)
            query_model = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            optimizer.eval()
            averaged = score(model, tensors, data, config)
            if not all(
                math.isfinite(measured["loss"])
                for measured in (*query.values(), *averaged.values())
            ):
                msg = f"nonfinite evaluation at epoch {epoch}"
                raise ValueError(msg)
            for mode, measured in (("averaged", averaged), ("query", query)):
                for column in data.columns:
                    key = (
                        "accuracy"
                        if data.metadata.get("accuracy_criterion")
                        else "skill"
                    )
                    train_value = measured["train"]["columns"][column][key]
                    test_value = measured["test"]["columns"][column][key]
                    if train_value is not None and test_value is not None:
                        criteria[mode][column].observe(
                            epoch,
                            min(1.0, max(0.0, train_value)),
                            min(1.0, max(0.0, test_value)),
                        )
            selected = averaged.get("validation")
            improved = selected is not None and (
                selector["best_validation_loss"] is None
                or selected["loss"] < selector["best_validation_loss"]
            )
            if improved:
                selector.update(
                    {
                        "best_validation_loss": selected["loss"],
                        "best_epoch": epoch,
                        "best_metrics": averaged,
                    }
                )
            if (
                selected is not None
                and selector["early_stop_epoch"] is None
                and epoch - selector["best_epoch"] >= config.early_stopping_patience
            ):
                selector.update(
                    {
                        "early_stop_epoch": epoch,
                        "early_stop_metrics": selector["best_metrics"],
                        "early_stop_selected_epoch": selector["best_epoch"],
                    }
                )
                shutil.copyfile(
                    directory / "best-validation.pt", directory / "early-stopping.pt"
                )
            stable = (
                stable + 1
                if all(
                    value.first99 is not None for value in criteria["averaged"].values()
                )
                and all(
                    (
                        averaged["test"]["columns"][column]["accuracy"]
                        if data.metadata.get("accuracy_criterion")
                        else averaged["test"]["columns"][column]["skill"] or 0.0
                    )
                    >= TARGET_ACCURACY
                    for column in data.columns
                )
                else 0
            )
            elapsed = before + time.monotonic() - started
            progress = optimizer_progress(optimizer)
            row = {
                "epoch": epoch,
                "step": progress["step"],
                "seconds": elapsed,
                "training_flops": epoch * flops_per_epoch,
                "base_rate": progress["base_rate"],
                "averaging_coefficient": progress["averaging_coefficient"],
                **_flatten(averaged),
                **_flatten(query, "query_"),
            }
            writer.writerow(row)
            handle.flush()
            status = {
                "config": asdict(config),
                "epoch": epoch,
                "seconds": elapsed,
                "training_flops": row["training_flops"],
                "metrics": averaged,
                "query_metrics": query,
                "criteria": {
                    mode: {column: asdict(value) for column, value in columns.items()}
                    for mode, columns in criteria.items()
                },
                "validation_selection": selector,
            }
            write_json(directory / "status.json", status)
            done = stable >= config.after99 + 1
            crossing = any(
                value.first99 == epoch
                for columns in criteria.values()
                for value in columns.values()
            )
            checkpoint = {
                "format_version": FORMAT_VERSION,
                "model": model.state_dict(),
                "query_model": query_model,
                "optimizer": optimizer.state_dict(),
                "normalization": NORMALIZATION,
                "config": asdict(config),
                "epoch": epoch,
                "seconds": elapsed,
                "criteria": status["criteria"],
                "metrics": averaged,
                "query_metrics": query,
                "sampler_state": sampler.state,
                "sampler_order": order,
                "stable": stable,
                "data_hashes": data.fingerprint(),
                "source_sha256": hashes,
                "validation_selection": selector,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            }
            if improved:
                temporary = directory / "best-validation.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(directory / "best-validation.pt")
            if (
                epoch % config.checkpoint_every == 0
                or crossing
                or done
                or epoch == config.epochs
            ):
                temporary = directory / "checkpoint.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(directory / "checkpoint.pt")
            if epoch % 100 == 0 or crossing or done or epoch == config.epochs:
                print(
                    json.dumps(
                        {
                            "run": config.name(),
                            "epoch": epoch,
                            "seconds": elapsed,
                            "train_loss": averaged["train"]["loss"],
                            "test_loss": averaged["test"]["loss"],
                            "test_accuracy": averaged["test"]["accuracy"],
                            "test_skill": averaged["test"]["skill"],
                        }
                    ),
                    flush=True,
                )
            if done:
                break
    if config.data.dataset in {"weather", "etth1", "m4_hourly", "ar", "seasonal_ar"}:
        status["forecast"] = {
            "final": forecast_one_row(
                model, tensors["test"], data.columns, config.data.vocab_size
            )
        }
        if (directory / "early-stopping.pt").exists():
            early = load_checkpoint(directory / "early-stopping.pt", device)
            model.load_state_dict(early["model"])
            status["forecast"]["early_stopping"] = forecast_one_row(
                model, tensors["test"], data.columns, config.data.vocab_size
            )
            status["forecast"]["early_stopping"]["selected_epoch"] = early["epoch"]
        status["forecast"]["comparison"] = (
            "selectors share the same training trajectory and FLOP budget; "
            "the early-stop checkpoint freezes at the first patience crossing"
        )
    write_json(directory / "result.json", status)
    return status


@torch.no_grad()
def verify(directory: Path, device: str) -> dict[str, Any]:
    """Replay complete splits in both saved optimizer states."""
    saved = load_checkpoint(directory / "checkpoint.pt", device)
    config = config_from_dict(saved["config"])
    data = dataset(config.data)
    if (
        data.fingerprint() != saved["data_hashes"]
        or source_hashes() != saved["source_sha256"]
    ):
        msg = "checkpoint data or source fingerprint changed"
        raise ValueError(msg)
    tensors = {name: _tensor(split, device) for name, split in data.splits.items()}
    model = build(config, tensors["train"]["x"].shape[1], device)
    model.load_state_dict(saved["model"])
    model.eval()
    optimizer = ScheduleFreeAdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(saved["optimizer"])
    averaged = score(model, tensors, data, config)
    optimizer.train()
    if saved.get("format_version") != FORMAT_VERSION or "query_model" not in saved:
        msg = "checkpoint needs the explicit scored query weights"
        raise ValueError(msg)
    for name, value in model.state_dict().items():
        if not torch.allclose(value, saved["query_model"][name], rtol=1e-5, atol=1e-6):
            msg = "packed optimizer state disagrees with the saved query weights"
            raise ValueError(msg)
    model.load_state_dict(saved["query_model"])
    query = score(model, tensors, data, config)
    for actual, expected in (
        (averaged, saved["metrics"]),
        (query, saved["query_metrics"]),
    ):
        for key, value in _flatten(actual).items():
            previous = _flatten(expected)[key]
            if value is not None and not math.isclose(
                value, previous, rel_tol=1e-5, abs_tol=2e-5
            ):
                msg = f"checkpoint replay mismatch: {key}"
                raise ValueError(msg)
    result = {
        "verified": True,
        "epoch": saved["epoch"],
        "normalization": NORMALIZATION,
        "split_hashes": data.fingerprint(),
        "both_weight_states": True,
        "explicit_scored_query_weights": True,
    }
    write_json(directory / "verification.json", result)
    return result


def main() -> None:
    """Run a frozen plan, prepare datasets, resume, or replay saved states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/grokking-series/experiments"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.verify:
        print(json.dumps(verify(args.verify, args.device), indent=2))
        return
    configs = (
        [config_from_dict(value) for value in json.loads(args.plan.read_text())]
        if args.plan
        else [Config()]
    )
    for config in configs:
        directory = args.output / config.name()
        if args.prepare:
            data = dataset(config.data)
            print(
                json.dumps(
                    {
                        "run": config.name(),
                        "data": data.metadata,
                        "references": data.references,
                    }
                )
            )
            continue
        if (directory / "result.json").exists() and (
            not args.resume
            or json.loads((directory / "result.json").read_text())["epoch"]
            >= config.epochs
        ):
            print(f"Completed: {directory}", flush=True)
            continue
        run(config, directory, args.device, resume=args.resume and directory.exists())
        verify(directory, args.device)


if __name__ == "__main__":
    main()
