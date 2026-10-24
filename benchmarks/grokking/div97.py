"""
Full-split grokking experiments with the real DiffGPT at derivative order k >= 0.

Run with ``uv run --locked -m benchmarks.grokking.div97 --help``. Each independent
equation is a two-token context [a, b]; only its next token q is supervised.
For ``--order`` k > 0, the codec takes k-th differences after k fixed zeros.
The website's SplitMix64 split and epoch sampler are reproduced exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import ENCODER_VERSION, decode, encode
from diff_gpt.model.gpt import GPT
from diff_gpt.optimizer.schedule_free import ScheduleFreeAdamW
from diff_gpt.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from collections.abc import Iterable

P = 97
TRAIN = P * (P - 1) // 2
TOTAL = 2 * TRAIN
BATCH = 512
UPDATES_PER_EPOCH = math.ceil(TRAIN / BATCH)
MASK64 = (1 << 64) - 1
DOMAIN = np.array([96.0], dtype=np.float64)
VOCAB = 256  # k=0 must distinguish all 97 nonnegative residues, without clipping.
BLOCK = 3  # Two prompt tokens and room for DiffGPT.predict's one output token.
LOW_ACCURACY = 0.1
RISE_ACCURACY = 0.9
TARGET_ACCURACY = 0.99
MIN_PLATEAU = 25
MAX_RISE = 250
REPLAY_LOSS_TOLERANCE = 2e-5
ROOT = Path(__file__).resolve().parents[2]
# The library files a training run executes; metadata records their hashes.
LIBRARY_SOURCES = (
    ROOT / "src/diff_gpt/encoder_decoder.py",
    ROOT / "src/diff_gpt/model/gpt.py",
    ROOT / "src/diff_gpt/model/block.py",
    ROOT / "src/diff_gpt/model/mha.py",
    ROOT / "src/diff_gpt/model/feed_forward.py",
    ROOT / "src/diff_gpt/model/rope.py",
    ROOT / "src/diff_gpt/optimizer/schedule_free.py",
)
# Peri-RMSNorm and Q/K RMSNorm have no parameters, so weights of an earlier model
# load without error and compute another function; saved runs carry this marker.
NORMALIZATION = "peri_rms_qk_rms"


class SplitMix64:
    """The browser's 64-bit random stream, including rejection sampling."""

    def __init__(self, seed: int) -> None:
        """Start the stream at an unsigned 64-bit seed."""
        self.state = seed & MASK64

    def next(self) -> int:
        """Return the next unsigned 64-bit integer."""
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def shuffle(self, values: list) -> None:
        """Apply the website's Fisher-Yates permutation in place."""
        for end in range(len(values) - 1, 0, -1):
            bound = end + 1
            threshold = (1 << 64) % bound
            value = self.next()
            while value < threshold:
                value = self.next()
            index = value % bound
            values[end], values[index] = values[index], values[end]


def division_rows(seed: int) -> np.ndarray:
    """Return all 9,312 equations, shuffled exactly as in the website."""
    rows = [(a, b, a * pow(b, -1, P) % P) for a in range(P) for b in range(1, P)]
    SplitMix64(seed ^ 0x73706C6974730002).shuffle(rows)
    return np.array(rows, dtype=np.int64)


class EpochSampler:
    """Visit each training equation once per epoch, including the 48-row tail."""

    def __init__(self, seed: int) -> None:
        """Match the browser's initial order and stream."""
        self.random = SplitMix64(seed ^ 0x6261746368657303)
        self.order = list(range(TRAIN))
        self.random.shuffle(self.order)
        self.epochs = 0

    def batches(self) -> Iterable[list[int]]:
        """Yield one complete epoch without crossing into the next."""
        if self.epochs:
            self.random.shuffle(self.order)
        self.epochs += 1
        for start in range(0, TRAIN, BATCH):
            yield self.order[start : start + BATCH]


@dataclass
class Criterion:
    """The website's ShortGrokking rule, evaluated at every complete epoch."""

    plateau_start: int | None = None
    plateau_epochs: int = 0
    last_low: int | None = None
    first90: int | None = None
    first99: int | None = None
    rise_epochs: int | None = None
    delayed: bool = False

    def observe(self, epoch: int, train: float, test: float) -> None:
        """Freeze the first 90% crossing and classify at the first 99% one."""
        if self.first90 is None:
            low = test <= LOW_ACCURACY
            if low and train >= TARGET_ACCURACY:
                if self.plateau_start is None:
                    self.plateau_start = epoch
                self.plateau_epochs = max(
                    self.plateau_epochs,
                    epoch - self.plateau_start,
                )
            else:
                self.plateau_start = None
            if low:
                self.last_low = epoch
            if test >= RISE_ACCURACY:
                self.first90 = epoch
                if self.last_low is not None:
                    self.rise_epochs = epoch - self.last_low
        if self.first99 is None and test >= TARGET_ACCURACY:
            self.first99 = epoch
            self.delayed = (
                self.plateau_epochs >= MIN_PLATEAU
                and self.rise_epochs is not None
                and self.rise_epochs <= MAX_RISE
            )


@dataclass(frozen=True)
class RunConfig:
    """One run; model, initialization and optimizer use repository defaults."""

    width: int = 64
    layers: int = 1
    heads: int = 4
    seed: int = 11
    optimizer: str = "schedule-free"
    learning_rate: float = 0.01
    weight_decay: float = 1.0
    beta2: float = 0.98
    warmup: int = 10
    epochs: int = 1000
    after99: int = 25
    labels: str = "division"
    order: int = 0

    def name(self) -> str:
        """Return a readable directory name, without warmup, epochs and after99."""
        return (
            f"{self.labels}{f'-k{self.order}' if self.order else ''}"
            f"-w{self.width}-l{self.layers}-h{self.heads}"
            f"-{self.optimizer}-lr{self.learning_rate:g}-wd{self.weight_decay:g}"
            f"-b2{self.beta2:g}-s{self.seed}"
        )


def vocabulary(order: int = 0) -> int:
    """Leave room for integer differences and their rounding headroom."""
    return VOCAB << max(0, order - 1)


def domain(order: int = 0) -> np.ndarray:
    """Use the original k=0 grid, and an exact unit grid for positive orders."""
    if not order:
        return DOMAIN.copy()
    return np.array([(vocabulary(order) - 2**order) / 2], dtype=np.float64)


def token_table(order: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Check first tokens after a fixed zero prefix with the actual codec."""
    if order:
        values = np.vstack((np.zeros((order, P)), np.arange(P)))
        domains = np.repeat(domain(order), P)
    else:
        values = np.arange(P, dtype=np.float64).reshape(-1, 1)
        domains = domain(order)
    vocab = vocabulary(order)
    start, grid, tokens = encode(values, vocab, domains, order, tokenizer="lgt")
    decoded = decode(start, grid, tokens, vocab, order, tokenizer="lgt")
    error = float(np.max(np.abs(decoded - values)))
    if np.unique(tokens).size != P or not np.array_equal(np.rint(decoded), values):
        msg = "tokenization must retain all 97 residue classes"
        raise ValueError(msg)
    table = tokens.reshape(-1)
    return table, {
        "order_of_derivative": order,
        "tokenizer": "lgt",
        "encoder_version": ENCODER_VERSION,
        "vocab_size": vocab,
        "domain": domain(order).tolist(),
        "scale": grid.scale[:1].tolist(),
        "residue_tokens": table.tolist(),
        "fixed_initial_prefix": [0] * order,
        "max_decoding_error": error,
        "rounded_decoding_exact": True,
    }


def make_tensors(
    rows: np.ndarray,
    lookup: np.ndarray,
    device: str,
    order: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give GPT only [a,b]; mask the unpredictable a->b target with -100."""
    offset = vocabulary(order) // 2
    levels = lookup[rows] - offset
    a, b, q = levels.T
    encoded = (
        np.stack(
            (a, b - order * a, q - order * b + order * (order - 1) // 2 * a), axis=1
        )
        + offset
    )
    if encoded.min() < 0 or encoded.max() >= vocabulary(order):
        msg = "difference tokens exceed the vocabulary"
        raise ValueError(msg)
    inputs = torch.as_tensor(encoded[:, :2].copy(), device=device)
    targets = torch.full_like(inputs, -100)
    targets[:, -1] = torch.as_tensor(encoded[:, 2], device=device)
    return inputs, targets


@torch.no_grad()
def evaluate(
    model: GPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 2048,
    *,
    order: int,
    answers: torch.Tensor,
) -> dict[str, float]:
    """Score every equation with unrestricted argmax over the complete vocab."""
    totals = torch.zeros(3 if order else 2, device=inputs.device, dtype=torch.float64)
    for x, y, raw in zip(
        inputs.split(batch_size),
        targets.split(batch_size),
        answers.split(batch_size),
        strict=True,
    ):
        logits, _ = model(x)
        logits = logits[:, -1]
        truth = y[:, -1]
        totals[0] += F.cross_entropy(logits, truth, reduction="sum")
        predicted = logits.argmax(-1)
        totals[1] += (predicted == truth).sum()
        if order:
            # Positive orders use the exact unit grid. These are the actual
            # inverse finite differences, before comparing decoded integers.
            offset = vocabulary(order) // 2
            a = x[:, 0] - offset
            b = x[:, 1] - offset + order * a
            correction = order * b - order * (order - 1) // 2 * a - offset
            totals[2] += (predicted + correction == raw).sum()
    values = (totals / len(inputs)).tolist()
    result = {"loss": values[0], "accuracy": values[1]}
    if order:
        result["decoded_accuracy"] = values[2]
    return result


def evaluate_splits(
    model: GPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    answers: torch.Tensor,
    order: int,
) -> dict[str, dict[str, float]]:
    """Compare both complete splits to their original arithmetic answers."""
    return {
        name: evaluate(
            model, inputs[part], targets[part], order=order, answers=answers[part]
        )
        for name, part in (("train", slice(None, TRAIN)), ("test", slice(TRAIN, None)))
    }


def write_json(path: Path, value: object) -> None:
    """Replace a JSON file atomically, so a reader never sees half of it."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(array: np.ndarray) -> str:
    """Return the SHA-256 of an integer array as little-endian int64."""
    return hashlib.sha256(array.astype("<i8", copy=False).tobytes()).hexdigest()


def require_current_model(record: dict[str, Any]) -> None:
    """Reject a run record saved by a model before Peri-RMSNorm and Q/K RMSNorm."""
    if record.get("normalization") != NORMALIZATION:
        msg = (
            "checkpoint architecture differs from the current Peri/QK-RMSNorm model; "
            "replay it with the source its metadata records"
        )
        raise ValueError(msg)


def load_checkpoint(source: Path | BinaryIO, device: str) -> dict[str, Any]:
    """Load a checkpoint, rejecting weights that an earlier model trained."""
    checkpoint = torch.load(source, map_location=device, weights_only=True)
    require_current_model(checkpoint)
    return checkpoint


def _metadata(config: RunConfig, rows: np.ndarray, device: str) -> dict[str, Any]:
    sources = [Path(__file__), *LIBRARY_SOURCES]
    return {
        "config": asdict(config),
        "normalization": NORMALIZATION,
        "git_head": subprocess.check_output(  # noqa: S603 - fixed git argv, no shell.
            [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name()
        if device == "cuda"
        else platform.processor(),
        "tf32": False,
        "precision": "float32",
        "train_rows": TRAIN,
        "test_rows": TRAIN,
        "train_rows_sha256": digest(rows[:TRAIN]),
        "test_rows_sha256": digest(rows[TRAIN:]),
        "batch_size": BATCH,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "context": [0] * config.order + ["a", "b"],
        "target_encoding": f"order-{config.order} difference of integer grid levels",
        "block_size": BLOCK,
        "supervised_positions": [1],
        "evaluation": "complete train/test every epoch; no early stopping on loss",
    }


@torch.no_grad()
def verify_checkpoint(
    directory: Path,
    device: str = "cpu",
    *,
    full_predict: bool = False,
) -> dict[str, Any]:
    """Reload saved averaged weights and replay both splits and DiffGPT.predict."""
    metadata = json.loads((directory / "metadata.json").read_text())
    config = RunConfig(**metadata["config"])
    checkpoint = load_checkpoint(directory / "checkpoint.pt", device)
    if metadata["codec"]["encoder_version"] != ENCODER_VERSION:
        msg = "checkpoint was trained with a different encoder version"
        raise ValueError(msg)
    model = GPT(
        vocab_size=vocabulary(config.order),
        n_embd=config.width,
        block_size=BLOCK,
        n_head=config.heads,
        n_layer=config.layers,
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = checkpoint["rows"].cpu().numpy()
    if (
        digest(rows[:TRAIN]) != metadata["train_rows_sha256"]
        or digest(rows[TRAIN:]) != metadata["test_rows_sha256"]
    ):
        msg = "saved split differs from the measured split"
        raise ValueError(msg)
    lookup, _ = token_table(config.order)
    inputs, targets = make_tensors(rows, lookup, device, config.order)
    answers = torch.as_tensor(rows[:, 2], device=device)
    metrics = evaluate_splits(model, inputs, targets, answers, config.order)
    for split in ("train", "test"):
        expected = checkpoint["metrics"][split]
        if (
            abs(metrics[split]["loss"] - expected["loss"]) > REPLAY_LOSS_TOLERANCE
            or metrics[split]["accuracy"] != expected["accuracy"]
        ):
            msg = f"checkpoint {split} metrics disagree: {metrics[split]} != {expected}"
            raise ValueError(msg)
    wrapper = DiffGPT(
        model,
        order_of_derivative=config.order,
        domain_of_definition=domain(config.order),
        tokenizer="lgt",
    )
    probes = []
    for index in (TRAIN, TRAIN + 19, TOTAL - 1):
        a, b, q = map(int, rows[index])
        context = np.array([0.0] * config.order + [a, b], dtype=np.float64).reshape(
            -1, 1
        )
        prediction = wrapper.predict(
            pd.DataFrame(context, columns=["value"]),
            max_new_points=1,
            sampler=TemperatureSampler(temperature=0.0),
        )
        predicted = float(prediction.iloc[-1, 0])
        logits, _ = model(inputs[index : index + 1])
        token = int(logits[0, -1].argmax())
        _, grid, _ = encode(
            context,
            vocabulary(config.order),
            domain(config.order),
            config.order,
            tokenizer="lgt",
        )
        decoded = decode(
            context[-config.order :] if config.order else np.empty((0, 1)),
            grid,
            np.array([[token]], dtype=np.int64),
            vocabulary(config.order),
            config.order,
            tokenizer="lgt",
        )
        if predicted != float(decoded[-1, 0]):
            msg = "DiffGPT.predict disagrees with benchmark logits and decoder"
            raise ValueError(msg)
        probes.append(
            {
                "a": a,
                "b": b,
                "q": q,
                "prediction": predicted,
                "rounded": round(predicted),
            }
        )
    result = {
        "epoch": checkpoint["epoch"],
        "metrics": metrics,
        "diffgpt_predict": probes,
    }
    if full_predict:
        forecasts = wrapper.predict_many(
            [
                pd.DataFrame(
                    {"value": [0.0] * config.order + row[:2].astype(float).tolist()}
                )
                for row in rows[TRAIN:]
            ],
            max_new_points=1,
            sampler=TemperatureSampler(temperature=0.0),
        )
        public = np.rint([frame.iloc[-1, 0] for frame in forecasts]).astype(int)
        correct = int(np.count_nonzero(public == rows[TRAIN:, 2]))
        result["diffgpt_predict_many"] = {
            "test_rows": TRAIN,
            "correct": correct,
            "accuracy": correct / TRAIN,
        }
    if config.optimizer == "schedule-free":
        optimizer = ScheduleFreeAdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.train()
        query_metrics = evaluate_splits(model, inputs, targets, answers, config.order)
        with (directory / "history.csv").open(newline="") as handle:
            last = list(csv.DictReader(handle))[-1]
        for split in ("train", "test"):
            if query_metrics[split]["accuracy"] != float(
                last[f"query_{split}_accuracy"]
            ):
                msg = (
                    "restored query weights disagree with "
                    f"the measured {split} accuracy"
                )
                raise ValueError(msg)
        result["query_metrics"] = query_metrics
    write_json(directory / "verification.json", result)
    return result


def run_experiment(
    config: RunConfig,
    directory: Path,
    *,
    device: str = "cpu",
    log_every: int = 25,
) -> dict[str, Any]:
    """Train the repository GPT and optimizer while retaining every epoch."""
    directory.mkdir(parents=True, exist_ok=False)
    lookup, codec = token_table(config.order)
    rows = division_rows(config.seed)
    if config.labels == "random":
        rng = np.random.default_rng(config.seed ^ 0x72616E646F6D)
        rows[:, 2] = rng.integers(0, P, size=TOTAL)
    inputs, targets = make_tensors(rows, lookup, device, config.order)
    answers = torch.as_tensor(rows[:, 2], device=device)
    torch.manual_seed(config.seed)
    model = GPT(
        vocab_size=vocabulary(config.order),
        n_embd=config.width,
        block_size=BLOCK,
        n_head=config.heads,
        n_layer=config.layers,
        use_checkpoint=False,
    ).to(device)
    metadata = _metadata(config, rows, device)
    metadata["codec"] = codec
    metadata["parameters"] = sum(p.numel() for p in model.parameters())
    write_json(directory / "metadata.json", metadata)
    if config.optimizer == "schedule-free":
        optimizer = ScheduleFreeAdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, config.beta2),
            weight_decay=config.weight_decay,
            warmup=config.warmup,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, config.beta2),
            weight_decay=config.weight_decay,
            eps=1e-8,
            foreach=False,
        )
    sampler = EpochSampler(config.seed)
    criterion = Criterion()
    start_time = time.monotonic()
    fields = [
        "epoch",
        "step",
        "seconds",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "query_train_accuracy",
        "query_test_accuracy",
        "train_decoded_accuracy",
        "test_decoded_accuracy",
    ]
    with (directory / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(config.epochs + 1):
            if epoch:
                model.train()
                if isinstance(optimizer, ScheduleFreeAdamW):
                    optimizer.train()
                for batch_number, batch in enumerate(sampler.batches(), start=1):
                    index = torch.tensor(batch, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    _, loss = model(inputs[index], targets[index])
                    loss.backward()
                    if not isinstance(optimizer, ScheduleFreeAdamW):
                        step = (epoch - 1) * UPDATES_PER_EPOCH + batch_number
                        ramp = min(1.0, step / config.warmup) if config.warmup else 1.0
                        optimizer.param_groups[0]["lr"] = config.learning_rate * ramp
                    optimizer.step()
            model.eval()
            query = {}
            if isinstance(optimizer, ScheduleFreeAdamW):
                # Query and averaged evaluation weights are distinct model states.
                query_values = evaluate_splits(
                    model, inputs, targets, answers, config.order
                )
                query = {
                    f"query_{split}_accuracy": values["accuracy"]
                    for split, values in query_values.items()
                }
                optimizer.eval()
            metrics = evaluate_splits(model, inputs, targets, answers, config.order)
            if not all(
                math.isfinite(value)
                for split in metrics.values()
                for value in split.values()
            ):
                msg = f"nonfinite evaluation at epoch {epoch}"
                raise ValueError(msg)
            criterion.observe(
                epoch, metrics["train"]["accuracy"], metrics["test"]["accuracy"]
            )
            elapsed = time.monotonic() - start_time
            row = {
                "epoch": epoch,
                "step": epoch * UPDATES_PER_EPOCH,
                "seconds": elapsed,
                **{
                    f"{split}_{key}": value
                    for split, values in metrics.items()
                    for key, value in values.items()
                },
                **query,
            }
            writer.writerow(row)
            handle.flush()
            status = {
                "config": asdict(config),
                "epoch": epoch,
                "seconds": elapsed,
                "metrics": metrics,
                "criterion": asdict(criterion),
            }
            write_json(directory / "status.json", status)
            if epoch % log_every == 0 or epoch in (config.epochs, criterion.first99):
                print(
                    json.dumps(
                        {"run": config.name(), **row, "criterion": asdict(criterion)}
                    ),
                    flush=True,
                )
            if (
                criterion.first99 is not None
                and epoch >= criterion.first99 + config.after99
            ):
                break
    # The saved model is exactly the scored evaluation state. Retain optimizer
    # iterates too, so averaged and query predictions can both be audited.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rows": torch.from_numpy(rows),
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(config),
            "normalization": NORMALIZATION,
            "sampler_order": sampler.order,
            "sampler_rng": sampler.random.state,
        },
        directory / "checkpoint.pt",
    )
    write_json(directory / "result.json", status)
    return status


def main() -> None:
    """Run one experiment, a resumable JSON plan, or checkpoint verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/grokking-div97")
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--verify", type=Path, help="Directory containing a saved checkpoint"
    )
    parser.add_argument(
        "--full-predict",
        action="store_true",
        help="Verify every test row through DiffGPT.predict_many",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--optimizer", choices=["adamw", "schedule-free"], default="schedule-free"
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--after99", type=int, default=25)
    parser.add_argument("--labels", choices=["division", "random"], default="division")
    parser.add_argument("--order", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.verify:
        print(
            json.dumps(
                verify_checkpoint(
                    args.verify, args.device, full_predict=args.full_predict
                ),
                indent=2,
            )
        )
        return
    if args.plan:
        configs = [RunConfig(**config) for config in json.loads(args.plan.read_text())]
    else:
        configs = [
            RunConfig(
                **{name: getattr(args, name) for name in RunConfig.__dataclass_fields__}
            )
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    for config in configs:
        directory = args.output / config.name()
        if (directory / "result.json").exists():
            recorded = json.loads((directory / "metadata.json").read_text())
            if RunConfig(**recorded["config"]) != config:
                msg = f"different configuration already exists at {directory}"
                raise ValueError(msg)
            if not (directory / "verification.json").exists():
                verify_checkpoint(directory, args.device)
            print(f"Completed: {directory}", flush=True)
            continue
        if directory.exists():
            msg = (
                f"incomplete run exists at {directory}; "
                "preserve it and use another --output"
            )
            raise FileExistsError(msg)
        run_experiment(config, directory, device=args.device, log_every=args.log_every)
        verify_checkpoint(directory, args.device)


if __name__ == "__main__":
    main()
