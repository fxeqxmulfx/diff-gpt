"""Compare native DiffGPT samplers and tokenizers on frozen temporal weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from benchmarks.grokking.div97 import P, digest, load_checkpoint, write_json
from benchmarks.grokking.recurrence import (
    Config,
    build,
    encode_windows,
    future_grid,
    grid_domain,
    grid_vocabulary,
    tensors,
)
from benchmarks.grokking.recurrence_field_sampler import CanonicalFieldSampler
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import encode
from diff_gpt.optimizer.schedule_free import ScheduleFreeAdamW
from diff_gpt.sampler.median import MedianSampler
from diff_gpt.sampler.nucleus import NucleusSampler
from diff_gpt.sampler.sampler import Sampler
from diff_gpt.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from collections.abc import Callable

    from diff_gpt.model.gpt import GPT

SOURCE_SNAPSHOT = Path(__file__).read_bytes()
FIELD_SOURCE_SNAPSHOT = (
    Path(__file__).with_name("recurrence_field_sampler.py").read_bytes()
)
ALIAS_CONTEXT = 2


def next_values(
    tokens: torch.Tensor,
    past: torch.Tensor,
    config: Config,
    decoder_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert one difference using observed past rows and the actual decoder."""
    value = tokens - grid_vocabulary(config) // 2
    if config.grid_domain or config.vocab_size:
        if decoder_grid is None:
            decoder_grid = torch.as_tensor(
                future_grid(past.cpu().numpy(), config), device=past.device
            )
        return torch.round(decoder_grid[:, :1] + value * decoder_grid[:, 1:])
    for lag in range(1, config.order + 1):
        coefficient = (-1) ** lag * math.comb(config.order, lag)
        value = value - coefficient * past[:, -lag, None]
    return value


class MeanSampler(Sampler):
    """Return the nearest ordinal token to the posterior mean."""

    def __init__(self, temperature: float = 1.0) -> None:
        """Optionally sharpen the distribution before taking its expectation."""
        if temperature <= 0:
            msg = "the posterior mean needs a positive temperature"
            raise ValueError(msg)
        self.temperature = temperature

    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        """Compute a deterministic point estimate without a field or range prior."""
        del rng
        bins = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        probabilities = (logits / self.temperature).softmax(-1)
        return (probabilities * bins).sum(-1, keepdim=True).round().long()


class RangeSampler(Sampler):
    """
    Restrict a native readout to the known range of the next original value.

    This one-step diagnostic uses the observed prefix and the range only. It
    neither evaluates the recurrence nor reads a future target. Predictions
    with this extra support constraint are reported separately from native
    model predictions.
    """

    def __init__(self, past: torch.Tensor, config: Config, sampler: Sampler) -> None:
        """Bind a readout to one observed batch on the integer grid."""
        self.sampler = sampler
        candidates = torch.arange(grid_vocabulary(config), device=past.device)
        values = next_values(candidates[None, :], past, config)
        lower = 1 if config.task == "division" else 0
        self.allowed = (values >= lower) & (values < P)
        if not self.allowed.any(-1).all():
            msg = "the observed prefix leaves no legal next token"
            raise ValueError(msg)

    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        """Apply the underlying sampler to the range-restricted logits."""
        if logits.shape != self.allowed.shape:
            msg = "RangeSampler is bound to one batch and one forecast step"
            raise ValueError(msg)
        return self.sampler(logits.masked_fill(~self.allowed, -torch.inf), rng=rng)


class FieldSampler(CanonicalFieldSampler):
    """
    Combine model probability of equivalent residues and emit a legal token.

    The extra prior is that original values are canonical residues modulo P.
    Neither a recurrence operation nor a future value enters this readout.
    This finite-field diagnostic is separate from the native samplers.
    """

    def __init__(self, past: torch.Tensor, config: Config) -> None:
        """Bind the unit-grid decoder to observed past rows of one batch."""
        if config.grid_domain or config.vocab_size:
            msg = "FieldSampler currently requires the exact integer grid"
            raise ValueError(msg)
        super().__init__(
            past,
            config.order,
            grid_vocabulary(config),
            P,
            lower=int(config.task == "division"),
        )


class QuantizedFieldSampler(Sampler):
    """Pool rounded decoded residues on a non-unit grid using the actual codec."""

    def __init__(self, past: torch.Tensor, config: Config) -> None:
        """Build output classes and canonical tokens from observed rows alone."""
        bins = torch.arange(grid_vocabulary(config), device=past.device)[None, :]
        self.residues = next_values(bins, past, config).long().remainder(P)
        self.lower = int(config.task == "division")
        self.past = past.cpu().numpy().copy()
        self.config = config

    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        """Return the actual derivative token of the chosen canonical residue."""
        del rng
        if logits.shape != self.residues.shape:
            msg = "quantized field readout is bound to one batch and one forecast step"
            raise ValueError(msg)
        mass = logits.new_zeros((len(logits), P))
        mass.scatter_add_(1, self.residues, logits.softmax(-1))
        if self.lower:
            mass[:, : self.lower] = -1
        value = mass.argmax(-1, keepdim=True)
        # The tokenizer can depend on the observed window's anchor. Encode the
        # selected continuation rather than assuming a unit-grid token shift.
        predicted = np.concatenate((self.past, value.cpu().numpy()), axis=1)
        tokens, _ = encode_windows(predicted, self.config)
        return torch.as_tensor(tokens[:, -1:], device=logits.device)


class AliasSampler(FieldSampler):
    """
    Average model forecasts of equivalent input-difference residues modulo P.

    Four representations of the two observed differences are queried. The
    only extra prior is invariance to the integer representatives of field
    elements; no recurrence operation or future row is used.
    """

    def __init__(self, past: torch.Tensor, config: Config, model: GPT) -> None:
        """Encode observed rows and build their bounded equivalent prompts."""
        if config.order != 1 or config.context != ALIAS_CONTEXT:
            msg = "AliasSampler currently supports order 1 and context 2"
            raise ValueError(msg)
        super().__init__(past, config)
        tokens, _ = encode_windows(past.cpu().numpy(), config)
        residues = (torch.as_tensor(tokens, device=past.device) - self.center) % P
        bound = P - 1 - self.lower
        positive = torch.where(residues <= bound, residues, residues - P)
        negative = torch.where(residues - P >= -bound, residues - P, residues)
        self.prompts = (
            torch.stack(
                [
                    torch.stack((first[:, 0], second[:, 1]), dim=1)
                    for first in (positive, negative)
                    for second in (positive, negative)
                ],
                dim=1,
            )
            + self.center
        )
        self.model = model

    @torch.inference_mode()
    def __call__(
        self, logits: torch.Tensor, rng: torch.Generator | None = None
    ) -> torch.Tensor:
        """Use four model queries per context, then pool equivalent output bins."""
        del rng
        if logits.shape != self.residues.shape:
            msg = "AliasSampler is bound to one batch and one forecast step"
            raise ValueError(msg)
        prompts = self.prompts.reshape(-1, 2)
        probabilities = torch.cat(
            [
                self.model(prompts[first : first + 256])[0][:, -1].softmax(-1)
                for first in range(0, len(prompts), 256)
            ]
        )
        return self.choose(probabilities.reshape(len(logits), 4, -1).mean(1))


def tokenizer_comparison(raw: np.ndarray, config: Config) -> dict[str, object]:
    """Compare actual encoders, batching trajectories as independent columns."""
    encoded = {}
    for tokenizer in ("lgt", "mash"):
        _, scale, tokens = encode(
            raw.T.astype(np.float64),
            grid_vocabulary(config),
            np.repeat(grid_domain(config), len(raw)),
            config.order,
            tokenizer=tokenizer,
        )
        encoded[tokenizer] = tokens.T.copy()
    return {
        "domain": grid_domain(config).tolist(),
        "vocab_size": grid_vocabulary(config),
        "unit_grid": bool(np.all(scale == 1)),
        "scale": float(scale[0]),
        "tokens": int(encoded["lgt"].size),
        "different_tokens": int(np.count_nonzero(encoded["lgt"] != encoded["mash"])),
        "sha256": {key: digest(value) for key, value in encoded.items()},
    }


def sampler_factories(
    past: torch.Tensor, config: Config, model: GPT
) -> dict[str, Callable[[], Sampler]]:
    """Use the repository's existing samplers and explicit support controls."""
    factories = {
        "argmax": lambda: TemperatureSampler(temperature=0.0),
        "median": MedianSampler,
        "mean_1": MeanSampler,
        "mean_0.3": lambda: MeanSampler(temperature=0.3),
        "nucleus_0.9_mean": lambda: NucleusSampler(0.9, MeanSampler()),
        "nucleus_0.5_mean": lambda: NucleusSampler(0.5, MeanSampler()),
        "temperature_0.1": lambda: TemperatureSampler(temperature=0.1),
        "temperature_0.3": lambda: TemperatureSampler(temperature=0.3),
        "temperature_0.5": lambda: TemperatureSampler(temperature=0.5),
        "temperature_1": lambda: TemperatureSampler(temperature=1.0),
        "nucleus_0.9_temperature_0.3": lambda: NucleusSampler(
            0.9, TemperatureSampler(temperature=0.3)
        ),
        "nucleus_0.9_temperature_1": lambda: NucleusSampler(
            0.9, TemperatureSampler(temperature=1.0)
        ),
        "range_argmax": lambda: RangeSampler(
            past, config, TemperatureSampler(temperature=0.0)
        ),
        "range_median": lambda: RangeSampler(past, config, MedianSampler()),
    }
    if config.grid_domain or config.vocab_size:
        factories["field_argmax"] = lambda: QuantizedFieldSampler(past, config)
    else:
        factories["field_argmax"] = lambda: FieldSampler(past, config)
        if config.order == 1 and config.context == ALIAS_CONTEXT:
            factories["alias_field_argmax"] = lambda: AliasSampler(past, config, model)
    return factories


@torch.inference_mode()
def compare(directory: Path, device: str, repeats: int) -> dict[str, object]:
    """Freeze a durable checkpoint; score the entire disjoint test-orbit set."""
    # Load once: a running trainer can atomically replace checkpoint.pt later.
    checkpoint = directory if directory.is_file() else directory / "checkpoint.pt"
    directory = checkpoint.parent
    saved = load_checkpoint(checkpoint, device)
    frozen = directory / f"readouts-epoch{saved['epoch']}-checkpoint.pt"
    if not frozen.exists():
        torch.save(saved, frozen)
    source_digest = hashlib.sha256(SOURCE_SNAPSHOT).hexdigest()
    source_file = directory / f"readouts-source-{source_digest[:16]}.py"
    source_file.write_bytes(SOURCE_SNAPSHOT)
    field_digest = hashlib.sha256(FIELD_SOURCE_SNAPSHOT).hexdigest()
    field_source_file = directory / f"readouts-field-source-{field_digest[:16]}.py"
    field_source_file.write_bytes(FIELD_SOURCE_SNAPSHOT)
    config = Config(**saved["config"])
    model = build(config, device)
    model.load_state_dict(saved["model"])
    model.eval()
    raw = saved["raw"].cpu().numpy()
    train = saved["train_rows"]
    test = torch.as_tensor(raw[train:], device=device)
    past = test[:, :-1]
    decoder_grid = (
        torch.as_tensor(future_grid(raw[train:, :-1], config), device=device)
        if config.grid_domain or config.vocab_size
        else None
    )
    x, y = tensors(raw[train:], config, device)
    target = y[:, -1]
    optimizer = ScheduleFreeAdamW(model.parameters())
    optimizer.load_state_dict(saved["optimizer"])
    comparisons = {}
    for mode in ("averaged", "query"):
        if mode == "query":
            optimizer.train()
        logits = torch.cat(
            [model(x[first : first + 256])[0][:, -1] for first in range(0, len(x), 256)]
        )
        measurements = {}
        for name, factory in sampler_factories(past, config, model).items():
            sampler = factory()
            deterministic = name in {
                "argmax",
                "median",
                "mean_1",
                "mean_0.3",
                "nucleus_0.9_mean",
                "nucleus_0.5_mean",
                "range_argmax",
                "range_median",
                "field_argmax",
                "alias_field_argmax",
            }
            samples = []
            for repeat in range(1 if deterministic else repeats):
                rng = torch.Generator(device=device).manual_seed(1000 + repeat)
                predicted = sampler(logits, rng=rng)
                values = next_values(predicted, past, config, decoder_grid)[:, 0]
                samples.append(
                    {
                        "token_accuracy": float(
                            (predicted[:, 0] == target).double().mean()
                        ),
                        "decoded_accuracy": float(
                            (values == test[:, -1]).double().mean()
                        ),
                        "decoded_mae": float(
                            (values - test[:, -1]).abs().double().mean()
                        ),
                    }
                )
            measurements[name] = {
                "repeats": len(samples),
                "mean": {
                    key: float(np.mean([row[key] for row in samples]))
                    for key in samples[0]
                },
                "min_accuracy": min(row["decoded_accuracy"] for row in samples),
                "max_accuracy": max(row["decoded_accuracy"] for row in samples),
            }
        comparisons[mode] = measurements
    return {
        "directory": str(directory),
        "frozen_checkpoint": str(frozen),
        "checkpoint_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "readout_source_sha256": source_digest,
        "readout_source": str(source_file),
        "field_source_sha256": field_digest,
        "field_source": str(field_source_file),
        "checkpoint_epoch": saved["epoch"],
        "config": asdict(config),
        "test_rows": len(x),
        "tokenizers": tokenizer_comparison(raw, config),
        "samplers": comparisons,
        "support_constraint": (
            "range_argmax/median use observed past and the known range only; "
            "field_argmax additionally combines probability of residues modulo P"
        ),
    }


@torch.inference_mode()
def verify_readout(directory: Path, name: str, device: str) -> dict[str, object]:
    """Check a selected readout through the actual public DiffGPT API."""
    checkpoint = directory if directory.is_file() else directory / "checkpoint.pt"
    saved = load_checkpoint(checkpoint, device)
    config = Config(**saved["config"])
    model = build(config, device)
    model.load_state_dict(saved["model"])
    model.eval()
    raw = saved["raw"][saved["train_rows"] :].cpu().numpy()
    wrapper = DiffGPT(
        model, config.order, grid_domain(config), tokenizer=config.tokenizer
    )
    values = []
    for first in range(0, len(raw), 256):
        batch = raw[first : first + 256]
        past = torch.as_tensor(batch[:, :-1], device=device)
        sampler = sampler_factories(past, config, model)[name]()
        predictions = wrapper.predict_many(
            [pd.DataFrame({"x": row[:-1].astype(float)}) for row in batch],
            max_new_points=1,
            sampler=sampler,
            batch_size=len(batch),
        )
        values.extend(frame.iloc[-1, 0] for frame in predictions)
    return {
        "checkpoint_epoch": saved["epoch"],
        "sampler": name,
        "public_rows": len(raw),
        "public_accuracy": float(np.mean(np.rint(values).astype(int) == raw[:, -1])),
    }


def main() -> None:
    """Persist frozen-weight comparisons without interrupting training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--verify-readout")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for directory in args.directories:
        if args.verify_readout:
            result = verify_readout(directory, args.verify_readout, args.device)
            filename = f"readout-{args.verify_readout}-verification.json"
        else:
            result = compare(directory, args.device, args.repeats)
            filename = f"readouts-epoch{result['checkpoint_epoch']}.json"
        output = directory.parent if directory.is_file() else directory
        write_json(output / filename, result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
