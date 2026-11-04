"""Task-data adapters for the unchanged token transformer."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from lab.domain.basis import BASIS
from lab.domain.memorization import Memorization
from lab.domain.modular_composition import ModularChain
from lab.infrastructure.benchmarks.modular_composition import ModularCompositionTask
from lab.infrastructure.benchmarks.synthetic.rows import Rows
from lab.infrastructure.benchmarks.synthetic.training import SyntheticTask
from torch.nn import functional as F

from diff_gpt.application.grokking import div97, mqar_div97, recurrence
from diff_gpt.application.grokking import series as series_metrics
from diff_gpt.application.grokking.recurrence_field_sampler import CanonicalFieldSampler
from diff_gpt.application.grokking.series_data import DataConfig, dataset
from diff_gpt.domain.encoder_decoder import decode, encode
from research.current_grokking import forecast
from research.current_grokking.data import DenseTask, tensor_digest

if TYPE_CHECKING:
    from research.current_grokking.protocol import Case
    from research.current_grokking.train import Logits


def synthetic_task(benchmark: object, case: Case, device: str) -> object:
    """Use the original free-generation scorer and declared selection split."""
    task = SyntheticTask(benchmark, case.data_seed, device)
    task.context = max(
        problem.task.context(problem.length)
        for problem in (benchmark.problem, *benchmark.probes.values())
    )
    original_selection = benchmark.selection
    task.selection = original_selection
    novel_name = original_selection + "/novel"
    if novel_name not in task.splits:
        train_keys = {
            example.prompt or example.tokens
            for example in task.splits["train"].examples
        }
        novel = [
            example
            for example in task.splits[original_selection].examples
            if (example.prompt or example.tokens) not in train_keys
        ]
        if novel:
            task.splits[novel_name] = Rows(novel, device, generate=True)
            task.fingerprints[novel_name] = tensor_digest(task.splits[novel_name].rows)
    if task.splits.get(novel_name) is not None:
        task.selection = novel_name
    task.observed = tuple(dict.fromkeys(("train", original_selection, task.selection)))
    task.baseline = 0.5 if type(benchmark.task).__name__ == "Parity" else 0.0
    task.metadata = {
        "benchmark": str(benchmark),
        "free_generation": benchmark.task.generative,
        "original_selection": original_selection,
        "primary_selection": task.selection,
        "selection_rule": (
            "Score novel prompts/inputs when available; retain source "
            "validation separately."
        ),
    }
    return task


def basis(case: Case, device: str) -> object:
    """Preserve original Basis distributions, including hard depth's OOD selection."""
    name = case.data["task"]
    mode = case.data.get("mode", "easy")
    benchmark = BASIS[mode][name][0]
    if name == "parity":
        benchmark = replace(benchmark, length=case.data["bits"])
        if case.data.get("noise"):
            benchmark = replace(
                benchmark,
                train=512,
                validation=128,
                test=256,
                study=Memorization(
                    noise=case.data["noise"], noise_seed=2, disjoint=True
                ),
            )
    if case.data.get("neutral_fraction") is not None:
        benchmark = replace(
            benchmark,
            task=replace(
                benchmark.task,
                neutral_fraction=case.data["neutral_fraction"],
                blocks=case.data["blocks"],
            ),
        )
    return synthetic_task(benchmark, case, device)


def chain(case: Case, device: str) -> DenseTask:
    """Keep source chain sampling, hidden intermediates and doubled-length test."""
    spec = ModularChain(
        prime=case.data["prime"],
        steps=case.data["steps"],
        train_examples=2048,
        heldout_examples=1024,
        longer_examples=1024,
    )
    source = ModularCompositionTask(spec, case.data_seed, "cpu")
    splits = {}
    for name, rows in source.splits.items():
        x = rows[:, :-1]
        y = torch.full_like(x, -100)
        y[:, -1] = rows[:, -1]
        splits[
            {"heldout": "validation", "longer": "test/length-double"}.get(name, name)
        ] = (x, y)
    return DenseTask(splits, source.vocab, device, source.metadata)


def differenced_division(case: Case, device: str) -> DenseTask:
    """Keep the actual codec and exact first/second-order equation encoding."""
    order = case.data["order"]
    lookup, codec = div97.token_table(order)
    rows = div97.division_rows(case.data_seed)
    x, y = div97.make_tensors(rows, lookup, "cpu", order)
    cut = div97.TRAIN
    return DenseTask(
        {"train": (x[:cut], y[:cut]), "validation": (x[cut:], y[cut:])},
        div97.vocabulary(order),
        device,
        {"codec": codec, "source": "div97.make_tensors"},
    )


def temporal_records(config: object) -> tuple:
    """Reuse orbit splits, encoding every tokenizer through its public API."""
    orbits = recurrence.trajectory_orbits(config.task)
    order = list(range(len(orbits)))
    div97.SplitMix64(config.seed ^ 0x6F72626974730001).shuffle(order)
    train, heldout, train_ids, heldout_ids = [], [], [], []
    for index in order:
        values, identities = (
            (train, train_ids) if len(train) <= len(heldout) else (heldout, heldout_ids)
        )
        values.extend(orbits[index])
        identities.append(index)
    div97.SplitMix64(config.seed ^ 0x747261696E000002).shuffle(train)
    div97.SplitMix64(config.seed ^ 0x7465737400000003).shuffle(heldout)
    raw = recurrence.windows([*train, *heldout], config)
    vocab = recurrence.grid_vocabulary(config)
    domains = np.repeat(recurrence.grid_domain(config), len(raw))
    start, grid, tokens = encode(
        raw.T.astype(np.float64),
        vocab,
        domains,
        config.order,
        tokenizer=config.tokenizer,
    )
    _, past_grid, past_tokens = encode(
        raw[:, :-1].T.astype(np.float64),
        vocab,
        domains,
        config.order,
        tokenizer=config.tokenizer,
    )
    np.testing.assert_array_equal(tokens[:-1], past_tokens)
    np.testing.assert_array_equal(grid.scale, past_grid.scale)
    np.testing.assert_array_equal(grid.zero, past_grid.zero)
    center = np.full((1, len(raw)), vocab // 2, dtype=np.int64)
    bases = decode(
        start,
        past_grid,
        np.vstack((past_tokens, center)),
        vocab,
        config.order,
        tokenizer=config.tokenizer,
    )[-1]
    decoder = np.column_stack((bases, 1 / past_grid.scale))
    x = torch.from_numpy(tokens[:-1].T.copy())
    y = torch.full_like(x, -100)
    y[:, -1] = torch.from_numpy(tokens[-1].copy())
    metadata = {
        "task": config.task,
        "tokenizer": config.tokenizer,
        "order": config.order,
        "train_orbits": train_ids,
        "validation_orbits": heldout_ids,
        "train_rows": len(train),
        "validation_rows": len(heldout),
        "orbit_lengths": [len(orbit) for orbit in orbits],
        "encoder": (
            "actual public encode, including MASH-half; no integer-grid shortcut"
        ),
        "decoder": (
            "actual public decode with observed prefix tokens and a "
            "center-token probe; no future values used"
        ),
        "raw_sha256": div97.digest(raw),
    }
    return raw, len(train), x, y, decoder, metadata


class RecurrenceTask(DenseTask):
    """Count decoded future values, retaining native-token accuracy separately."""

    def __init__(self, case: Case, device: str) -> None:
        """Build entire disjoint orbits with the original temporal codec."""
        config = recurrence.Config(
            **case.data,
            seed=case.data_seed,
            readout="field"
            if case.data["task"] == "fibonacci" and case.data["tokenizer"] == "lgt"
            else "argmax",
        )
        raw, cut, x, y, decoder, metadata = temporal_records(config)
        super().__init__(
            {"train": (x[:cut], y[:cut]), "validation": (x[cut:], y[cut:])},
            recurrence.grid_vocabulary(config),
            device,
            metadata,
        )
        self.context += 1
        self.config = config
        self.raw = torch.as_tensor(raw, device=device)
        self.decoder = torch.as_tensor(decoder, device=device)
        self.splits["train"].rows[:, 2, 0] = torch.arange(cut, device=device)
        self.splits["validation"].rows[:, 2, 0] = torch.arange(
            cut, len(raw), device=device
        )
        self.fingerprints["raw"] = tensor_digest(self.raw)
        self.fingerprints.update(
            {name: tensor_digest(rows.rows) for name, rows in self.splits.items()}
        )

    def accumulator(self) -> torch.Tensor:
        """Loss, field-token hits, decoded hits, native hits and examples."""
        return torch.zeros(5, dtype=torch.float64, device=self.device)

    def accumulate(
        self,
        model: Logits,
        chunk: torch.Tensor,
        sums: torch.Tensor,
        *,
        static: bool = False,
    ) -> None:
        """Evaluate the same field/readout and decoder rules as the original runner."""
        _ = static
        logits = model(chunk[:, 0])[:, -1]
        target = chunk[:, 1, -1]
        indices = chunk[:, 2, 0]
        raw = self.raw[indices]
        native = logits.argmax(-1)
        predicted = (
            CanonicalFieldSampler(raw[:, :-1], self.config.order, self.vocab, 97)(
                logits
            )[:, 0]
            if self.config.readout == "field"
            else native
        )
        value = predicted - self.vocab // 2
        if self.decoder is None:
            for lag in range(1, self.config.order + 1):
                value = (
                    value
                    - (-1) ** lag * math.comb(self.config.order, lag) * raw[:, -1 - lag]
                )
        else:
            grid = self.decoder[indices]
            value = torch.round(grid[:, 0] + value * grid[:, 1])
        sums.add_(
            torch.stack(
                (
                    F.cross_entropy(logits, target, reduction="sum").double(),
                    (predicted == target).sum().double(),
                    (value == raw[:, -1]).sum().double(),
                    (native == target).sum().double(),
                    torch.ones_like(target).sum().double(),
                )
            )
        )

    @staticmethod
    def metrics(sums: list[float], rows: object) -> dict:
        """Report decoded future accuracy as the declared sequence metric."""
        _ = rows
        loss, field, decoded, native, count = sums
        return {
            "loss": loss / count,
            "sequence_accuracy": decoded / count,
            "decoded_accuracy": decoded / count,
            "token_accuracy": field / count,
            "native_token_accuracy": native / count,
            "examples": int(count),
        }


def associative(case: Case, device: str) -> DenseTask:
    """Freeze complete contexts while preserving equation/register split oracles."""
    config = mqar_div97.RunConfig(
        seed=case.data_seed,
        loads=case.data["loads"],
        labels=case.data["labels"],
        computed=case.data["compute"],
        overwrites=case.data["overwrites"],
        contexts="fixed",
    )
    source = mqar_div97.Task(config)
    splits = {
        "train": source.tensors(source.evaluation["train"], "cpu"),
        "validation": source.tensors(source.evaluation["test"], "cpu"),
    }
    vocab = int(source.codec.metadata["vocab_size"])
    return DenseTask(
        splits,
        vocab,
        device,
        {
            "config": vars(config),
            "data_protocol": config.data_protocol,
            "original_oracle": "mqar_div97.Task",
            "contexts": (
                "fixed before training; not the historical fresh-context sampler"
            ),
        },
    )


class SeriesTask(DenseTask):
    """Proper CE, decoded errors and known-floor skill on train-fitted codecs."""

    def __init__(self, case: Case, device: str) -> None:
        """Reuse dataset preparation and never fit the codec on held-out data."""
        config_data = dict(case.data)
        config_data.setdefault("context", 2)
        if config_data["dataset"] in ("div97", "fibonacci"):
            config_data.setdefault("vocab_size", 256)
        self.config = DataConfig(**config_data, seed=case.data_seed)
        self.dataset = dataset(self.config)
        self.features = len(self.dataset.columns)
        names_map = {name: name for name in self.dataset.splits}
        if "validation" not in names_map:
            names_map["test"] = "validation"
        splits = {}
        for name, split in self.dataset.splits.items():
            x = torch.as_tensor(split.x.copy(), dtype=torch.long)
            y = torch.full_like(x, -100)
            y[:, -self.features :] = torch.as_tensor(
                split.y[:, -self.features :].copy()
            )
            splits[names_map[name]] = (x, y)
        super().__init__(splits, self.config.vocab_size, device, self.dataset.metadata)
        if case.metric in {"skill", "relative_ce_gain"}:
            self.baseline = 0.0
        self.context += 1
        self.extras = {}
        optional = ("bayes", "seasonal", "past")
        names = [
            "clean_y",
            "base",
            "step",
            "truth",
            *[
                key
                for key in optional
                if all(
                    getattr(split, key) is not None
                    for split in self.dataset.splits.values()
                )
            ],
        ]
        for key in names:
            values = np.concatenate(
                [getattr(split, key) for split in self.dataset.splits.values()]
            )
            self.extras[key] = torch.as_tensor(values.copy(), device=device)
        offset = 0
        original_names = {value: key for key, value in names_map.items()}
        for name, rows in self.splits.items():
            rows.rows[:, 2, 0] = torch.arange(offset, offset + len(rows), device=device)
            rows.name = original_names[name]
            offset += len(rows)
        self.fingerprints["full_dataset"] = self.dataset.fingerprint()
        self.fingerprints.update(
            {name: tensor_digest(rows.rows) for name, rows in self.splits.items()}
        )

    def accumulator(self) -> torch.Tensor:
        """Keep all per-column counts and proper scoring totals on device."""
        return torch.zeros((self.features, 10), dtype=torch.float64, device=self.device)

    def final_metrics(self, model: Logits) -> dict:
        """Evaluate free-running forecasts against equal-origin naive controls."""
        return forecast.evaluate(self, model)

    def accumulate(
        self,
        model: Logits,
        chunk: torch.Tensor,
        sums: torch.Tensor,
        *,
        static: bool = False,
    ) -> None:
        """Match the existing series evaluator, including its field marginalizer."""
        _ = static
        idx = chunk[:, 2, 0]
        logits = model(chunk[:, 0])[:, -self.features :]
        target = chunk[:, 1, -self.features :]
        native = logits.argmax(-1)
        predicted = native
        if self.config.dataset == "fibonacci":
            predicted = CanonicalFieldSampler(
                self.extras["past"][idx], self.config.order, self.vocab, 97
            )(logits[:, 0])
        log_probability = F.log_softmax(logits, dim=-1)
        loss = -log_probability.gather(-1, target[..., None])[..., 0]
        grid = (
            torch.arange(self.vocab, device=self.device, dtype=log_probability.dtype)
            - self.vocab // 2
        )
        expected = (log_probability.exp() * grid).sum(-1)
        value = self.extras["base"][idx] + self.extras["step"][idx] * expected
        error = value - self.extras["truth"][idx]
        bayes = (
            -(self.extras["bayes"][idx] * log_probability).sum(-1)
            if "bayes" in self.extras
            else torch.zeros_like(loss)
        )
        seasonal = (
            self.extras["seasonal"][idx] - self.extras["truth"][idx]
            if "seasonal" in self.extras
            else torch.zeros_like(loss)
        )
        values = (
            loss,
            predicted == target,
            predicted == self.extras["clean_y"][idx, -self.features :],
            native == target,
            error.square(),
            error.abs(),
            bayes,
            torch.ones_like(loss),
            seasonal.square(),
            seasonal.abs(),
        )
        sums.add_(torch.stack([value.double().sum(0) for value in values], dim=-1))

    def metrics(self, sums: list[list[float]], rows: object) -> dict:
        """Report per-column ceilings and explicit normalized target metrics."""
        columns = {}
        for index, column in enumerate(self.dataset.columns):
            values = sums[index]
            count = values[7]
            record = {
                key: values[position] / count
                for position, key in enumerate(
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
            reference = self.dataset.references[rows.name][column]
            observed_loss = (
                record["bayes_loss"]
                if "bayes" in self.extras and rows.name != "train"
                else record["loss"]
            )
            record["skill"] = (
                None
                if reference.get("input_independent_control")
                else series_metrics.skill(
                    observed_loss, reference["baseline_loss"], reference["floor_loss"]
                )
            )
            record["relative_ce_gain"] = (
                (reference["baseline_loss"] - record["loss"])
                / reference["baseline_loss"]
                if reference["baseline_loss"]
                else None
            )
            if "bayes" not in self.extras:
                record["bayes_loss"] = None
            if "seasonal" in self.extras:
                record.update(
                    seasonal_mse=values[8] / count, seasonal_mae=values[9] / count
                )
            columns[column] = record
        keys = next(iter(columns.values()))
        result = {
            key: float(np.mean([row[key] for row in columns.values()]))
            if all(row[key] is not None for row in columns.values())
            else None
            for key in keys
        }
        result["columns"] = columns
        return result


def build(case: Case, device: str) -> object:
    """Build additional task data; this module never instantiates a GPT."""
    factories = {
        "basis": basis,
        "chain": chain,
        "differenced_division": differenced_division,
        "recurrence": RecurrenceTask,
        "associative": associative,
        "series": SeriesTask,
    }
    if case.kind not in factories:
        message = f"No additional adapter for {case.kind}"
        raise KeyError(message)
    return factories[case.kind](case, device)
