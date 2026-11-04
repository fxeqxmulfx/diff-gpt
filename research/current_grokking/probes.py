"""
New token-GPT versions of the catalog's supplied-feature diagnostics.

Every learned network remains the public GPT. Numeric inputs and answers
are serialized explicitly; answers are scored by free autoregressive
generation. These tasks test full-model learning, not the original
query-only or gate-only optimization parameterizations.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch
from lab.domain.synthetic import Synthetic
from lab.domain.tasks import MQAR
from lab.infrastructure.benchmarks.synthetic.generation import rollout, static_rollout
from lab.infrastructure.benchmarks.synthetic.records import generation_example
from lab.infrastructure.benchmarks.synthetic.rows import Rows
from lab.infrastructure.benchmarks.synthetic.training import SyntheticTask
from lab.infrastructure.benchmarks.synthetic.vocabulary import (
    BOS,
    EOS,
    IDENTITY_BASE,
    SEP,
)

from research.current_grokking.adapters import synthetic_task
from research.current_grokking.data import DenseTask, tensor_digest
from research.current_grokking.protocol import LEAN, ROOT

if TYPE_CHECKING:
    from research.current_grokking.protocol import Case


class NumericMetrics:
    """Add freely decoded vector MSE, including explicit invalid-token penalties."""

    def accumulator(self) -> torch.Tensor:
        """Append squared error, component count and valid numeric outputs."""
        ordinary = super().accumulator()
        return torch.cat(
            (ordinary, torch.zeros(3, dtype=ordinary.dtype, device=ordinary.device))
        )

    def accumulate(
        self, model: object, chunk: object, sums: torch.Tensor, *, static: bool = False
    ) -> None:
        """Keep original sequence metrics and score the independent numeric teacher."""
        super().accumulate(model, chunk, sums[:-3], static=static)
        storage = chunk.rows.untyped_storage().data_ptr()
        owner, truth = self.numeric_rows[storage]
        start = (
            chunk.rows.storage_offset() - owner.rows.storage_offset()
        ) // owner.rows.stride(0)
        teacher = truth[start : start + len(chunk.rows)]
        width = teacher.shape[1]
        if chunk.answers is not None:
            generate = static_rollout if static else rollout
            predicted, lengths = generate(model, chunk.answers)
            predicted = predicted[:, :width]
            reached = torch.arange(width, device=self.device) < lengths[:, None]
        else:
            tokens, targets = chunk.rows[:, 0], chunk.rows[:, 1]
            positions = (
                (targets >= IDENTITY_BASE)
                .int()
                .argsort(dim=1, descending=True, stable=True)[:, :width]
            )
            predicted = model(tokens).argmax(-1).gather(1, positions)
            reached = torch.ones_like(predicted, dtype=torch.bool)
        valid = reached & (predicted >= IDENTITY_BASE) & (predicted < self.vocab)
        value = (predicted - IDENTITY_BASE - self.numeric_offset) * self.numeric_step
        error = torch.where(
            valid, (value - teacher).square(), torch.full_like(teacher, 64.0)
        )
        sums[-3:].add_(
            torch.stack(
                (error.sum(), torch.ones_like(error).sum(), valid.sum().double())
            )
        )

    def metrics(self, sums: list[float], rows: object) -> dict:
        """Numeric failures cannot look good merely by emitting EOS early."""
        result = super().metrics(sums[:-3], rows)
        result["mse"] = sums[-3] / sums[-2]
        result["numeric_output_coverage"] = sums[-1] / sums[-2]
        return result


class GeneratedTask:
    """Delegate masked training and honest free generation to the source scorer."""

    def __new__(
        cls,
        examples: dict,
        vocab: int,
        device: str,
        metadata: dict,
        truths: dict | None = None,
        step: float = 0.25,
        offset: int = 16,
    ) -> object:
        """Construct an evaluator-compatible supplied-data task without an old model."""

        class NumericSyntheticTask(NumericMetrics, SyntheticTask):
            """Keep the original evaluator with an additional numeric score."""

        task = object.__new__(
            NumericSyntheticTask if truths is not None else SyntheticTask
        )
        task.spec = SimpleNamespace(task=SimpleNamespace(final_token=False), study=None)
        task.device, task.vocab = device, vocab
        task.generator = SimpleNamespace(token_name=str)
        task.splits = {
            name: Rows(rows, device, generate=not name.startswith("train"))
            for name, rows in examples.items()
        }
        task.measures = None
        task.context = max(
            max(len(row.tokens), len(row.prompt) + row.generation_limit - 1)
            for rows in examples.values()
            for row in rows
        )
        task.observed, task.selection, task.baseline = (
            ("train", "validation"),
            "validation",
            0.0,
        )
        task.metadata = metadata
        task.fingerprints = {
            name: hashlib.sha256(
                json.dumps([vars(row) for row in rows], sort_keys=True).encode()
            ).hexdigest()
            for name, rows in examples.items()
        }
        if truths is not None:
            task.metadata["numeric_grid"] = {
                "step": step,
                "offset": offset,
                "unquantized_teacher_sha256": {
                    name: tensor_digest(value) for name, value in truths.items()
                },
            }
            task.numeric_step, task.numeric_offset = step, offset
            task.numeric_rows = {
                rows.rows.untyped_storage().data_ptr(): (rows, truths[name].to(device))
                for name, rows in task.splits.items()
            }
            mean = truths["train"].mean(0)
            task.baseline = -float((truths["validation"] - mean).square().mean())
            task.metadata["numeric_grid"][
                "invalid_or_missing_component_squared_error"
            ] = 64.0
        return task


def probe(case: Case, device: str) -> object:
    """Freeze supplied-feature rules before training all GPT parameters."""
    name = case.data["task"]
    if name == "key-value-binding":
        benchmark = Synthetic(
            task=MQAR(symbols=32, pairs=2, queries=1),
            length=12,
            train=2048,
            validation=512,
            test=512,
            target=0.99,
        )
        task = synthetic_task(benchmark, case, device)
        task.metadata["design"] = (
            "Two bindings and one independent query, with 32 keys/values; "
            "no second-query elimination shortcut."
        )
        return task
    if name == "order-discrimination":
        pairs = torch.tensor([(a, b) for a in range(32) for b in range(32) if a != b])
        order = torch.randperm(
            len(pairs), generator=torch.Generator().manual_seed(case.data_seed)
        )
        pairs = pairs[order]
        x = pairs + IDENTITY_BASE
        y = torch.full_like(x, -100)
        y[:, -1] = (pairs[:, 0] < pairs[:, 1]).long() + IDENTITY_BASE + 32
        cut = len(x) // 2
        return DenseTask(
            {"train": (x[:cut], y[:cut]), "validation": (x[cut:], y[cut:])},
            IDENTITY_BASE + 34,
            device,
            {
                "rule": "a < b for distinct tokenized integers 0..31",
                "design": (
                    "Generalizes the original two-order witness to held-out "
                    "pairs; all embeddings remain learned."
                ),
            },
        )
    examples, truths = {}, {}
    width = 16 if name == "routing-mixture" else 32
    branches = 4 if name == "conditional-amplitudes" else 1
    step = 1 / 32 if name == "routing-mixture" else 0.25
    offset = 128 if name == "routing-mixture" else 16
    maximum = 256 if name == "routing-mixture" else 32
    for split, count, nonce in (
        ("train", 1024, 1000),
        ("validation", 256, 2000),
        ("test", 256, 3000),
    ):
        rng = torch.Generator().manual_seed(case.data_seed + nonce)
        if name == "routing-mixture":
            sources = (
                torch.randn(count, 8, width, generator=rng, dtype=torch.float64) / 2 * 8
            ).round().clamp(-16, 16) / 8
            selected = torch.randint(8, (count,), generator=rng)
            sources[:, :, 0] = -4
            sources[torch.arange(count), selected, 0] = 4
            scores = (
                sources[:, :, 0]
                / ((sources.square().mean(-1) + 1e-6) * (1 + 1e-6)).sqrt()
            )
            answer = (scores.softmax(-1)[..., None] * sources).sum(1)
            inputs = sources.flatten(1)
            design = (
                "Eight shuffled width-16 source vectors; markers +/-4; "
                "independent Gaussian payloads rounded to 1/8 and clipped to "
                "+/-2. Teacher is the explicit finite-epsilon normalized soft "
                "mixture. Input markers stay exact; numeric answers use a "
                "1/32 grid."
            )
        else:
            signs = (
                2 * torch.randint(2, (count, branches + 1, width), generator=rng) - 1
            )
            signs = signs.double()
            identity, residuals = signs[:, 0], signs[:, 1:]
            coefficients = (
                torch.tensor([0.25, 0.5, 0.75, 0.25], dtype=torch.float64)[:branches]
                .expand(count, -1)
                .clone()
            )
            if name == "conditional-amplitudes":
                other = torch.tensor([0.75, 0.25, 0.5, 0.75], dtype=torch.float64)
                coefficients[identity[:, 0] > 0] = other
            answer = (
                identity / 4
                if name == "linear-transport"
                else identity
                + (coefficients[..., None] * residuals / (1 + 2**-23) ** 0.5).sum(1)
            )
            inputs = identity if name == "linear-transport" else signs.flatten(1)
            design = (
                "Disjoint supplied Rademacher sign vectors. Predict x/4, "
                "x+sum(alpha_i*u_i/RMS_eps(u_i)), or conditional coefficients "
                "selected by the first identity sign. Quantized vector "
                "answers are generated freely; all GPT parameters train."
            )
        input_ids = torch.round(inputs / step).long() + offset + IDENTITY_BASE
        answer_ids = torch.round(answer / step).long() + offset + IDENTITY_BASE
        if input_ids.min() < IDENTITY_BASE or input_ids.max() > IDENTITY_BASE + maximum:
            msg = "Numeric probe exceeds its declared input grid"
            raise ValueError(msg)
        if (
            answer_ids.min() < IDENTITY_BASE
            or answer_ids.max() > IDENTITY_BASE + maximum
        ):
            msg = "Numeric probe exceeds its declared answer grid"
            raise ValueError(msg)
        examples[split] = [
            generation_example(
                name,
                (BOS, *map(int, prompt), SEP),
                (*map(int, target), EOS),
                len(target) + 1,
            )
            for prompt, target in zip(input_ids, answer_ids, strict=True)
        ]
        truths[split] = answer
    return GeneratedTask(
        examples,
        IDENTITY_BASE + maximum + 1,
        device,
        {
            "task": name,
            "design": design,
            "vector_width": width,
            "branches": branches,
            "answer_metric": (
                "Exact freely generated quantized vector, including EOS; not "
                "a query-only/gate-only convergence claim."
            ),
        },
        truths=truths,
        step=step,
        offset=offset,
    )


def fashion(case: Case, device: str) -> DenseTask:
    """Train public GPT on local, checksum-verified Fashion-MNIST image tokens."""
    reference = LEAN / (
        "experiments/archive/synthetic_trainers/baselines/fashion_rff"
        "_20261002/plan.json"
    )
    files = json.loads(reference.read_text())["dataset"]["files"]
    base = LEAN / "experiments/runs/paper_data/fashion-mnist"
    arrays, hashes = {}, {}
    for name, specification in files.items():
        path = base / name
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != specification["sha256"]:
            msg = "Fashion-MNIST data do not match the published archive"
            raise ValueError(msg)
        decoded = gzip.decompress(raw)
        images = "images" in name
        header = 16 if images else 8
        values = np.frombuffer(decoded[header:], dtype=np.uint8).copy()
        arrays[name] = torch.from_numpy(
            values.reshape(-1, 28, 28) if images else values
        )
        hashes[name] = digest
    train_images = arrays["train-images-idx3-ubyte.gz"]
    labels = arrays["train-labels-idx1-ubyte.gz"]
    permutation = torch.randperm(
        len(labels), generator=torch.Generator().manual_seed(case.data_seed)
    )

    def encode(
        images: torch.Tensor, answers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patches = (
            images.float()
            .reshape(-1, 14, 2, 14, 2)
            .mean((2, 4))
            .round()
            .long()
            .flatten(1)
        )
        x = torch.cat((torch.full((len(images), 1), 266), patches), dim=1)
        y = torch.full_like(x, -100)
        y[:, -1] = answers.long() + 256
        return x, y

    cut = 50_000
    splits = {
        "train": encode(train_images[permutation[:cut]], labels[permutation[:cut]]),
        "validation": encode(
            train_images[permutation[cut:]], labels[permutation[cut:]]
        ),
        "test": encode(
            arrays["t10k-images-idx3-ubyte.gz"], arrays["t10k-labels-idx1-ubyte.gz"]
        ),
    }
    return DenseTask(
        splits,
        267,
        device,
        {
            "source_sha256": hashes,
            "input": "BOS then 196 row-major rounded mean intensities of 2x2 patches",
            "classes": 10,
            "training_rows": cut,
            "validation_rows": 10_000,
            "official_test_rows": 10_000,
            "design": (
                "New trained token-GPT image classifier; does not reproduce "
                "the original frozen random-feature double-descent solver."
            ),
        },
    )


def text(case: Case, device: str) -> DenseTask:
    """Use disjoint chronological Tiny Shakespeare segments and fixed contexts."""
    path = ROOT / "data/tinyshakespeare.txt"
    raw = path.read_text()
    alphabet = sorted(set(raw))
    lookup = {char: index for index, char in enumerate(alphabet)}
    tokens = torch.tensor([lookup[char] for char in raw], dtype=torch.long)
    first, second = int(len(tokens) * 0.8), int(len(tokens) * 0.9)
    spans = {
        "train": tokens[:first],
        "validation": tokens[first:second],
        "test": tokens[second:],
    }
    splits = {}
    context = 128
    for name, values in spans.items():
        count = 50_000 if name == "train" else 1024
        generator = torch.Generator().manual_seed(
            case.data_seed + {"train": 0, "validation": 1000, "test": 2000}[name]
        )
        starts = torch.randperm(len(values) - context, generator=generator)[:count]
        x = torch.stack([values[start : start + context] for start in starts])
        y = torch.stack([values[start + 1 : start + context + 1] for start in starts])
        splits[name] = (x, y)
    return DenseTask(
        splits,
        len(alphabet),
        device,
        {
            "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "alphabet": alphabet,
            "context": context,
            "split": "chronological 80/10/10, fixed training-only window pool",
            "design": (
                "Language-model quality/generalization check, without "
                "asserting a grokking plateau."
            ),
        },
    )
