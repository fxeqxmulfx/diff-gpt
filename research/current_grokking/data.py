"""Reuse task oracles and data only; never construct historical models."""

from __future__ import annotations

import hashlib
import importlib
import runpy
from functools import cache
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from research.current_grokking.protocol import ARITHMETIC, LEAN

if TYPE_CHECKING:
    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case
    from research.current_grokking.train import Logits


@cache
def arithmetic_registry() -> dict:
    """Load the finite arithmetic functions without their training code."""
    return runpy.run_path(str(ARITHMETIC / "src/training/domain/tasks.py"))


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash actual serialized values, shape and dtype."""
    tensor = tensor.detach().cpu().contiguous()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


class TensorRows:
    """Fixed-width tokens, target masks and per-position change masks."""

    def __init__(
        self, tokens: torch.Tensor, targets: torch.Tensor, device: str
    ) -> None:
        """Initialize the declared data, model and execution state."""
        self.rows = torch.stack((tokens, targets, torch.zeros_like(tokens)), dim=1).to(
            device
        )

    def __len__(self) -> int:
        """Count retained examples."""
        return len(self.rows)

    def __getitem__(self, part: slice) -> torch.Tensor:
        """Return consecutive examples without changing their masks."""
        return self.rows[part]


class DenseTask:
    """A masked token-prediction task with exhaustive split measurements."""

    def __init__(
        self,
        splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
        vocab: int,
        device: str,
        metadata: dict | None = None,
    ) -> None:
        """Initialize the declared data, model and execution state."""
        self.device, self.vocab = device, vocab
        self.splits = {
            name: TensorRows(x, y, device) for name, (x, y) in splits.items()
        }
        self.context = max(rows.rows.shape[-1] for rows in self.splits.values())
        self.fingerprints = {
            name: tensor_digest(rows.rows) for name, rows in self.splits.items()
        }
        self.metadata = metadata or {}
        self.observed = ("train", "validation")
        self.selection = "validation"
        labels = splits["validation"][1]
        labels = labels[labels >= 0]
        self.baseline = float(
            torch.bincount(labels, minlength=vocab).max() / len(labels)
        )

    def inputs(self, indices: torch.Tensor, *, static: bool = False) -> torch.Tensor:
        """Gather the indexed training rows at the frozen padded width."""
        _ = static
        return self.splits["train"].rows.index_select(0, indices.to(self.device))

    def forward(
        self, model: Logits, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and the task-supervised labels."""
        return model(batch[:, 0]), batch[:, 1]

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Average cross entropy over supervised tokens only."""
        return F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), ignore_index=-100
        )

    def accumulator(self) -> torch.Tensor:
        """Allocate exhaustive measurement totals."""
        return torch.zeros(5, dtype=torch.float64, device=self.device)

    def accumulate(
        self,
        model: Logits,
        chunk: torch.Tensor,
        sums: torch.Tensor,
        *,
        static: bool = False,
    ) -> None:
        """Add a complete evaluation chunk to the device totals."""
        _ = static
        logits, targets = self.forward(model, chunk)
        valid = targets >= 0
        hits = logits.argmax(-1) == targets
        losses = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum"
        )
        sums.add_(
            torch.stack(
                (
                    losses.double(),
                    valid.sum().double(),
                    (hits & valid).sum().double(),
                    (hits | ~valid).all(1).sum().double(),
                    valid.any(1).sum().double(),
                )
            )
        )

    @staticmethod
    def metrics(sums: list[float], rows: object) -> dict:
        """Report complete-split losses and correctness counts."""
        _ = rows
        loss, targets, correct, exact, examples = sums
        return {
            "loss": loss / targets,
            "token_accuracy": correct / targets,
            "sequence_accuracy": exact / examples,
            "correct": int(correct),
            "targets": int(targets),
            "exact": int(exact),
            "examples": int(examples),
        }


def arithmetic(case: Case, device: str) -> DenseTask:
    """Use exact function labels and a disjoint fixed half-table split."""
    registry = arithmetic_registry()
    prime = case.data["prime"]
    operation = case.data["operation"]
    first_b = int(operation in ("div", "power-division"))
    pairs = [(a, b) for a in range(prime) for b in range(first_b, prime)]
    random = registry["SplitMix64"](case.data_seed)
    random.shuffle(pairs)
    x = torch.tensor(pairs, dtype=torch.long)
    if operation == "power-division":
        labels = [
            pow(a * pow(b, -1, prime) % prime, case.data["power"], prime)
            for a, b in pairs
        ]
    else:
        function = registry["task"](operation)
        labels = [function(a, b, prime) for a, b in pairs]
    y = torch.full_like(x, -100)
    y[:, -1] = torch.tensor(labels)
    mid = len(x) // 2
    return DenseTask(
        {"train": (x[:mid], y[:mid]), "validation": (x[mid:], y[mid:])},
        prime,
        device,
        {
            "rule": case.data,
            "training_rows": mid,
            "heldout_rows": len(x) - mid,
            "split": "disjoint pairs, fixed 50/50; heldout used for validation",
            "oracle_source": str(ARITHMETIC / "src/training/domain/tasks.py"),
        },
    )


@cache
def synthetic_experiments() -> dict:
    """Read declarative benchmark recipes; their historical models are never built."""
    return runpy.run_path(str(LEAN / "experiments/synthetic_amsgradw/experiment.py"))[
        "experiments"
    ]


def synthetic(case: Case, device: str) -> Task:
    """Keep the source generators, masks, free-generation scorer and task sizes."""
    benchmark = synthetic_experiments()[case.data["label"]].benchmark
    return importlib.import_module("research.current_grokking.adapters").synthetic_task(
        benchmark, case, device
    )


def build(case: Case, device: str) -> Task:
    """Select the declared data adapter, with no fallback to an old GPT."""
    factories = {"arithmetic": arithmetic, "synthetic": synthetic}
    if case.kind in factories:
        return factories[case.kind](case, device)
    if case.kind in {"probe", "fashion", "text"}:
        probes = importlib.import_module("research.current_grokking.probes")
        return {"probe": probes.probe, "fashion": probes.fashion, "text": probes.text}[
            case.kind
        ](case, device)
    adapters = importlib.import_module("research.current_grokking.adapters")
    return adapters.build(case, device)
