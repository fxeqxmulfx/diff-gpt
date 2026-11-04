"""Interfaces shared by native tensor tasks and the external task evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch
    from torch import nn


class Rows(Protocol):
    """A tensor-backed collection with evaluator-specific chunks."""

    rows: torch.Tensor

    def __len__(self) -> int:
        """Count the examples, including a short epoch tail."""
        ...

    def __getitem__(self, part: slice) -> object:
        """Return the task's chunk for consecutive examples."""
        ...


class Task(Protocol):
    """The data/loss/measurement contract; no model construction is permitted."""

    device: str
    vocab: int
    context: int
    splits: dict[str, Rows]
    fingerprints: dict
    metadata: dict
    observed: tuple[str, ...]
    selection: str
    baseline: float

    def inputs(self, indices: torch.Tensor, *, static: bool = False) -> torch.Tensor:
        """Gather an actual training batch through the frozen sampler indices."""
        ...

    def forward(
        self, model: nn.Module, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return supervised logits and masked labels."""
        ...

    def loss(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Reduce the declared training objective."""
        ...

    def accumulator(self) -> torch.Tensor:
        """Allocate device-side measurement totals."""
        ...

    def accumulate(
        self,
        model: nn.Module,
        chunk: object,
        sums: torch.Tensor,
        *,
        static: bool = False,
    ) -> None:
        """Add all examples of one evaluation chunk."""
        ...

    def metrics(self, sums: list, rows: Rows) -> dict:
        """Convert complete-split totals into reported measurements."""
        ...
