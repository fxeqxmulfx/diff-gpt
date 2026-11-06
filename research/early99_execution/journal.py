"""Durably bracket each issued update without changing its batch or optimizer math."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from research.early99.protocol import exposure

if TYPE_CHECKING:
    from collections.abc import Callable


def digest(path: Path) -> str:
    """Bind an archived execution segment to its original bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def durable_write(path: Path, value: dict) -> None:
    """Fsync an atomic record and its directory before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class UpdateJournal:
    """Count completed updates and one possible in-flight batch per segment."""

    def __init__(self, path: Path, train_examples: int, batch: int) -> None:
        """Defer every write until the first genuine update, excluding preparation."""
        self.path, self.train_examples, self.batch = path, train_examples, batch
        self.current: dict | None = None

    def examples(self, completed: int) -> int:
        """Use the benchmark's exact tail-batch exposure convention."""
        return exposure(completed, self.train_examples, self.batch)["examples_seen"]

    def _start(self, completed: int) -> None:
        extra_lower = extra_upper = 0
        archives = []
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if (
                old["train_examples"] != self.train_examples
                or old["batch"] != self.batch
                or old["source_sha256"] != digest(Path(__file__))
                or old["torch"] != str(torch.__version__)
                or old["canonical_completed"] < completed
                or not 0 <= old["extra_examples_lower"] <= old["extra_examples_upper"]
            ):
                message = "Resume changed journal source, data size, batch or position"
                raise ValueError(message)
            for prior in old["prior_journals"]:
                if digest(self.path.parent / prior["file"]) != prior["sha256"]:
                    message = "Prior execution journal changed"
                    raise ValueError(message)
            archive = self.path.with_name(f"execution-before-{time.time_ns()}.json")
            durable_write(archive, old)
            archives = [
                *old["prior_journals"],
                {"file": archive.name, "sha256": digest(archive)},
            ]
            known = self.examples(old["canonical_completed"])
            extra_lower = old["extra_examples_lower"] + known - self.examples(completed)
            extra_upper = (
                old["extra_examples_upper"]
                + known
                + old["pending_batch_examples"]
                - self.examples(completed)
            )
        self.current = {
            "version": 1,
            "train_examples": self.train_examples,
            "batch": self.batch,
            "source_sha256": digest(Path(__file__)),
            "torch": str(torch.__version__),
            "generation_start_completed": completed,
            "canonical_completed": completed,
            "pending_batch_examples": 0,
            "extra_examples_lower": extra_lower,
            "extra_examples_upper": extra_upper,
            "prior_journals": archives,
        }

    def begin(self, completed: int, batch_examples: int) -> None:
        """Persist a possible next batch before issuing any training operation."""
        expected = self.examples(completed + 1) - self.examples(completed)
        if batch_examples != expected:
            message = "Issued batch differs from the frozen nonwrapping epoch sampler"
            raise ValueError(message)
        if self.current is None:
            self._start(completed)
        if (
            self.current["canonical_completed"] != completed
            or self.current["pending_batch_examples"]
        ):
            message = "Journal update position is inconsistent"
            raise ValueError(message)
        self.current["pending_batch_examples"] = batch_examples
        durable_write(self.path, self.current)

    def finish(self, completed: int) -> None:
        """Persist completion only after the actual optimizer update has finished."""
        if (
            self.current is None
            or self.current["canonical_completed"] + 1 != completed
            or not self.current["pending_batch_examples"]
        ):
            message = "No matching pending update to finish"
            raise ValueError(message)
        self.current.update(canonical_completed=completed, pending_batch_examples=0)
        durable_write(self.path, self.current)

    def summary(self, completed: int) -> dict:
        """Report exposure bounds, retaining every interrupted segment."""
        if self.current is None:
            if completed and not self.path.exists():
                message = "No journal for issued updates"
                raise ValueError(message)
            self._start(completed)
            durable_write(self.path, self.current)
        if (
            self.current["canonical_completed"] != completed
            or self.current["pending_batch_examples"]
        ):
            message = "Final journal does not match completed optimizer updates"
            raise ValueError(message)
        canonical = self.examples(completed)
        lower = canonical + self.current["extra_examples_lower"]
        upper = canonical + self.current["extra_examples_upper"]
        return {
            "canonical_completed": completed,
            "canonical_examples": canonical,
            "actual_examples_lower": lower,
            "actual_examples_upper": upper,
            "actual_epochs_lower": lower / self.train_examples,
            "actual_epochs_upper": upper / self.train_examples,
            "extra_examples_lower": self.current["extra_examples_lower"],
            "extra_examples_upper": self.current["extra_examples_upper"],
            "resumed_segments": len(self.current["prior_journals"]),
            "journal_sha256": digest(self.path) if self.path.exists() else None,
        }


def recorded_step(
    journal: UpdateJournal,
    indices: torch.Tensor,
    completed: int,
    device: str,
    update: Callable[[torch.Tensor, int], None],
) -> None:
    """Bracket one real step; CUDA synchronization makes the completed marker sound."""
    journal.begin(completed, len(indices))
    update(indices, completed)
    if device == "cuda":
        torch.cuda.synchronize()
    journal.finish(completed + 1)


def verify_saved_journal(
    directory: Path,
    completed: int,
    train_examples: int,
    batch: int,
    resumptions: list[dict],
    expected: dict,
) -> dict:
    """Recompute exposure from every archived segment and its restored checkpoint."""
    path = directory / "execution-journal.json"
    final = json.loads(path.read_text())
    archives = final["prior_journals"]
    if len(archives) != len(resumptions):
        message = "Journal segments disagree with checkpoint resumptions"
        raise ValueError(message)
    stages = []
    for archive in archives:
        archived = directory / archive["file"]
        if archived.parent != directory or digest(archived) != archive["sha256"]:
            message = "Archived execution segment changed"
            raise ValueError(message)
        stages.append(json.loads(archived.read_text()))
    stages.append(final)
    journal = UpdateJournal(path, train_examples, batch)
    previous = None
    for index, record in enumerate(stages):
        start = resumptions[index - 1]["completed"] if index else 0
        lower = upper = 0
        if previous is not None:
            replay = journal.examples(
                previous["canonical_completed"]
            ) - journal.examples(start)
            lower = previous["extra_examples_lower"] + replay
            upper = (
                previous["extra_examples_upper"]
                + replay
                + previous["pending_batch_examples"]
            )
        next_size = journal.examples(
            record["canonical_completed"] + 1
        ) - journal.examples(record["canonical_completed"])
        if (
            record["source_sha256"] != digest(Path(__file__))
            or record["train_examples"] != train_examples
            or record["batch"] != batch
            or record["generation_start_completed"] != start
            or record["canonical_completed"] < start
            or record["prior_journals"] != archives[:index]
            or record["pending_batch_examples"] not in {0, next_size}
            or record["extra_examples_lower"] != lower
            or record["extra_examples_upper"] != upper
            or not 0 <= lower <= upper
        ):
            message = "Execution journal violates the restored-trajectory accounting"
            raise ValueError(message)
        previous = record
    journal.current = final
    actual = journal.summary(completed)
    if actual != expected:
        message = "Final execution exposure differs from the saved result"
        raise ValueError(message)
    return actual
