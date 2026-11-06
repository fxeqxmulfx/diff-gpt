"""Verify actual replay accounting across tail batches and repeated interruptions."""

import json
from typing import TYPE_CHECKING

import pytest
import torch

from research.early99_execution.journal import (
    UpdateJournal,
    recorded_step,
    verify_saved_journal,
)

if TYPE_CHECKING:
    from pathlib import Path

BATCH = 8
TAIL = 5
RESUMPTIONS = 2


def test_pending_tail_and_two_resumptions_preserve_all_actual_work(
    tmp_path: Path,
) -> None:
    """A pending five-example tail gives a five-example interval, not unbounded loss."""
    path = tmp_path / "journal.json"
    first = UpdateJournal(path, 21, 8)
    assert not path.exists()
    for completed in (0, 1):
        first.begin(completed, 8)
        first.finish(completed + 1)
    first.begin(2, 5)
    assert json.loads(path.read_text())["pending_batch_examples"] == TAIL
    resumed = UpdateJournal(path, 21, 8)
    resumed.begin(1, 8)
    resumed.finish(2)
    resumed.begin(2, 5)
    resumed.finish(3)
    summary = resumed.summary(3)
    assert (
        summary["canonical_examples"],
        summary["actual_examples_lower"],
        summary["actual_examples_upper"],
    ) == (21, 29, 34)
    repeated = UpdateJournal(path, 21, 8)
    repeated.begin(2, 5)
    repeated.finish(3)
    summary = repeated.summary(3)
    assert (summary["actual_examples_lower"], summary["actual_examples_upper"]) == (
        34,
        39,
    )
    assert summary["resumed_segments"] == RESUMPTIONS
    assert len(list(tmp_path.glob("execution-before-*.json"))) == RESUMPTIONS


def test_recorded_step_marks_pending_before_callback_and_completes_after(
    tmp_path: Path,
) -> None:
    """The durable marker must bracket the real operation in execution order."""
    path = tmp_path / "journal.json"
    journal = UpdateJournal(path, 21, 8)
    issued = []

    def update(indices: torch.Tensor, completed: int) -> None:
        value = json.loads(path.read_text())
        assert value["canonical_completed"] == completed
        assert value["pending_batch_examples"] == len(indices)
        issued.append(indices.clone())

    recorded_step(journal, torch.arange(8), 0, "cpu", update)
    assert len(issued) == 1
    assert journal.summary(1)["actual_examples_lower"] == BATCH
    assert journal.summary(1)["actual_examples_upper"] == BATCH


def test_failed_operation_retains_pending_marker_and_wrong_batch_is_rejected(
    tmp_path: Path,
) -> None:
    """Do not declare an interrupted update completed or accept altered batch order."""
    path = tmp_path / "journal.json"
    journal = UpdateJournal(path, 21, 8)
    with pytest.raises(ValueError, match="Issued batch"):
        journal.begin(0, 5)
    assert not path.exists()

    def failure(_indices: torch.Tensor, _completed: int) -> None:
        message = "Interrupted"
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="Interrupted"):
        recorded_step(journal, torch.arange(8), 0, "cpu", failure)
    value = json.loads(path.read_text())
    assert value["canonical_completed"] == 0
    assert value["pending_batch_examples"] == BATCH
    with pytest.raises(ValueError, match="Final journal"):
        journal.summary(0)


def test_saved_segment_audit_reconstructs_replay_and_rejects_changed_archives(
    tmp_path: Path,
) -> None:
    """Validate the durable chain against an independent restored-checkpoint record."""
    path = tmp_path / "execution-journal.json"
    first = UpdateJournal(path, 21, BATCH)
    first.begin(0, BATCH)
    first.finish(1)
    first.begin(1, BATCH)
    resumed = UpdateJournal(path, 21, BATCH)
    resumed.begin(1, BATCH)
    resumed.finish(2)
    summary = resumed.summary(2)
    verified = verify_saved_journal(tmp_path, 2, 21, BATCH, [{"completed": 1}], summary)
    assert (verified["actual_examples_lower"], verified["actual_examples_upper"]) == (
        16,
        24,
    )
    archive = next(tmp_path.glob("execution-before-*.json"))
    archive.write_text("{}\n")
    with pytest.raises(ValueError, match="Archived execution"):
        verify_saved_journal(tmp_path, 2, 21, BATCH, [{"completed": 1}], summary)


def test_zero_update_observation_has_an_exact_zero_exposure_record(
    tmp_path: Path,
) -> None:
    """Finalizing an initial crossing records no training or replayed examples."""
    journal = UpdateJournal(tmp_path / "execution-journal.json", 21, BATCH)
    summary = journal.summary(0)
    assert summary["actual_examples_lower"] == summary["actual_examples_upper"] == 0
    assert verify_saved_journal(tmp_path, 0, 21, BATCH, [], summary) == summary


def test_completed_checkpoint_can_finalize_without_issuing_another_update(
    tmp_path: Path,
) -> None:
    """A restored first-crossing checkpoint keeps its exact exposure and zero replay."""
    path = tmp_path / "execution-journal.json"
    first = UpdateJournal(path, 21, BATCH)
    first.begin(0, BATCH)
    first.finish(1)
    resumed = UpdateJournal(path, 21, BATCH)
    summary = resumed.summary(1)
    assert summary["actual_examples_lower"] == summary["actual_examples_upper"] == BATCH
    assert (
        verify_saved_journal(tmp_path, 1, 21, BATCH, [{"completed": 1}], summary)
        == summary
    )
