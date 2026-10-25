"""Audit a genuine recurrence run: its states, encoder, optimizer and curve."""

from typing import TYPE_CHECKING

import pytest

from diff_gpt.application.grokking.recurrence import Config, run
from diff_gpt.application.grokking.recurrence_audit import audit

if TYPE_CHECKING:
    from pathlib import Path


def test_audit_checks_a_run_and_rejects_it_without_a_stable_tail(
    tmp_path: Path,
) -> None:
    """One epoch passes the state, encoder and optimizer checks, not the tail's."""
    directory = tmp_path / "run"
    run(Config(width=8, layers=1, heads=1, epochs=1), directory, "cpu")
    with pytest.raises(ValueError, match="not enough stability observations"):
        audit(directory, "cpu")
