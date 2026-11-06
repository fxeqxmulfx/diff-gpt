"""Reject undeclared or unfinished score-study comparisons before promotion."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from research.early99.protocol import selected
from research.early99_score_scale_reporting import run_screens

if TYPE_CHECKING:
    from pathlib import Path


def setup_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Provide isolated declared metadata without touching scientific records."""
    name = "test-score-arm"
    study, runs = tmp_path / "study", tmp_path / "runs"
    study.mkdir()
    (runs / name).mkdir(parents=True)
    configuration = {"name": name, "score_multiplier": 0.5}
    plan = {
        "configurations": [configuration],
        "screen_tasks": [row["name"] for row in selected("short")[:6]],
        "screen_control": str(tmp_path / "control"),
    }
    (study / "plan.json").write_text(json.dumps(plan))
    (runs / name / "plan.json").write_text(json.dumps({"configuration": configuration}))
    monkeypatch.setattr(run_screens, "STUDY", study)
    monkeypatch.setattr(run_screens, "RUNS", runs)
    monkeypatch.setattr(run_screens, "RESULTS", tmp_path / "results")
    return name


def test_rejects_executed_configuration_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching name cannot hide a different score multiplier."""
    name = setup_report(tmp_path, monkeypatch)
    path = run_screens.RUNS / name / "plan.json"
    changed = json.loads(path.read_text())
    changed["configuration"]["score_multiplier"] = 0.25
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differs from the declared"):
        run_screens.report_screen(name)
    assert not run_screens.RESULTS.exists()


def test_pending_cells_cannot_finalize_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep an unfinished run distinct from a pass or an exhausted budget."""
    name = setup_report(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_screens,
        "paired_cell",
        lambda *_: {
            "baseline_status": "passed",
            "candidate_status": "pending",
            "baseline_first99_epochs": 1.0,
            "candidate_first99_epochs": None,
            "epoch_ratio": None,
        },
    )
    with pytest.raises(ValueError, match="Pending or erroneous"):
        run_screens.report_screen(name)
    assert not run_screens.RESULTS.exists()
