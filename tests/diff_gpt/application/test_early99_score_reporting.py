"""Reject undeclared or unfinished score-study comparisons before promotion."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from research.early99.protocol import selected
from research.early99_score_scale_reporting import run_long, run_screens, verify_results

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


def test_full_audit_requires_finished_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial full-suite summary cannot authorize freezing or a long run."""
    monkeypatch.setattr(verify_results, "RESULTS", tmp_path)
    with pytest.raises(ValueError, match="selection is not yet complete"):
        verify_results.verify()
    assert not list(tmp_path.iterdir())


def test_summary_reconstruction_keeps_censored_row() -> None:
    """Recompute from rows even if a stale aggregate incorrectly claims two passes."""
    passed = {
        "case": "passing",
        "seed": 0,
        "baseline_status": "passed",
        "candidate_status": "passed",
        "candidate_first99_epochs": 2.0,
        "epoch_ratio": 2.0,
        "candidate_seconds_last_run_segment": 1.0,
    }
    censored = {
        "case": "miss",
        "seed": 1,
        "baseline_status": "passed",
        "candidate_status": "budget_exhausted",
        "candidate_first99_epochs": None,
        "epoch_ratio": None,
        "candidate_seconds_last_run_segment": 3.0,
        "candidate_censor_epochs": 100.0,
    }
    report = {
        "runs": [passed, censored],
        "expected_cells": 2,
        "candidate_passed": 2,
        "geometric_mean_epoch_ratio": None,
    }
    result = verify_results.measurement({"parent": report})
    assert result["passed"] == 1
    assert result["geometric_mean_epochs"] is None
    assert result["failures"] == [censored]
    expected_seconds = 4.0
    assert result["trainer_seconds"] == expected_seconds
    assert result["parent_later"] == 1


@pytest.mark.parametrize("change", ["duplicate", "error", "maximum"])
def test_stress_audit_rejects_misleading_pass_summary(change: str) -> None:
    """A saved all-pass count cannot hide missing cases or inaccurate results."""
    stress = json.loads((verify_results.RESULTS / "score-stress.json").read_text())
    settings = json.loads((verify_results.STUDY / "plan.json").read_text())[
        "score_stress"
    ]
    verify_results.verify_stress(stress, settings)
    if change == "duplicate":
        stress["rows"][1] = stress["rows"][0].copy()
        message = "declared grid"
    elif change == "error":
        stress["rows"][0]["query_relative_gradient_error"] = 1.0
        message = "pass flags differ"
    else:
        stress["maximum_relative_gradient_error"] = 0.0
        message = "maximum error differs"
    with pytest.raises(ValueError, match=message):
        verify_results.verify_stress(stress, settings)


def test_long_gate_requires_frozen_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial short result cannot authorize a long diagnostic."""
    monkeypatch.setattr(run_long, "RESULTS", tmp_path)
    with pytest.raises(ValueError, match="Freeze and commit"):
        run_long.frozen_choice()
    assert not list(tmp_path.iterdir())


def test_long_gate_rejects_changed_short_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed artifact stops the launcher before it can call a trainer."""
    frozen = {"frozen_files_sha256": {"report": "before"}, "selection_sha256": "choice"}
    (tmp_path / "selection-integrity.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(run_long, "RESULTS", tmp_path)
    monkeypatch.setattr(
        run_long,
        "verify",
        lambda: {**frozen, "frozen_files_sha256": {"report": "changed"}},
    )
    with pytest.raises(ValueError, match="Frozen short evidence changed"):
        run_long.frozen_choice()


def test_long_gate_rejects_retained_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared policy permits long training only for a new shared winner."""
    frozen = {"frozen_files_sha256": {}, "selection_sha256": "choice"}
    (tmp_path / "selection-integrity.json").write_text(json.dumps(frozen))
    (tmp_path / "shared-selection.json").write_text(
        json.dumps({"new_shared_winner": False})
    )
    monkeypatch.setattr(run_long, "RESULTS", tmp_path)
    monkeypatch.setattr(run_long, "verify", lambda: frozen)
    with pytest.raises(ValueError, match="No new shared winner"):
        run_long.frozen_choice()


def test_completed_long_report_cannot_be_overwritten(tmp_path: Path) -> None:
    """Keep existing evidence when a repeated report would change it."""
    path = tmp_path / "report.json"
    run_long.publish(path, b"original")
    run_long.publish(path, b"original")
    with pytest.raises(ValueError, match="Completed long artifact changed"):
        run_long.publish(path, b"changed")
    assert path.read_bytes() == b"original"
