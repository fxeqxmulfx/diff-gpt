"""Guard the gain comparison against filtered failures and invalid pairing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from research.cubic_usual_gain.factories import plan
from research.cubic_usual_gain_reporting import audit

if TYPE_CHECKING:
    from pathlib import Path


def fixture_cell(
    root: Path,
    row: dict,
    seed: int,
    *,
    passed: bool,
    resumed: bool = False,
    initialization: str = "paired",
) -> None:
    """Write a small synthetic raw history, without using the metrics under test."""
    stop = row["case"]["observe"] if passed else row["case"]["max_updates"]
    history = [
        {
            "step": step,
            "seconds": float(step),
            "train": {"examples": row["train_examples"]},
            "validation": {"sequence_accuracy": float(passed and step == stop)},
        }
        for step in range(0, stop + 1, row["case"]["observe"])
    ]
    metadata = {
        "case": row["case"],
        "seed": seed,
        "batch_seed": seed + 10000,
        "device": "cpu",
        "torch": "synthetic-report-fixture",
        "compiled_loss": False,
        "data_fingerprints": row["data_fingerprints"],
        "data_metadata": {"fixture": True},
        "observe_updates": row["case"]["observe"],
        "model_source_reference": "synthetic-report-fixture",
        "sources": {},
        "initial_parameters_sha256": initialization,
        "parameters": 1,
        "model_class": "diff_gpt.domain.model.gpt.GPT",
        "resumptions": [{"fixture": True}] if resumed else [],
    }
    result = {
        "completed": stop,
        "first_target": stop if passed else None,
        "endpoint": history[-1],
        "metadata": metadata,
        "seconds": float(stop),
        "optimizer_steps_verified": True,
        "parameters_finite": True,
    }
    directory = root / row["name"] / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(result))
    (directory / "history.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in history)
    )


@pytest.fixture
def comparison_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Use raw-file readers and pairing checks, with a clearly synthetic budget."""
    search = plan()
    search["task"]["case"]["max_updates"] = 2 * search["task"]["case"]["observe"]
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    for seed in search["seeds"]:
        fixture_cell(base, search["task"], seed, passed=True)
        fixture_cell(candidate, search["task"], seed, passed=seed != 2)  # noqa: PLR2004
    return search, base, candidate


def test_censored_seed_prevents_filtered_speedup(comparison_files: tuple) -> None:
    """Fast successful seeds must not produce a complete-study epoch ratio."""
    search, base, candidate = comparison_files
    result = audit.compare_control(base, candidate, search)
    assert len(result["runs"]) == 3  # noqa: PLR2004
    assert result["task_metrics"]["candidate_passed_seeds"] == 2  # noqa: PLR2004
    assert result["task_metrics"]["candidate_median_first99_epochs"] is None
    assert result["geometric_mean_epoch_ratio"] is None
    assert result["runs"][-1]["candidate_censor_epochs"] == 2  # noqa: PLR2004
    assert all(row["seconds_ratio"] is None for row in result["runs"])
    for seeds in [[0, 1], [0, 0, 2]]:
        with pytest.raises(ValueError, match="three-seed"):
            audit.compare_control(base, candidate, {**search, "seeds": seeds})


def test_unknown_replay_stays_an_interval(comparison_files: tuple) -> None:
    """An unjournaled resumed control cannot imply an exact exposure ratio."""
    search, base, candidate = comparison_files
    for seed in search["seeds"]:
        fixture_cell(base, search["task"], seed, passed=True, resumed=True)
        fixture_cell(candidate, search["task"], seed, passed=True)
    result = audit.compare_control(base, candidate, search)
    assert result["geometric_mean_epoch_ratio"] is None
    for row in result["runs"]:
        assert row["baseline_replay_epochs_upper"] is None
        assert row["epoch_ratio_lower"] == 0
        assert row["epoch_ratio_upper"] == 1
        assert row["epoch_ratio"] is None


def test_changed_initializer_and_missing_cell_are_rejected(
    comparison_files: tuple,
) -> None:
    """A gain-only comparison cannot silently include different weights or seeds."""
    search, base, candidate = comparison_files
    fixture_cell(candidate, search["task"], 0, passed=True, initialization="changed")
    with pytest.raises(ValueError, match="original initialization"):
        audit.compare_control(base, candidate, search)
    fixture_cell(candidate, search["task"], 0, passed=True)
    (candidate / search["task"]["name"] / "seed2/result.json").unlink()
    with pytest.raises(ValueError, match="Missing comparison cell"):
        audit.compare_control(base, candidate, search)
