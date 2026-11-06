"""Protect capacity conclusions against dropped failures and replay uncertainty."""

from __future__ import annotations

import copy

import pytest

from research.cubic_capacity.factories import plan
from research.cubic_capacity_reporting.audit import verify_observations
from research.cubic_capacity_reporting.report import (
    contrast_metrics,
    group_metrics,
    paired_ratio,
    report,
)

CENSORED_SEED = 2


def cell(size: str, arm: str, seed: int, *, passed: bool = True) -> dict:
    """Build an explicit complete or budget-censored reporting fixture."""
    shape = next(row for row in plan()["shapes"] if row["name"] == size)
    return {
        "size": size,
        "arm": arm,
        "seed": seed,
        "layers": shape["layers"],
        "width": shape["width"],
        "heads": shape["heads"],
        "head_width": shape["head_width"],
        "parameters": shape["unique_parameters"],
        "peak_learning_rate": 0.002,
        "initial_parameters_sha256": f"{size}-{seed}",
        "status": "passed" if passed else "budget_exhausted",
        "first99_epochs_lower": 10.0 if passed else None,
        "first99_epochs_upper": 10.0 if passed else None,
        "canonical_epochs": 10.0 if passed else 100.0,
        "actual_epochs_lower": 10.0 if passed else 100.0,
        "actual_epochs_upper": 10.0 if passed else 100.0,
        "completed_updates": 190 if passed else 1900,
        "seconds_last_run_segment": 20.0,
        "resumptions": 0,
        "reused": size == "current",
    }


def test_keep_censored_seed_and_reject_incomplete_groups() -> None:
    """A fast successful subset cannot replace the full three-seed outcome."""
    rows = [
        cell("width", "normalized_lr2", seed, passed=seed != CENSORED_SEED)
        for seed in range(3)
    ]
    saved = group_metrics(rows)
    assert saved["passed"] == 2  # noqa: PLR2004
    assert saved["budget_exhausted"] == 1
    for metric in [
        "median_first99_epochs",
        "worst_first99_epochs",
        "geometric_mean_first99_epochs",
    ]:
        assert saved[metric] is None
        assert saved[metric + "_lower"] is None
        assert saved[metric + "_upper"] is None
    assert saved["total_actual_epochs_lower"] == 120  # noqa: PLR2004
    for invalid in [rows[:2], [rows[0], rows[0], rows[2]]]:
        with pytest.raises(ValueError, match="seed group"):
            group_metrics(invalid)
    pairs = [
        paired_ratio(
            cell("current", "normalized_lr2", r["seed"], passed=False), r, "capacity"
        )
        for r in rows
    ]
    contrast = contrast_metrics(pairs)
    assert contrast["recovered"] == 2  # noqa: PLR2004
    assert contrast["both_censored"] == 1
    assert contrast["geometric_mean_epoch_ratio"] is None


def test_replay_bounds_and_changed_capacity_are_explicit() -> None:
    """Do not collapse uncertain work to a point estimate or claim matched tensors."""
    base = cell("current", "original_lr1", 0)
    new = cell("width", "original_lr1", 0)
    base.update(first99_epochs_lower=10, first99_epochs_upper=12, resumptions=1)
    new.update(first99_epochs_lower=24, first99_epochs_upper=30)
    compared = paired_ratio(base, new, "capacity")
    assert compared["epoch_ratio_lower"] == 2  # noqa: PLR2004
    assert compared["epoch_ratio_upper"] == 3  # noqa: PLR2004
    assert compared["epoch_ratio"] is None
    assert compared["seconds_ratio"] is None
    assert not compared["same_size_initialization"]
    base["first99_epochs_upper"] = None
    compared = paired_ratio(base, new, "capacity")
    assert compared["epoch_ratio_lower"] == 0
    assert compared["epoch_ratio_upper"] == 3  # noqa: PLR2004
    wrong = {**new, "initial_parameters_sha256": "changed"}
    with pytest.raises(ValueError, match="initialization"):
        paired_ratio(new, wrong, "architecture")


def test_all_grid_cells_and_every_predeclared_contrast_are_retained() -> None:
    """Require all 36 outcomes, including old failures, without selecting a winner."""
    search = plan()
    cells = [
        cell(
            shape["name"],
            arm["name"],
            seed,
            passed=not (shape["name"] == "current" and arm["name"] == "normalized_lr2"),
        )
        for shape in search["shapes"]
        for arm in search["arms"]
        for seed in search["seeds"]
    ]
    audited = {
        "complete": True,
        "new_training_cells": 27,
        "reused_controls": 9,
        "rows": cells,
    }
    saved = report(audited, search)
    assert len(saved["cells"]) == 36  # noqa: PLR2004
    assert len(saved["groups"]) == 12  # noqa: PLR2004
    assert len(saved["comparisons"]) == 17  # noqa: PLR2004
    assert len(saved["paired_cells"]) == 51  # noqa: PLR2004
    assert not saved["fixed75_selection_changed"]
    assert not saved["new_long_suite_authorized_by_this_study"]
    for invalid in [cells[:-1], [*cells, cells[0]]]:
        with pytest.raises(ValueError, match="Missing or duplicate"):
            report({**audited, "rows": invalid}, search)


def test_exact_oracle_and_first_crossing_must_match_full_observation() -> None:
    """Reject an extra step after success, shortened oracle or altered accuracy."""
    row = {
        "case": {"observe": 2, "max_updates": 4},
        "train_examples": 3,
        "heldout_examples": 4,
    }
    history = [
        {
            "step": step,
            "train": {
                "examples": 3,
                "exact": 3,
                "sequence_accuracy": 1.0,
                "loss": 0.01,
            },
            "validation": {
                "examples": 4,
                "exact": exact,
                "sequence_accuracy": exact / 4,
                "loss": 0.1,
            },
        }
        for step, exact in [(0, 1), (2, 4)]
    ]
    verify_observations(history, row, 2)
    with pytest.raises(ValueError, match="first crossing"):
        verify_observations([*history, {**history[-1], "step": 4}], row, 4)
    for key, value in [("examples", 2), ("sequence_accuracy", 0.995)]:
        altered = copy.deepcopy(history)
        altered[-1]["validation"][key] = value
        with pytest.raises(ValueError, match="exact oracle"):
            verify_observations(altered, row, 2)
