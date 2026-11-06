"""Guard against omitted failures or CSVs misrepresenting frozen JSON records."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import pytest

from research.early99_reporting.round2_audit import (
    geometric_mean_epochs,
    screen_decision,
    verify_csv,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_csv_retains_censoring_and_every_row(tmp_path: Path) -> None:
    """A plausible-looking CSV cannot drop a miss or relabel it as successful."""
    rows = [
        {"seed": 0, "status": "passed", "first99": 2.0, "censor_epochs": None},
        {
            "seed": 1,
            "status": "budget_exhausted",
            "first99": None,
            "censor_epochs": 99.0,
        },
    ]
    path = tmp_path / "report.csv"

    def write(records: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(records)

    write(rows)
    verify_csv(path, rows)
    write(rows[:1])
    with pytest.raises(ValueError, match="CSV disagrees"):
        verify_csv(path, rows)
    write([rows[0], {**rows[1], "status": "passed"}])
    with pytest.raises(ValueError, match="CSV disagrees"):
        verify_csv(path, rows)


def test_improved_screen_medians_cannot_override_a_budget_miss() -> None:
    """Require full cell coverage even if every reported task median improves."""
    rows = [{"candidate_status": "passed"} for _ in range(18)]
    tasks = [{"median_paired_epoch_ratio": 0.5} for _ in range(6)]
    assert screen_decision(rows, tasks) == (18, 6, True)
    rows[-1]["candidate_status"] = "budget_exhausted"
    assert screen_decision(rows, tasks) == (17, 6, False)


def test_aggregate_requires_all_expected_successes() -> None:
    """Do not turn pending or censored cells into a filtered speed improvement."""
    rows = [
        {"candidate_status": "passed", "candidate_first99_epochs": value}
        for value in (1.0, 2.0, 4.0)
    ]
    expected = 2.0
    assert (
        geometric_mean_epochs({"expected_cells": 3, "runs": rows}, "candidate")
        == expected
    )
    assert (
        geometric_mean_epochs({"expected_cells": 4, "runs": rows}, "candidate") is None
    )
    rows[-1] = {
        "candidate_status": "budget_exhausted",
        "candidate_first99_epochs": None,
    }
    assert (
        geometric_mean_epochs({"expected_cells": 3, "runs": rows}, "candidate") is None
    )
