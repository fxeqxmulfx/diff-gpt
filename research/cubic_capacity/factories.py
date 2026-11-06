"""Resolve only the declared dimensions, architecture and existing rate choices."""

from __future__ import annotations

import json
from pathlib import Path

from research.current_grokking.protocol import Case
from research.early99.experiments.factories import Configuration as NativeConfiguration
from research.early99.experiments.factories import Factories, baseline_model
from research.early99.protocol import selected
from research.early99_dimension_gain.factories import (
    Configuration as NormalizedConfiguration,
)
from research.early99_dimension_gain.factories import dimension_model
from research.early99_reporting.round2_audit import require

STUDY = Path(__file__).parent
ROOT = STUDY.parents[1]
RUNS = ROOT / "runs/cubic-capacity"


def plan() -> dict:
    """Load the separately declared capacity study without changing suite membership."""
    value = json.loads((STUDY / "plan.json").read_text())
    original = next(
        row for row in selected("long") if row["name"] == value["task"]["name"]
    )
    require(value["task"] == original, "The underlying task changed")
    return value


def resolve(name: str) -> dict:
    """Reject undeclared configurations, including current-size retraining."""
    configurations = [row for row in plan()["configurations"] if row["name"] == name]
    require(len(configurations) == 1, f"Unknown capacity configuration: {name}")
    return configurations[0]


def task_row(specification: dict) -> dict:
    """Change only layer count, model width and its corresponding head count."""
    search = plan()
    require(
        specification == resolve(specification["name"]), "Undeclared capacity settings"
    )
    shape = next(
        row for row in search["shapes"] if row["name"] == specification["size"]
    )
    require(not shape["reuse_existing"], "Reuse the completed current-size controls")
    allowed = {"layers", "width", "heads"}
    original = search["task"]["case"]
    require(
        {k: v for k, v in shape["case"].items() if k not in allowed}
        == {k: v for k, v in original.items() if k not in allowed},
        "Capacity study changed a noncapacity task field",
    )
    return {**search["task"], "case": shape["case"]}


def configuration(specification: dict) -> NativeConfiguration:
    """Use the original constant native-AdamW settings with explicit rate and model."""
    require(
        specification == resolve(specification["name"]), "Undeclared capacity settings"
    )
    constructor = (
        NormalizedConfiguration
        if specification["architecture"] == "inverse_width"
        else NativeConfiguration
    )
    return constructor(
        name=specification["name"],
        learning_rate_multiplier=specification["learning_rate_multiplier"],
        beta1=0.9,
        beta2=0.98,
        schedule="constant",
    )


def factories(specification: dict) -> Factories:
    """Return the unchanged original or inverse-width construction."""
    require(
        specification == resolve(specification["name"]), "Undeclared capacity settings"
    )
    return Factories(
        model=dimension_model
        if specification["architecture"] == "inverse_width"
        else baseline_model
    )


def case_for(specification: dict) -> Case:
    """Instantiate the declared case while retaining its original training budget."""
    return Case(**task_row(specification)["case"])
