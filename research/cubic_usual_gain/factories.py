"""Use the existing full normalized SSMax parent without the inverse-width swap."""

from __future__ import annotations

import json
from pathlib import Path

from research.current_grokking import run as _bootstrap  # noqa: F401
from research.current_grokking.protocol import Case
from research.early99.experiments.factories import (
    Configuration,
    Factories,
    peri_rms_scaled_ssmax_model,
)
from research.early99.protocol import selected
from research.early99_cosine.trainer import run as native_run
from research.early99_reporting.round2_audit import require

STUDY = Path(__file__).parent
ROOT = STUDY.parents[1]
RUNS = ROOT / "runs/cubic-usual-gain"


def plan() -> dict:
    """Reject changes to the reused frozen cubic-sum case."""
    value = json.loads((STUDY / "plan.json").read_text())
    require(
        value["task"]
        == next(
            row for row in selected("long") if row["name"] == value["task"]["name"]
        ),
        "Changed frozen task",
    )
    return value


def configuration() -> Configuration:
    """Retain every existing normalized block and constant native-AdamW setting."""
    return Configuration(**plan()["configuration"])


def factories() -> Factories:
    """Use the unchanged full normalized usual-gain factory directly."""
    return Factories(model=peri_rms_scaled_ssmax_model)


def case_for() -> Case:
    """Keep original depth, width, batch size, data and update budget."""
    return Case(**plan()["task"]["case"])


def train(
    case: Case,
    seed: int,
    output: Path,
    device: str,
    *,
    resume: bool = False,
    compiled: bool = True,
) -> dict:
    """Reuse the native journaled loop with explicit model and optimizer factories."""
    return native_run(
        case,
        seed,
        output,
        device,
        configuration=configuration(),
        factories=factories(),
        resume=resume,
        compiled=compiled,
    )
