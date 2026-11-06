"""Bind the original-model adapters to the unchanged journaled native training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

# The frozen environment bootstrap exposes its pinned external lab dependency.
from research.current_grokking import run as _bootstrap  # noqa: F401
from research.early99_core_ablations.factories import configured_factories
from research.early99_cosine.trainer import run as journaled_run

if TYPE_CHECKING:
    from pathlib import Path

    from research.current_grokking.protocol import Case
    from research.early99.experiments.factories import Factories
    from research.early99_core_ablations.factories import Configuration


def run(
    case: Case,
    seed: int,
    output: Path,
    device: str = "cuda",
    *,
    configuration: Configuration,
    resume: bool = False,
    compiled: bool = True,
    factories: Factories | None = None,
) -> dict:
    """Pass explicit factories; change no sampler, update, observation or stop rule."""
    return journaled_run(
        case,
        seed,
        output,
        device,
        resume=resume,
        compiled=compiled,
        configuration=configuration,
        factories=configured_factories(configuration)
        if factories is None
        else factories,
    )
