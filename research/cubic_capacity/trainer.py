"""Call the unchanged native trainer with explicit capacity-study factories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from research.cubic_capacity.factories import configuration, factories
from research.current_grokking import run as _bootstrap  # noqa: F401
from research.early99_cosine.trainer import run as native_run

if TYPE_CHECKING:
    from pathlib import Path

    from research.current_grokking.protocol import Case


def run(
    case: Case,
    seed: int,
    output: Path,
    device: str,
    *,
    specification: dict,
    resume: bool = False,
    compiled: bool = True,
) -> dict:
    """Retain native sampling, capture, journals, optimizer and first crossing."""
    return native_run(
        case,
        seed,
        output,
        device,
        configuration=configuration(specification),
        factories=factories(specification),
        resume=resume,
        compiled=compiled,
    )
