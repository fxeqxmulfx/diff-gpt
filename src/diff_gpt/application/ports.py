"""
The ports of the application layer: what a use case asks of the outside.

A use case takes each of them as an argument; the command lines pass the
adapters of the infrastructure, and a test may pass its own.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

GitHead = Callable[[], str]
"""Return the commit of the checked-out code."""

FileLock = Callable[[Path], AbstractContextManager[None]]
"""Hold an exclusive lock on the file at the path while the block runs."""

BrowserReplay = Callable[[list[dict[str, Any]], Path], list[dict[str, Any]]]
"""Replay the entries through the browser module at the path; return its verdicts."""


class UsageError(ValueError):
    """An argument the use case cannot run with: its CLI reports it as a usage error."""
