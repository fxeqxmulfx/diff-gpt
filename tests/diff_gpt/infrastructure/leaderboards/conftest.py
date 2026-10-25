"""Shared fixtures: the tiny GIFT-Eval and TIME datasets of tiny.py on disk."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.diff_gpt.infrastructure.leaderboards.tiny import tiny, tiny_multi, write

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a GIFT-Eval storage directory that holds "tiny" and "tiny_multi"."""
    root = tmp_path_factory.mktemp("gift-eval")
    write(root, "tiny", tiny())
    write(root, "tiny_multi", tiny_multi())
    return root


@pytest.fixture(scope="session")
def time_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a TIME storage directory that holds "tiny/H" and "tiny_multi/H"."""
    root = tmp_path_factory.mktemp("time")
    write(root, "tiny/H", tiny())
    write(root, "tiny_multi/H", tiny_multi())
    return root
