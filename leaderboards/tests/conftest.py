"""Shared fixtures: the tiny GIFT-Eval datasets of tests.tiny on disk."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.tiny import tiny, tiny_multi, write

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a GIFT_EVAL directory that holds "tiny" and "tiny_multi"."""
    root = tmp_path_factory.mktemp("gift-eval")
    write(root, "tiny", tiny())
    write(root, "tiny_multi", tiny_multi())
    return root
