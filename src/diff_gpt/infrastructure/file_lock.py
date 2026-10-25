"""An exclusive lock on a file, held across processes."""

import fcntl
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold the lock on the file at `path`, so one run at a time writes beside it."""
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
