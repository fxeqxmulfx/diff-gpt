"""The provenance of a run: the commit of the checked-out code."""

import shutil
import subprocess

from diff_gpt.application.paths import ROOT


def git_head() -> str:
    """Return the commit that HEAD names in the repository."""
    return subprocess.check_output(  # noqa: S603 - fixed git argv, no shell.
        [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
