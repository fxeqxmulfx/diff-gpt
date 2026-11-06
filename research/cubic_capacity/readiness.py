"""Require validated committed sources before any capacity training update."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING

from research.cubic_capacity.factories import ROOT, STUDY, plan
from research.early99_reporting.round2_audit import require, verify_hash

if TYPE_CHECKING:
    from pathlib import Path

VALIDATION_TESTS = 22
CUDA_TESTS = 2


def sha256(path: Path) -> str:
    """Hash an entire scientific input or source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict:
    """Pin all inherited scientific sources and this study's isolated adapters."""
    parent = ROOT / "runs/early99/candidates/core-ssmax-lr2/plan.json"
    hashes = json.loads(parent.read_text())["source_sha256"]
    for path, expected in hashes.items():
        verify_hash(ROOT / path, expected)
    for path in [
        *STUDY.glob("*.py"),
        STUDY / "plan.json",
        ROOT / "research/early99_reporting/round2_audit.py",
    ]:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


def verify_training_ready() -> dict:
    """Reject unvalidated code, changed inputs or an incomplete prior diagnostic."""
    search = plan()
    for group in ["control_artifacts_sha256", "evidence_sha256"]:
        for path, expected in search[group].items():
            verify_hash(ROOT / path, expected)
    validation_path = STUDY / "results/validation.json"
    validation = json.loads(validation_path.read_text())
    require(
        validation["passed"] == VALIDATION_TESTS
        and validation["failed"] == 0
        and validation["skipped"] == 0
        and validation["cuda_tests_passed"] == CUDA_TESTS,
        "Incomplete native update, resume or benchmark validation",
    )
    require(
        validation["source_sha256"] == source_hashes(),
        "Validated training sources changed",
    )
    for path, expected in validation["evidence_sha256"].items():
        verify_hash(ROOT / path, expected)
    paths = [
        validation_path,
        *[ROOT / path for path in validation["source_sha256"]],
        *[ROOT / path for path in validation["evidence_sha256"]],
    ]
    for path in set(paths):
        committed = subprocess.run(  # noqa: S603 -- Fixed local committed file.
            ["/usr/bin/git", "show", f"HEAD:{path.relative_to(ROOT)}"],
            check=True,
            capture_output=True,
        ).stdout
        require(
            committed == path.read_bytes(),
            f"Commit validated evidence before training: {path}",
        )
    return validation
