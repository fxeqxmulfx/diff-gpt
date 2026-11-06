"""Require the completed capacity study and validated committed gain-only code."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from research.cubic_capacity.factories import plan as capacity_plan
from research.cubic_capacity.readiness import sha256
from research.cubic_capacity.readiness import source_hashes as capacity_sources
from research.cubic_usual_gain.factories import ROOT, STUDY, plan
from research.early99_reporting.round2_audit import read, require, verify_hash

if TYPE_CHECKING:
    from pathlib import Path

VALIDATION_CHECKS = 6
CUDA_CHECKS = 1


def source_hashes() -> dict:
    """Extend the unchanged inherited scientific source manifest."""
    hashes = capacity_sources()
    for path in [*STUDY.glob("*.py"), STUDY / "plan.json"]:
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    return hashes


def require_committed(paths: list[Path]) -> None:
    """Reject uncommitted source or completion evidence before scientific training."""
    for path in set(paths):
        committed = subprocess.run(  # noqa: S603 -- Fixed local git read.
            ["/usr/bin/git", "show", f"HEAD:{path.relative_to(ROOT)}"],
            capture_output=True,
            check=True,
        ).stdout
        require(committed == path.read_bytes(), f"Commit required: {path}")


def capacity_completed() -> list[Path]:
    """Keep the user-declared experiment order and all completed capacity cells."""
    directory = ROOT / "research/cubic_capacity/results"
    audit_path, comparison_path = (
        directory / "complete-audit.json",
        directory / "comparison.json",
    )
    require(
        audit_path.is_file() and comparison_path.is_file(),
        "Finish and audit all capacity runs first",
    )
    audited, comparison = read(audit_path), read(comparison_path)
    expected = capacity_plan()
    require(
        audited["complete"]
        and audited["new_training_cells"] == expected["new_training_cells"]
        and audited["reused_controls"] == expected["reused_current_size_cells"]
        and comparison["complete"]
        and comparison["complete_audit_sha256"] == sha256(audit_path),
        "Capacity study is incomplete",
    )
    verify_hash(
        ROOT / "research/cubic_capacity_reporting/audit.py", audited["auditor_sha256"]
    )
    for row in audited["rows"]:
        for path, expected in row["artifacts_sha256"].items():
            verify_hash(ROOT / path, expected)
    return [audit_path, comparison_path]


def verify_training_ready() -> dict:
    """Require real CUDA checks and complete prior results before issuing updates."""
    completion_paths = capacity_completed()
    search = plan()
    for group in ["evidence_sha256", "control_artifacts_sha256"]:
        for path, digest in search[group].items():
            verify_hash(ROOT / path, digest)
    validation_path = STUDY / "results/validation.json"
    validation = read(validation_path)
    require(
        validation["passed"] == VALIDATION_CHECKS
        and validation["cuda_tests_passed"] == CUDA_CHECKS
        and validation["failed"] == 0
        and validation["skipped"] == 0,
        "Incomplete gain-only validation",
    )
    require(validation["source_sha256"] == source_hashes(), "Changed validated sources")
    for path, digest in validation["evidence_sha256"].items():
        verify_hash(ROOT / path, digest)
    require_committed(
        [
            validation_path,
            *completion_paths,
            *[ROOT / p for p in validation["source_sha256"]],
            *[ROOT / p for p in validation["evidence_sha256"]],
        ]
    )
    return validation
