"""Require fixed controls, successful checks and committed sources before training."""

from __future__ import annotations

import subprocess
from pathlib import Path

from research.early99_dimension_gain.readiness import (
    read,
    require,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
EXPECTED_TESTS = 28
CUDA_TESTS = 2


def verify_training_ready() -> None:
    """Reject stale tests, changed code, or starting before the prior long report."""
    plan = read(STUDY / "plan.json")
    for path, expected in plan["evidence_sha256"].items():
        require(sha256(ROOT / path) == expected, f"Changed control: {path}")
    for configuration in plan["configurations"]:
        require(
            read(STUDY / "configs" / f"{configuration['name']}.json") == configuration,
            "Configuration differs from declared optimizer search",
        )
    validation = read(RESULTS / "validation.json")
    require(
        validation["exit_code"] == 0
        and validation["passed"] == EXPECTED_TESTS
        and validation["cuda_tests_passed"] == CUDA_TESTS
        and validation["skipped"] == 0,
        "Incomplete Schedule-Free CPU/CUDA validation",
    )
    for field in ("source_sha256", "evidence_sha256"):
        for path, expected in validation[field].items():
            require(sha256(ROOT / path) == expected, f"Validated file changed: {path}")
    proof = read(RESULTS / "guarantees.json")
    require(
        proof["exit_code"] == 0
        and proof["warnings"] == 0
        and not proof["extra_axioms"],
        "Unverified optimizer identities",
    )
    for path, expected in proof["source_sha256"].items():
        require(sha256(ROOT / path) == expected, f"Checked proof changed: {path}")
    for path, expected in proof["external_source_sha256"].items():
        require(
            sha256(Path(proof["external_project"]) / path) == expected,
            f"Checked library changed: {path}",
        )
    previous = (
        ROOT / "research/early99_dimension_gain_recovery/results/complete-audit.json"
    )
    closure = read(previous)
    require(closure["ready_for_final_report"], "Finish the prior long diagnostic first")
    for path, expected in closure["long_files_sha256"].items():
        require(sha256(ROOT / path) == expected, f"Prior long report changed: {path}")
    own = [
        *STUDY.glob("*.py"),
        *STUDY.glob("*.lean"),
        *STUDY.glob("configs/*.json"),
        STUDY / "plan.json",
        RESULTS / "validation.json",
        *[ROOT / path for path in validation["evidence_sha256"]],
        previous,
    ]
    for path in own:
        relative = str(path.relative_to(ROOT))
        saved = subprocess.run(  # noqa: S603 -- Read an exact local Git object.
            ["/usr/bin/git", "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        require(saved == path.read_bytes(), f"Commit before training: {relative}")
