"""Require committed Muon implementation, matrix checks and finished prior studies."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from research.early99_dimension_gain.readiness import read, require, sha256

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
EXPECTED_TESTS = 54
CUDA_TESTS = 2
PROOF_STATEMENTS = 6


def verify_training_ready() -> None:
    """Require fixed source, complete CPU/CUDA checks and finished prior work."""
    plan = read(STUDY / "plan.json")
    for path, digest in plan["evidence_sha256"].items():
        require(sha256(ROOT / path) == digest, f"Changed prior evidence: {path}")
    for cfg in plan["configurations"]:
        require(
            read(STUDY / "configs" / f"{cfg['name']}.json") == cfg,
            "Undeclared Muon configuration",
        )
    validation = read(RESULTS / "validation.json")
    require(str(torch.__version__) == validation["torch"], "Validated Torch changed")
    for reference in plan["references"]:
        if "path" in reference:
            require(
                sha256(ROOT / reference["path"]) == reference["sha256"],
                "Inspected native Muon source changed",
            )
    require(
        validation["exit_code"] == 0
        and validation["passed"] == EXPECTED_TESTS
        and validation["skipped"] == 0
        and validation["cuda_tests_passed"] == CUDA_TESTS,
        "Incomplete Muon/CUDA/protocol/journal validation",
    )
    for field in ["source_sha256", "evidence_sha256"]:
        for path, digest in validation[field].items():
            require(sha256(ROOT / path) == digest, f"Validated file changed: {path}")
    proof = read(RESULTS / "guarantees.json")
    require(
        proof["exit_code"] == 0
        and proof["warnings"] == 0
        and proof["statements"] == PROOF_STATEMENTS
        and not proof["extra_axioms"],
        "Unchecked finite-polynomial bounds",
    )
    for path, digest in proof["external_source_sha256"].items():
        require(
            sha256(Path(proof["external_project"]) / path) == digest,
            "Proof library changed",
        )
    prior = read(ROOT / "research/early99_cosine/results/complete-audit.json")
    require(
        prior["study_complete"] and not prior["requires_full_short"],
        "Finish cosine AdamW first",
    )
    for path, digest in prior["files_sha256"].items():
        require(sha256(ROOT / path) == digest, f"Prior study changed: {path}")
    own = [
        *STUDY.glob("*.py"),
        *STUDY.glob("*.lean"),
        *STUDY.glob("configs/*.json"),
        STUDY / "plan.json",
        RESULTS / "validation.json",
        *[ROOT / path for path in validation["source_sha256"]],
        *[ROOT / path for path in validation["evidence_sha256"]],
    ]
    for path in set(own):
        relative = str(path.relative_to(ROOT))
        committed = subprocess.run(  # noqa: S603 -- Read exact local committed bytes.
            ["/usr/bin/git", "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        require(committed == path.read_bytes(), f"Commit before training: {relative}")
