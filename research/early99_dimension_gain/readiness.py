"""Require committed pretraining evidence before any new benchmark update."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
TEST = ROOT / "tests/diff_gpt/application/test_early99_dimension_gain.py"
RESULTS = STUDY / "results"
EXPECTED_TESTS = 13
CUDA_TESTS = 2
PROOF_STATEMENTS = 4


def read(path: Path) -> dict:
    """Read an explicit local scientific record."""
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    """Bind evidence to exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:  # noqa: FBT001 -- Assertion helper.
    """Reject incomplete or changed evidence even with optimized Python."""
    if not condition:
        raise ValueError(message)


def verify_training_ready() -> None:
    """Check declared controls, all validation rows, checked proofs and Git objects."""
    plan = read(STUDY / "plan.json")
    for name, expected in plan["evidence_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Changed prior evidence: {name}")
    short = plan["short_evidence"]
    require(
        sha256(ROOT / short["evidence_file"]) == short["evidence_sha256"],
        "Changed short-only motivation",
    )
    for configuration in plan["configurations"]:
        require(
            read(STUDY / "configs" / f"{configuration['name']}.json") == configuration,
            "Configuration differs from the frozen plan",
        )
    validation = read(RESULTS / "validation.json")
    require(
        validation["exit_code"] == 0
        and validation["passed"] == EXPECTED_TESTS
        and validation["skipped"] == 0
        and validation["cuda_tests_passed"] == CUDA_TESTS,
        "Incomplete CPU/CUDA validation",
    )
    require(sha256(TEST) == validation["test_source_sha256"], "Validated test changed")
    for field in ("source_sha256", "evidence_sha256"):
        for name, expected in validation[field].items():
            require(
                sha256(ROOT / name) == expected, f"Validated evidence changed: {name}"
            )
    stress = read(RESULTS / "score-stress.json")
    grid = plan["score_stress"]
    expected = set(
        itertools.product(
            grid["head_widths"], grid["visible_keys"], grid["raw_qk_scales"]
        )
    )
    keys = [(r["head_width"], r["visible_keys"], r["qk_scale"]) for r in stress["rows"]]
    require(
        len(keys) == len(set(keys)) == grid["expected_cases"]
        and set(keys) == expected
        and stress["passed"] == grid["expected_cases"]
        and not stress["failures"]
        and stress["plan_sha256"] == sha256(STUDY / "plan.json"),
        "Incomplete or changed score stress grid",
    )
    for name, expected_hash in stress["source_sha256"].items():
        require(sha256(Path(name)) == expected_hash, "Stress implementation changed")
    require(
        all(
            all(
                row[k]
                for k in (
                    "finite",
                    "qk_rms_retained",
                    "within_declared_bound_slack",
                    "accurate_probability",
                    "accurate_gradients",
                )
            )
            for row in stress["rows"]
        ),
        "A declared score probe failed",
    )
    proof = read(RESULTS / "guarantees.json")
    require(
        proof["exit_code"] == 0
        and proof["warnings"] == 0
        and proof["statements"] == PROOF_STATEMENTS
        and not proof["extra_axioms"],
        "Missing checked mathematical specializations",
    )
    for name, expected_hash in proof["source_sha256"].items():
        require(sha256(ROOT / name) == expected_hash, "Checked proof source changed")
    own = [
        *STUDY.glob("*.py"),
        *STUDY.glob("*.lean"),
        *STUDY.glob("configs/*.json"),
        STUDY / "plan.json",
        TEST,
        RESULTS / "validation.json",
        *[ROOT / name for name in validation["evidence_sha256"]],
    ]
    for path in own:
        relative = str(path.relative_to(ROOT))
        committed = subprocess.run(  # noqa: S603 -- Read exact local Git objects.
            ["/usr/bin/git", "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        require(committed == path.read_bytes(), f"Commit before training: {relative}")
