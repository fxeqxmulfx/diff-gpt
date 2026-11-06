"""Audit the full declared local scaling grid and preserve every proof attempt."""

from __future__ import annotations

import csv
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

from research.early99_reporting.round2_audit import read, require, verify_hash
from research.rms_initialization_scaling.check import EXPECTED, EXTERNAL
from research.rms_initialization_scaling.check_runtime import (
    RESULTS,
    ROOT,
    STUDY,
    sha256,
)


def main() -> None:
    """Keep all 96 cases and 1152 checks without selecting an initializer."""
    plan_path, proof_path, runtime_path = [
        STUDY / "plan.json",
        RESULTS / "guarantees.json",
        RESULTS / "runtime.json",
    ]
    declared, proof, runtime = map(read, [plan_path, proof_path, runtime_path])
    require(proof["plan_sha256"] == sha256(plan_path), "Changed proof declaration")
    for group in [declared["sources_sha256"], proof["source_sha256"]]:
        for path, digest in group.items():
            verify_hash(ROOT / path, digest)
    for path, digest in proof["external_source_sha256"].items():
        verify_hash(EXTERNAL / path, digest)
    for path, digest in runtime["sources_sha256"].items():
        verify_hash(ROOT / path, digest)
    require(
        proof["exit_code"] == 0
        and proof["warnings"] == 0
        and not proof["extra_axioms"]
        and set(proof["axioms"]) == EXPECTED
        and all(
            set(names) <= {"propext", "Classical.choice", "Quot.sound"}
            for names in proof["axioms"].values()
        ),
        "Incomplete proof",
    )
    grid, rows = declared["numerical_grid"], runtime["rows"]
    actual_keys = [
        (r["width"], r["requested_amplitude"], r["scale"], r["dtype"]) for r in rows
    ]
    expected_keys = set(
        itertools.product(
            grid["widths"],
            grid["amplitudes"],
            grid["positive_scales"],
            grid["dtypes"],
        )
    )
    require(
        runtime["complete"]
        and runtime["optimizer_updates"] == 0
        and len(actual_keys) == len(set(actual_keys)) == grid["expected_cases"]
        and set(actual_keys) == expected_keys,
        "Missing or duplicate numerical cases",
    )
    checks = [check for row in rows for check in row["checks"].values()]
    require(
        len(checks) == runtime["checks"]
        and all(
            c["passed"]
            == (
                c["finite"]
                and c["relative_max_error"] is not None
                and c["relative_max_error"] <= grid["relative_error_limit"]
            )
            for c in checks
        )
        and all(
            row["passed"] == all(c["passed"] for c in row["checks"].values())
            for row in rows
        )
        and runtime["failures"] == [r for r in rows if not r["passed"]]
        and runtime["passed"] == sum(r["passed"] for r in rows),
        "Inconsistent numerical outcome",
    )
    attempts = sorted(RESULTS.glob("lean-attempt-*.json"))
    theorem_source = (STUDY / "Bounds.lean").read_text().split("-- A concrete")[0]
    require(
        all(
            read(path)["source_text"][
                str((STUDY / "Bounds.lean").relative_to(ROOT))
            ].split("-- A concrete")[0]
            == theorem_source
            for path in attempts
        ),
        "The exported statements or proofs changed across attempts",
    )
    for field in ["attempt", "attempt_sha256"]:
        require(field in runtime and field in proof, "Missing attempt provenance")
    verify_hash(ROOT / runtime["attempt"], runtime["attempt_sha256"])
    verify_hash(ROOT / proof["attempt"], proof["attempt_sha256"])
    require(
        read(ROOT / runtime["attempt"])
        == {k: v for k, v in runtime.items() if k not in {"attempt", "attempt_sha256"}},
        "Changed raw numerical attempt",
    )
    table = [{k: v for k, v in row.items() if k != "checks"} for row in rows]
    table_path = RESULTS / "scaling-summary.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    paths = [
        Path(__file__),
        plan_path,
        proof_path,
        runtime_path,
        table_path,
        RESULTS / "runtime-output.txt",
        *attempts,
        ROOT / runtime["attempt"],
    ]
    record = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "lean_statements": len(EXPECTED),
        "lean_attempts": len(attempts),
        "failed_lean_attempts_retained": sum(
            any(c["returncode"] != 0 for c in read(path)["commands"])
            for path in attempts
        ),
        "exported_theorems_unchanged_across_attempts": True,
        "cases": len(rows),
        "passed": runtime["passed"],
        "checks": len(checks),
        "maximum_finite_relative_error": runtime["maximum_finite_relative_error"],
        "optimizer_updates": 0,
        "initializer_selected": False,
        "fixed75_selection_changed": False,
        "source_and_evidence_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths
        },
        "limits": declared["limits"],
    }
    (RESULTS / "complete-audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in {"source_and_evidence_sha256", "limits"}
            }
        )
    )


if __name__ == "__main__":
    main()
