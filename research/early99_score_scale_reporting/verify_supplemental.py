"""Verify supplemental proof records without rewriting the frozen selection."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from research.early99_reporting.round2_audit import read, require, verify_hash
from research.early99_score_scale_reporting.run_screens import ROOT
from research.early99_score_scale_reporting.verify_results import digest


def verify() -> dict:
    """Bind separately observed kernel checks to unchanged sources and dependencies."""
    folder = ROOT / "research/normalization_guarantees/results"
    artifacts = set()
    statements = {}
    records = {
        "guarantees.json": 6,
        "attention-sensitivity.json": 4,
        "improved-sensitivity.json": 7,
    }
    for filename, expected_count in records.items():
        path = folder / filename
        record = read(path)
        artifacts.add(path)
        external = Path(record["external_project"])
        for directory, key in [
            (external, "external_commit"),
            (external / ".lake/packages/mathlib", "mathlib_revision"),
        ]:
            revision = subprocess.run(  # noqa: S603 -- Read fixed local Git revisions.
                ["/usr/bin/git", "-C", str(directory), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            require(revision == record[key], "Proof dependency revision changed")
        require(
            (external / "lean-toolchain").read_text().strip()
            == record["lean_toolchain"],
            "Lean toolchain changed",
        )
        for key, base in [
            ("local_dependency_sha256", ROOT),
            ("external_source_sha256", external),
            ("implementation_source_sha256", ROOT),
            ("reviewed_implementation_sha256", ROOT),
        ]:
            for relative, sha in record.get(key, {}).items():
                verify_hash(base / relative, sha)
        observed = {}
        for check in record.get("checks", [record]):
            fixture = ROOT / check["fixture"]
            output = (
                ROOT / check["output"]
                if "output" in check
                else folder / "lean-check.txt"
            )
            verify_hash(fixture, check["fixture_sha256"])
            verify_hash(
                output, check.get("output_sha256", check.get("lean_output_sha256"))
            )
            require(
                check["exit_code"] == 0
                and check["warnings"] == 0
                and not check["extra_axioms"],
                "Unsuccessful recorded proof check",
            )
            text = output.read_text()
            require(
                "error:" not in text and "warning:" not in text,
                "Proof output contains a diagnostic",
            )
            parsed = {
                name: [a.strip() for a in names.split(",")]
                for name, names in re.findall(
                    r"'([^']+)' depends on axioms: \[([^]]*)\]", text
                )
            }
            require(
                parsed == check["axioms"], "Recorded axioms differ from checker output"
            )
            require(
                all(
                    set(items) <= {"propext", "Classical.choice", "Quot.sound"}
                    for items in parsed.values()
                ),
                "Nonstandard proof axiom",
            )
            require(
                not set(parsed).intersection(observed), "Duplicate checked statement"
            )
            observed.update(parsed)
            artifacts.update({fixture, output})
        require(
            len(observed) == expected_count, "Missing supplemental proof statements"
        )
        statements[filename] = sorted(observed)
    return {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "checked_statements": statements,
        "total_statements": sum(map(len, statements.values())),
        "files_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in sorted(artifacts)
        },
        "verifier_sha256": digest(Path(__file__)),
        "proof_compilation_repeated": False,
        "scientific_experiments_or_selection_changed": False,
        "scope": "Verify records of separately observed successful Lean checks, their "
        "actual printed axioms, source/dependency hashes and pinned toolchain. "
        "This does not prove implementation equivalence or learning success.",
    }


if __name__ == "__main__":
    result = verify()
    destination = Path(__file__).with_name("results") / "supplemental-proof-audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "files_sha256"}, indent=2)
    )
