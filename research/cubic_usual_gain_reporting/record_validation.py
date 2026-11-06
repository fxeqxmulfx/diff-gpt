"""Freeze CPU/CUDA evidence for the user-requested usual-gain retest."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from research.cubic_usual_gain.factories import ROOT, STUDY, plan
from research.cubic_usual_gain.readiness import (
    CUDA_CHECKS,
    VALIDATION_CHECKS,
    sha256,
    source_hashes,
)
from research.early99_reporting.round2_audit import read, require


def main() -> None:
    """Require the unchanged native CUDA/resume harness and all three paired seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    results = STUDY / "results"
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    cuda_passed = 0
    names = []
    evidence = [
        Path(__file__),
        ROOT / "tests/diff_gpt/application/test_cubic_usual_gain.py",
        ROOT / "tests/diff_gpt/application/test_cubic_capacity.py",
        results / "inventory.json",
    ]
    for mode in ["cpu"] if args.cpu_only else ["cpu", "cuda"]:
        path = results / f"validation-{mode}.xml"
        suites = ET.parse(path).getroot().findall("testsuite")  # noqa: S314 -- Trusted local test output.
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.attrib[key])
            for case in suite.findall("testcase"):
                names.append(case.attrib["classname"] + "." + case.attrib["name"])
                if mode == "cuda" and not list(case):
                    cuda_passed += 1
        evidence.extend([path, results / f"validation-{mode}.txt"])
    expected_checks = (
        VALIDATION_CHECKS - CUDA_CHECKS if args.cpu_only else VALIDATION_CHECKS
    )
    expected_cuda = 0 if args.cpu_only else CUDA_CHECKS
    require(
        totals == {"tests": expected_checks, "failures": 0, "errors": 0, "skipped": 0}
        and cuda_passed == expected_cuda,
        "Incomplete usual-gain validation",
    )
    inventory = read(results / "inventory.json")
    require(
        inventory["complete"]
        and len(inventory["rows"]) == len(plan()["seeds"])
        and {row["seed"] for row in inventory["rows"]} == set(plan()["seeds"])
        and all(row["optimizer_updates"] == 0 for row in inventory["rows"]),
        "Missing paired initialization",
    )
    record = {
        "complete": not args.cpu_only,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "passed": totals["tests"],
        "failed": 0,
        "skipped": 0,
        "cuda_tests_passed": cuda_passed,
        "test_names": names,
        "initializations_verified": len(inventory["rows"]),
        "source_sha256": source_hashes(),
        "evidence_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in evidence
        },
        "limits": (
            "Gain-only model and actual native update/resume checks; no convergence "
            "guarantee or repair of preexisting numerical stress failures."
        ),
    }
    name = "cpu-validation.json" if args.cpu_only else "validation.json"
    (results / name).write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": record["passed"],
                "cuda_tests_passed": cuda_passed,
                "source_files": len(record["source_sha256"]),
            }
        )
    )


if __name__ == "__main__":
    main()
