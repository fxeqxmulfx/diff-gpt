"""Bind the complete CPU/CUDA validation and 27-model inventory to exact sources."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from research.cubic_capacity.factories import ROOT, STUDY, plan
from research.cubic_capacity.readiness import (
    CUDA_TESTS,
    VALIDATION_TESTS,
    sha256,
    source_hashes,
)
from research.early99_reporting.round2_audit import require


def main() -> None:
    """Reject missing/skipped checks and freeze their evidence before training."""
    results = STUDY / "results"
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    test_names, cuda_passed = [], 0
    evidence = [
        Path(__file__),
        results / "inventory.json",
        ROOT / "tests/diff_gpt/application/test_cubic_capacity.py",
        ROOT / "tests/diff_gpt/application/test_early99_benchmark.py",
        ROOT / "tests/diff_gpt/application/test_early99_execution_journal.py",
    ]
    for mode in ["cpu", "cuda"]:
        xml_path = results / f"validation-{mode}.xml"
        document = ET.parse(xml_path).getroot()  # noqa: S314 -- Locally generated JUnit.
        for suite in document.findall("testsuite"):
            for key in counts:
                counts[key] += int(suite.attrib[key])
            for case in suite.findall("testcase"):
                test_names.append(case.attrib["classname"] + "." + case.attrib["name"])
                if mode == "cuda" and not list(case):
                    cuda_passed += 1
        evidence.extend([xml_path, results / f"validation-{mode}.txt"])
    require(
        counts == {"tests": VALIDATION_TESTS, "failures": 0, "errors": 0, "skipped": 0},
        f"Incomplete validation: {counts}",
    )
    require(cuda_passed == CUDA_TESTS, "The actual CUDA cases did not pass")
    inventory = json.loads((results / "inventory.json").read_text())
    search = plan()
    expected = {
        (config["name"], seed)
        for config in search["configurations"]
        for seed in search["seeds"]
    }
    require(
        inventory["complete"]
        and inventory["optimizer_updates"] == 0
        and len(inventory["rows"]) == len(expected)
        and {(row["name"], row["seed"]) for row in inventory["rows"]} == expected,
        "Incomplete initialization inventory",
    )
    record = {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "passed": counts["tests"],
        "failed": 0,
        "skipped": 0,
        "cuda_tests_passed": cuda_passed,
        "initializations_verified": len(expected),
        "test_names": test_names,
        "source_sha256": source_hashes(),
        "evidence_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in evidence
        },
        "limits": (
            "Native compiled/captured update parity and real checkpoint continuation "
            "are tested locally at the largest shape. These checks do not establish "
            "training convergence or repair the ten existing numerical stress failures."
        ),
    }
    (results / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in record.items()
                if key not in {"source_sha256", "evidence_sha256", "test_names"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
