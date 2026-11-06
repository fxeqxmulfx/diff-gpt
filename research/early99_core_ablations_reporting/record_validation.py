"""Bind the completed implementation checks and retained failures to exact sources."""

from __future__ import annotations

import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.early99_core_ablations.readiness import CUDA_TESTS, EXPECTED_TESTS
from research.early99_core_ablations.run import Configuration, prepare_output
from research.early99_core_ablations_reporting.run_screens import RESULTS, ROOT


def digest(path: Path) -> str:
    """Hash checked local bytes, including complete failure logs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path: Path, expected: int) -> list:
    """Reject missing, skipped, failed or errored validation cases."""
    cases = list(ET.parse(path).getroot().iter("testcase"))  # noqa: S314 -- Local pytest output.
    if len(cases) != expected or any(len(case) for case in cases):
        message = f"Incomplete validation: {path}"
        raise ValueError(message)
    return cases


def main() -> None:
    """Record existing measurements; this never reruns training or a proof."""
    checked(RESULTS / "validation.xml", EXPECTED_TESTS)
    cuda = checked(RESULTS / "validation-cuda-final.xml", CUDA_TESTS)
    if any("test_cuda_" not in case.attrib["name"] for case in cuda):
        message = "Final validation omitted CUDA paths"
        raise ValueError(message)
    with tempfile.TemporaryDirectory(prefix="early99-core-source-") as directory:
        sources = prepare_output(
            Path(directory), Configuration(), "cpu", compiled=False
        )["source_sha256"]
    extra = [
        ROOT / "tests/diff_gpt/application/test_early99_core_ablations.py",
        ROOT / "tests/diff_gpt/application/test_early99_benchmark.py",
        ROOT / "tests/diff_gpt/application/test_early99_execution_journal.py",
        ROOT / "research/early99_optimizer_followups/parameter-inventory.json",
        Path(__file__),
        Path(__file__).with_name("diagnose_cuda.py"),
    ]
    sources.update({str(p.relative_to(ROOT)): digest(p) for p in extra})
    evidence = [
        *RESULTS.glob("validation*.txt"),
        *RESULTS.glob("validation*.xml"),
        *RESULTS.glob("lean-attempt-*.txt"),
        RESULTS / "guarantees.json",
        RESULTS / "cuda-rounding-diagnostic.json",
        RESULTS / "cuda-rounding-diagnostic.txt",
    ]
    record = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "exit_code": 0,
        "passed": EXPECTED_TESTS,
        "skipped": 0,
        "cuda_tests_passed": CUDA_TESTS,
        "source_sha256": sources,
        "evidence_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in sorted(evidence)
        },
        "torch": str(torch.__version__),
        "command": (
            "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest "
            "-q tests/diff_gpt/application/test_early99_execution_journal.py "
            "tests/diff_gpt/application/test_early99_benchmark.py "
            "tests/diff_gpt/application/test_early99_core_ablations.py"
        ),
        "final_cuda_command": (
            "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/pytest "
            "-q tests/diff_gpt/application/test_early99_core_ablations.py -k "
            "cuda"
        ),
        "post_full_run_cleanup": (
            "Split one compound finite-value assertion into two assertions. "
            "Reran all three affected CUDA cases after that style cleanup; no "
            "implementation change."
        ),
        "scope": (
            "18 architecture checks, 6 existing journal checks and 9 benchmark "
            "integrity checks. All35 task shapes preserve original weights, "
            "parameter order/count, buffers and RNG. Independent full "
            "forward/all parameter gradients at head widths8and32, actual FP32 "
            "epsilon and clipped XSA. Cache/singleton/causality/tied readout. "
            "Three CUDA mechanisms: graph versus uncaptured compiled native "
            "AdamW at unchanged elementwise tolerances, separate compiled "
            "versus uncompiled per-tensor relativeL2<=0.001, zero-rate first "
            "step, tails, RNG/moments and nonempty-state restoration. Three "
            "exact CPU full-trainer continuations."
        ),
        "initial_failures_preserved": (
            "The original CUDA test conflated compiler rounding with graph "
            "capture; its failed Peri-RMS momentum comparison and "
            "unchanged13-update diagnostic remain. Four Lean theorem proofs "
            "checked on the first attempt; two concrete numeral examples "
            "required elaboration fixes, with the complete failed compiler log "
            "retained. These validation issues changed no scientific model "
            "code."
        ),
        "limitations": (
            "A local numerical update comparison does not prove long-run "
            "compiled/eager trajectory equivalence or repair the separate "
            "full-stack numerical stress failures."
        ),
    }
    (RESULTS / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    checked(RESULTS / "selection-validation.xml", 5)
    selection_sources = [
        Path(__file__).with_name("selection.py"),
        Path(__file__).with_name("long_metrics.py"),
        ROOT / "tests/diff_gpt/application/test_early99_core_selection.py",
    ]
    selection_evidence = [
        RESULTS / f"selection-validation.{suffix}" for suffix in ("txt", "xml")
    ]
    selection = {
        "exit_code": 0,
        "passed": 5,
        "skipped": 0,
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in selection_sources
        },
        "evidence_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in selection_evidence
        },
        "scope": (
            "Five unchanged adversarial selection tests, rebound to this "
            "study: replay interval ambiguity, candidate overlap, ties, "
            "failed/incomplete grids, and censored long comparison counts."
        ),
    }
    (RESULTS / "selection-validation.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "implementation_checks": EXPECTED_TESTS,
                "cuda_checks": CUDA_TESTS,
                "selection_checks": 5,
                "sources_pinned": len(sources),
            }
        )
    )


if __name__ == "__main__":
    main()
