"""Measure all declared local normalization cases against a Decimal oracle."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import traceback
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.rms_norm import RMSNorm
from research.early99_reporting.round2_audit import read, require, verify_hash
from research.layernorm_affine_guarantees.reference import RMS, evaluate

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
OUTPUT = ROOT / "runs/layernorm-affine-guarantees/local-comparison-v1"


def sha256(path: Path) -> str:
    """Bind every comparison to the exact declared and executed bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_committed(paths: list[Path]) -> None:
    """Freeze local equations, oracle and validation before any new measurement."""
    for path in paths:
        committed = subprocess.run(  # noqa: S603 -- Fixed local git read.
            ["/usr/bin/git", "show", f"HEAD:{path.relative_to(ROOT)}"],
            check=True,
            capture_output=True,
        ).stdout
        require(committed == path.read_bytes(), f"Commit source/evidence first: {path}")


def readiness() -> dict:
    """Require the eight proofs and all independent reference derivative checks."""
    plan_path, proof_path = STUDY / "plan.json", RESULTS / "guarantees.json"
    declared, proof = read(plan_path), read(proof_path)
    require(
        proof["exit_code"] == 0
        and proof["warnings"] == 0
        and proof["statements"] == len(declared["proof_targets"])
        and not proof["extra_axioms"]
        and proof["plan_sha256"] == sha256(plan_path),
        "Complete the normalization proofs first",
    )
    for group in [declared["sources_sha256"], proof["source_sha256"]]:
        for path, digest in group.items():
            verify_hash(ROOT / path, digest)
    for path, digest in declared["torch_python_source_sha256"].items():
        verify_hash(Path(path), digest)
    require(
        str(torch.__version__) == declared["torch_version"]
        and torch.version.git_version == declared["torch_git_version"],
        "Changed native implementation version",
    )
    xml = RESULTS / "reference-validation.xml"
    suites = ET.parse(xml).getroot().findall("testsuite")  # noqa: S314 -- Trusted local test output.
    totals = {
        key: sum(int(s.attrib[key]) for s in suites)
        for key in ["tests", "failures", "errors", "skipped"]
    }
    require(
        totals == {"tests": 3, "failures": 0, "errors": 0, "skipped": 0},
        "Incomplete independent-reference validation",
    )
    paths = {
        *STUDY.glob("*.py"),
        *STUDY.glob("*.lean"),
        plan_path,
        proof_path,
        xml,
        RESULTS / "reference-validation.txt",
        ROOT / "tests/diff_gpt/application/test_layernorm_affine_reference.py",
        *[ROOT / path for path in declared["sources_sha256"]],
    }
    require_committed(list(paths))
    return {
        "source_and_evidence_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)
        },
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "independent_reference_checks_passed": totals["tests"],
        "proof_statements": proof["statements"],
    }


def compare_vector(
    actual: torch.Tensor, reference: list[Decimal], limit: float
) -> dict:
    """Keep full vectors, including every tiny-gradient or nonfinite discrepancy."""
    measured = actual.detach().double().flatten()
    expected = torch.tensor([float(v) for v in reference], dtype=torch.float64)
    require(torch.isfinite(expected).all().item(), "Nonfinite Decimal reference")
    finite = torch.isfinite(measured).all().item()
    denominator = torch.linalg.vector_norm(expected).item()
    difference = (
        torch.linalg.vector_norm(measured - expected).item() if finite else None
    )
    error = (
        difference / denominator
        if finite and denominator > 0
        else (0.0 if difference == 0 else None)
    )
    return {
        "actual": [x if math.isfinite(x) else str(x) for x in measured.tolist()],
        "reference": expected.tolist(),
        "reference_l2_norm": denominator,
        "absolute_l2_error": difference,
        "relative_l2_error": error,
        "actual_finite": finite,
        "reference_finite": True,
        "passed": error is not None and error <= limit,
    }


def measure(
    width: int, name: str, dtype: str, normalizer: str, profile: str, grid: dict
) -> dict:
    """Check one local native operation; no task data or optimizer is involved."""
    kind = getattr(torch, dtype)
    direction = torch.tensor([1, -1, -1, 1], dtype=kind).repeat((width + 3) // 4)[
        :width
    ]
    if name == "zero":
        value = torch.zeros(width, dtype=kind)
    elif name == "constant_3":
        value = torch.full((width,), 3, dtype=kind)
    elif name == "balanced_0.02":
        value = 0.02 * direction
    elif name == "balanced_1":
        value = direction.clone()
    elif name == "shifted_balanced_0.02":
        value = 3 + 0.02 * direction
    else:
        raise ValueError(name)
    value.requires_grad_()
    eps = grid["epsilon"]["numerator"] / grid["epsilon"]["denominator"]
    gamma = beta = None
    if profile == "nonuniform":
        gamma = torch.linspace(-2, 2, width, dtype=kind, requires_grad=True)
        beta = torch.linspace(-0.25, 0.25, width, dtype=kind, requires_grad=True)
    else:
        require(profile == "none", "Unknown affine profile")
    if normalizer == RMS:
        norm = RMSNorm(width)
        norm.eps = eps
        output = norm(value)
        if gamma is not None:
            output = output * gamma + beta
    else:
        output = F.layer_norm(value, (width,), gamma, beta, eps)
    inputs = (value, gamma, beta) if gamma is not None else (value,)
    derivatives = torch.autograd.grad(output, inputs, grad_outputs=direction)
    actual = {"output": output, "input_gradient": derivatives[0]}
    if gamma is not None:
        actual.update(gamma_gradient=derivatives[1], beta_gradient=derivatives[2])
    raw = {"input": value.tolist(), "upstream": direction.tolist()}
    if gamma is not None:
        raw.update(gamma=gamma.tolist(), beta=beta.tolist())
    with localcontext() as context:
        context.prec = 80
        decimal = {key: [Decimal(x) for x in vector] for key, vector in raw.items()}
        reference = evaluate(
            decimal["input"],
            Decimal(eps),
            normalizer,
            decimal["upstream"],
            decimal.get("gamma"),
            decimal.get("beta"),
        )
        comparisons = {
            key: compare_vector(
                tensor, reference[key], grid["relative_l2_error_threshold"]
            )
            for key, tensor in actual.items()
        }
    return {
        "width": width,
        "input": name,
        "dtype": dtype,
        "normalizer": normalizer,
        "affine_profile": profile,
        "affine_parameters": 0 if gamma is None else 2 * width,
        "represented_inputs": raw,
        "comparisons": comparisons,
        "passed": all(c["passed"] for c in comparisons.values()),
        "execution_error": None,
    }


def main() -> None:
    """Preserve the full grid, with numerical misses as outcomes and errors explicit."""
    manifest = readiness()
    declared = read(STUDY / "plan.json")
    grid = declared["numerical_grid"]
    require(
        not OUTPUT.exists() and not (RESULTS / "measurements.json").exists(),
        "Preserve the existing comparison before any rerun",
    )
    OUTPUT.mkdir(parents=True)
    for path in manifest["source_and_evidence_sha256"]:
        target = OUTPUT / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    keys = itertools.product(
        grid["widths"],
        grid["inputs"],
        grid["dtypes"],
        grid["normalizers"],
        grid["affine_profiles"],
    )
    for width, name, dtype, normalizer, profile in keys:
        try:
            row = measure(width, name, dtype, normalizer, profile, grid)
        except Exception:  # noqa: BLE001 -- Preserve all cells and fail after recording.
            row = {
                "width": width,
                "input": name,
                "dtype": dtype,
                "normalizer": normalizer,
                "affine_profile": profile,
                "passed": False,
                "execution_error": traceback.format_exc(),
            }
        rows.append(row)
        (OUTPUT / f"cell-{len(rows):03}.json").write_text(
            json.dumps(row, indent=2, allow_nan=False) + "\n"
        )
    require(len(rows) == grid["expected_cases"], "Incomplete numerical grid")
    errors = sum(row["execution_error"] is not None for row in rows)
    record = {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "execution_errors": errors,
        "numerical_failures": sum(not row["passed"] for row in rows) - errors,
        "optimizer_updates": 0,
        "manifest": manifest,
        "rows": rows,
        "limits": declared["limits"],
    }
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    (OUTPUT / "measurements.json").write_text(payload)
    (RESULTS / "measurements.json").write_text(payload)
    print(
        json.dumps(
            {
                k: record[k]
                for k in [
                    "complete",
                    "cases",
                    "passed",
                    "execution_errors",
                    "numerical_failures",
                ]
            }
        )
    )
    require(
        errors == 0, "Retained execution errors; do not interpret as numerical misses"
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    main()
