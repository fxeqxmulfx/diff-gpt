"""Reconstruct all reference vectors and retain every local gradient discrepancy."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

import torch

from research.early99_reporting.round2_audit import (
    read,
    require,
    verify_csv,
    verify_hash,
)
from research.layernorm_affine_guarantees.check import EXPECTED, EXTERNAL
from research.layernorm_affine_guarantees.compare import (
    OUTPUT,
    RESULTS,
    ROOT,
    STUDY,
    sha256,
)
from research.layernorm_affine_guarantees.reference import evaluate

KEYS = ("width", "input", "dtype", "normalizer", "affine_profile")


def declared_inputs(row: dict) -> dict:
    """Reconstruct represented inputs independently from the plan's scalar recipe."""
    width, dtype = row["width"], getattr(torch, row["dtype"])
    pattern = [1, -1, -1, 1]
    direction = torch.tensor(
        [pattern[i % len(pattern)] for i in range(width)], dtype=dtype
    )
    amplitude, shift = {
        "zero": (0.0, 0.0),
        "constant_3": (0.0, 3.0),
        "balanced_0.02": (0.02, 0.0),
        "balanced_1": (1.0, 0.0),
        "shifted_balanced_0.02": (0.02, 3.0),
    }[row["input"]]
    value = direction * amplitude + shift
    result = {"input": value.tolist(), "upstream": direction.tolist()}
    if row["affine_profile"] == "nonuniform":
        result.update(
            gamma=torch.linspace(-2, 2, width, dtype=dtype).tolist(),
            beta=torch.linspace(-0.25, 0.25, width, dtype=dtype).tolist(),
        )
    return result


def audit_comparison(check: dict, reference: list[Decimal], threshold: float) -> dict:
    """Recompute whole-vector error with scalar sums, separately from Torch metrics."""
    expected = [float(v) for v in reference]
    require(check["reference"] == expected, "Changed independent reference values")
    actual = [float(v) for v in check["actual"]]
    require(len(actual) == len(expected), "Incomplete observed vector")
    finite = all(math.isfinite(v) for v in actual)
    require(all(math.isfinite(v) for v in expected), "Nonfinite independent reference")
    reference_norm = math.sqrt(math.fsum(v * v for v in expected))
    error_norm = (
        math.sqrt(
            math.fsum((a - b) ** 2 for a, b in zip(actual, expected, strict=True))
        )
        if finite
        else None
    )
    relative = (
        error_norm / reference_norm
        if finite and reference_norm > 0
        else (0.0 if error_norm == 0 else None)
    )
    for key, value in [
        ("reference_l2_norm", reference_norm),
        ("absolute_l2_error", error_norm),
        ("relative_l2_error", relative),
    ]:
        require(
            check[key] is None
            if value is None
            else math.isclose(check[key], value, rel_tol=2e-12, abs_tol=0),
            f"Changed full-vector metric: {key}",
        )
    passed = finite and relative is not None and relative <= threshold
    require(
        passed == check["passed"]
        and check["actual_finite"] == finite
        and check["reference_finite"],
        "Changed numerical pass criterion",
    )
    return {
        "passed": passed,
        "actual_finite": finite,
        "reference_is_zero": reference_norm == 0,
        "reference_l2_norm": reference_norm,
        "absolute_l2_error": error_norm,
        "relative_l2_error": relative,
        "maximum_absolute_component_error": max(
            abs(a - b) for a, b in zip(actual, expected, strict=True)
        )
        if finite
        else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Retain all declared cases and metrics with stable LF line endings."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    verify_csv(path, rows)


def main() -> None:
    """Close this comparison without selecting a model or erasing numerical misses."""
    declared, proof = read(STUDY / "plan.json"), read(RESULTS / "guarantees.json")
    measurements = read(RESULTS / "measurements.json")
    require(measurements == read(OUTPUT / "measurements.json"), "Changed raw results")
    require(
        measurements["manifest"] == read(OUTPUT / "manifest.json"), "Changed archive"
    )
    require(
        measurements["complete"] and measurements["optimizer_updates"] == 0,
        "Unfinished or out-of-scope comparison",
    )
    manifest = measurements["manifest"]
    for path, digest in manifest["source_and_evidence_sha256"].items():
        verify_hash(ROOT / path, digest)
        verify_hash(OUTPUT / "source" / path, digest)
    for path, digest in proof["external_source_sha256"].items():
        verify_hash(EXTERNAL / path, digest)
    require(
        set(proof["axioms"]) == EXPECTED
        and proof["warnings"] == 0
        and proof["exit_code"] == 0
        and not proof["extra_axioms"]
        and all(
            set(a) <= {"propext", "Classical.choice", "Quot.sound"}
            for a in proof["axioms"].values()
        ),
        "Incomplete real-arithmetic proofs",
    )
    grid, rows = declared["numerical_grid"], measurements["rows"]
    expected_keys = set(
        itertools.product(
            grid["widths"],
            grid["inputs"],
            grid["dtypes"],
            grid["normalizers"],
            grid["affine_profiles"],
        )
    )
    actual_keys = [tuple(row[k] for k in KEYS) for row in rows]
    require(
        len(rows) == len(set(actual_keys)) == grid["expected_cases"]
        and set(actual_keys) == expected_keys,
        "Missing or duplicate local cases",
    )
    metrics, cells = [], []
    artifact_paths = [OUTPUT / "manifest.json", OUTPUT / "measurements.json"]
    for index, row in enumerate(rows, start=1):
        raw = OUTPUT / f"cell-{index:03}.json"
        require(read(raw) == row, "Changed individual case")
        artifact_paths.append(raw)
        require(row["execution_error"] is None, "Retained execution error needs review")
        inputs = declared_inputs(row)
        require(
            row["represented_inputs"] == inputs,
            "Changed represented input/affine values",
        )
        expected_parameters = (
            2 * row["width"] if row["affine_profile"] == "nonuniform" else 0
        )
        require(
            row["affine_parameters"] == expected_parameters, "Changed affine capacity"
        )
        with localcontext() as context:
            context.prec = 80
            decimal = {k: [Decimal(v) for v in values] for k, values in inputs.items()}
            reference = evaluate(
                decimal["input"],
                Decimal(grid["epsilon"]["numerator"])
                / Decimal(grid["epsilon"]["denominator"]),
                row["normalizer"],
                decimal["upstream"],
                decimal.get("gamma"),
                decimal.get("beta"),
            )
            require(
                set(reference) == set(row["comparisons"]), "Missing gradient or output"
            )
            group = [
                {
                    **{k: row[k] for k in KEYS},
                    "metric": name,
                    **audit_comparison(
                        check, reference[name], grid["relative_l2_error_threshold"]
                    ),
                }
                for name, check in row["comparisons"].items()
            ]
        require(row["passed"] == all(r["passed"] for r in group), "Changed cell result")
        metrics.extend(group)
        cells.append(
            {
                **{k: row[k] for k in KEYS},
                "passed": row["passed"],
                "failed_metrics": ",".join(
                    r["metric"] for r in group if not r["passed"]
                ),
                "affine_parameters": expected_parameters,
            }
        )
    failures = [r for r in metrics if not r["passed"]]
    require(
        measurements["cases"] == len(cells)
        and measurements["execution_errors"] == 0
        and measurements["passed"] == sum(r["passed"] for r in cells)
        and measurements["numerical_failures"] == sum(not r["passed"] for r in cells),
        "Changed full-grid summary",
    )
    grouped = defaultdict(list)
    for row in cells:
        grouped[(row["normalizer"], row["dtype"], row["affine_profile"])].append(row)
    groups = [
        {
            "normalizer": k[0],
            "dtype": k[1],
            "affine_profile": k[2],
            "cases": len(v),
            "passed": sum(r["passed"] for r in v),
        }
        for k, v in grouped.items()
    ]
    report_paths = []
    for name, records in [("cells", cells), ("metrics", metrics), ("groups", groups)]:
        path = RESULTS / f"{name}.csv"
        write_csv(path, records)
        report_paths.append(path)
    summary = {
        "complete": True,
        "cases": len(cells),
        "passed": measurements["passed"],
        "execution_errors": 0,
        "comparisons": len(metrics),
        "comparisons_passed": sum(r["passed"] for r in metrics),
        "all_forward_outputs_pass": all(
            r["passed"] for r in metrics if r["metric"] == "output"
        ),
        "groups": groups,
        "failed_comparisons": failures,
        "optimizer_updates": 0,
        "normalizer_selected": False,
        "fixed75_selection_changed": False,
        "limits": declared["limits"],
    }
    summary_path = RESULTS / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_paths.append(summary_path)
    proof_attempts = sorted(RESULTS.glob("lean-attempt-*.json"))
    record = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "cases_audited": len(cells),
        "complete_vectors_recomputed": len(metrics),
        "lean_target_statements": len(EXPECTED),
        "proof_attempts": [
            {
                "path": str(path.relative_to(ROOT)),
                "command_exit_codes": [c["returncode"] for c in read(path)["commands"]],
            }
            for path in proof_attempts
        ],
        "source_and_evidence_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                Path(__file__),
                RESULTS / "measurements.json",
                RESULTS / "comparison-output.txt",
                *report_paths,
                *proof_attempts,
            ]
        },
        "raw_artifacts_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in artifact_paths
        },
        "scope": (
            "Local implementations and real-arithmetic bounds; "
            "no training or model selection."
        ),
    }
    (RESULTS / "complete-audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "cases",
                    "passed",
                    "comparisons",
                    "comparisons_passed",
                    "all_forward_outputs_pass",
                ]
            }
        )
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
