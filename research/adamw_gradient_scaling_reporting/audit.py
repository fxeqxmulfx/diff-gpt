"""Reconstruct every synthetic input, Decimal reference and reported error."""

from __future__ import annotations

import csv
import itertools
import json
import math
import struct
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from pathlib import Path

from research.adamw_gradient_scaling.reference import decimal_trace, represented_trace
from research.adamw_gradient_scaling.run import FIELDS, RESULTS, RUN, STUDY, readiness
from research.early99_reporting.round2_audit import read, require, verify_hash
from research.stored_rope_guarantees.check import ROOT, digest


def represented(value: float, dtype: str) -> float:
    """Independently round declared inputs with the IEEE binary32 storage format."""
    return (
        struct.unpack("f", struct.pack("f", value))[0] if dtype == "float32" else value
    )


def rebuild_inputs(row: dict, grid: dict) -> None:
    """Check every input and rate against the frozen declaration without Torch."""
    dtype, scale, profile = row["dtype"], row["scale"], row["profile"]
    initial = [represented(v, dtype) for v in grid["initial_parameters"]]
    require(row["initial"] == initial, "Changed initial parameters")
    base = [represented(v, dtype) for v in grid["gradient_base"]]
    gradients = []
    quarter = grid["steps"] // 4
    for index in range(grid["steps"]):
        if profile == "constant":
            factor = 1.0
        elif profile == "alternating":
            factor = 1.0 if index % 2 == 0 else -0.5
        else:
            phase = index // quarter
            factor = [0.0, 1.0, -1.0, 0.0][phase]
        gradients.append([represented(factor * v, dtype) for v in base])
    require(
        row["gradients"]
        == {
            "counterpart": gradients,
            "scaled": [[represented(scale * v, dtype) for v in g] for g in gradients],
        },
        "Changed supplied history",
    )
    require(
        row["rates"]
        == [
            grid["peak_learning_rate"] * min(1.0, k / grid["warmup_completed_updates"])
            for k in range(grid["steps"])
        ]
        and row["epsilons"]
        == {"scaled": grid["epsilon"], "counterpart": grid["epsilon"] / scale}
        and row["rng_unchanged"],
        "Changed rates, epsilons or RNG",
    )


def independent_error(actual: list, reference: list, limit: float) -> dict:
    """Compute scalar squared-error sums independently of the measurement helper."""
    a, b = [float(v) for v in actual], [float(v) for v in reference]
    finite = all(math.isfinite(v) for v in [*a, *b])
    if not finite:
        return {
            "finite": False,
            "max_absolute": None,
            "relative_l2": None,
            "passed": False,
        }
    delta = [x - y for x, y in zip(a, b, strict=True)]
    absolute = max(abs(v) for v in delta)
    square = math.fsum(v * v for v in b)
    error = math.sqrt(math.fsum(v * v for v in delta))
    relative = error / math.sqrt(square) if square else (0.0 if error == 0 else None)
    return {
        "finite": True,
        "max_absolute": absolute,
        "relative_l2": relative,
        "passed": relative is not None and relative <= limit,
    }


def verify_case(row: dict, grid: dict) -> list[dict]:
    """Reconstruct reference states and all nine comparisons at every update."""
    rebuild_inputs(row, grid)
    expected_clocks = list(range(1, grid["steps"] + 1))
    for arm in ("scaled", "counterpart"):
        expected = represented_trace(
            decimal_trace(
                row["initial"],
                row["gradients"][arm],
                row["rates"],
                grid["betas"],
                row["epsilons"][arm],
                grid["weight_decay"],
            )
        )
        require(row["reference"][arm] == expected, "Changed independent reference")
        native = row["native"][arm]
        require([r["clock"] for r in native] == expected_clocks, "Missing update")
        require(
            native[0]["parameter"] == row["initial"], "Zero-rate step moved weights"
        )
        require(
            all(len(r[f]) == len(row["initial"]) for r in native for f in FIELDS),
            "Missing native coordinates",
        )
    metrics = []
    for step in range(grid["steps"]):
        for arm in ("scaled", "counterpart"):
            for field in FIELDS:
                metrics.append(  # noqa: PERF401 -- Keep all comparison axes explicit.
                    {
                        "clock": step + 1,
                        "kind": "decimal_reference",
                        "arm": arm,
                        "field": field,
                        **independent_error(
                            row["native"][arm][step][field],
                            row["reference"][arm][step][field],
                            grid["relative_error_limit"],
                        ),
                    }
                )
        for field, factor in zip(
            FIELDS, [1, row["scale"], row["scale"] ** 2], strict=True
        ):
            metrics.append(
                {
                    "clock": step + 1,
                    "kind": "scaling_identity",
                    "arm": "paired",
                    "field": field,
                    **independent_error(
                        row["native"]["scaled"][step][field],
                        [
                            factor * float(v)
                            for v in row["native"]["counterpart"][step][field]
                        ],
                        grid["relative_error_limit"],
                    ),
                }
            )
    require(
        len(metrics) == len(row["comparisons"]) == grid["comparisons_per_case"],
        "Comparison coverage changed",
    )
    for expected, saved in zip(metrics, row["comparisons"], strict=True):
        for key, value in expected.items():
            require(
                math.isclose(saved[key], value, rel_tol=1e-13, abs_tol=0)
                if key == "relative_l2" and value is not None
                else saved[key] == value,
                f"Changed comparison: {key}",
            )
    require(row["passed"] == all(m["passed"] for m in metrics), "Changed case outcome")
    return metrics


def csv_file(path: Path, rows: list[dict]) -> None:
    """Write every complete row with portable LF newlines."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def illustrate(grid: dict) -> list[dict]:
    """Evaluate nominal first-direction ratios; this executes no optimizer."""
    rows = []
    with localcontext() as context:
        context.prec = 80
        eps = Decimal(str(grid["epsilon"]))
        for scale, value in itertools.product(
            grid["positive_scales"],
            sorted({abs(v) for v in grid["gradient_base"] if v}),
        ):
            c, g = Decimal(str(scale)), Decimal(str(value))
            original, scaled = g / (g + eps), c * g / (c * g + eps)
            rows.append(
                {
                    "gradient_magnitude": value,
                    "gradient_scale": scale,
                    "optimizer_epsilon": grid["epsilon"],
                    "adaptive_direction_ratio": float(scaled / original),
                    "absolute_relative_change": float(abs(scaled / original - 1)),
                }
            )
    return rows


def main() -> None:
    """Audit full coverage and produce separate complete comparison artifacts."""
    measurement_path = RESULTS / "measurements.json"
    result, declared = read(measurement_path), read(STUDY / "plan.json")
    grid = declared["numerical_grid"]
    require(result["source_sha256"] == readiness(), "Changed measured source bytes")
    for path, expected in result["source_sha256"].items():
        verify_hash(RUN / "source" / path, expected)
    expected = set(
        itertools.product(grid["dtypes"], grid["positive_scales"], grid["profiles"])
    )
    cases = result["cases"]
    require(
        not result["execution_errors"]
        and len(cases) == len(expected) == grid["expected_cases"]
        and {(r["dtype"], r["scale"], r["profile"]) for r in cases} == expected,
        "Incomplete or duplicated declared grid",
    )
    cell_rows, metric_rows, files = [], [], []
    for row in cases:
        raw = RUN / f"{row['identifier']}.json"
        require(read(raw) == row, "Raw synthetic trace changed")
        files.append(raw)
        metrics = verify_case(row, grid)
        cell_rows.append(
            {
                "identifier": row["identifier"],
                "dtype": row["dtype"],
                "scale": row["scale"],
                "profile": row["profile"],
                "passed": row["passed"],
                "comparisons": len(metrics),
                "passed_comparisons": sum(m["passed"] for m in metrics),
                "maximum_relative_l2": max(
                    (m["relative_l2"] for m in metrics if m["relative_l2"] is not None),
                    default=None,
                ),
            }
        )
        metric_rows.extend({"identifier": row["identifier"], **m} for m in metrics)
    expected_count = grid["expected_cases"] * grid["comparisons_per_case"]
    require(len(metric_rows) == expected_count, "Omitted comparisons")
    groups = [
        {
            "dtype": dtype,
            "cases": sum(r["dtype"] == dtype for r in cell_rows),
            "passed_cases": sum(r["passed"] and r["dtype"] == dtype for r in cell_rows),
            "maximum_relative_l2": max(
                r["maximum_relative_l2"] for r in cell_rows if r["dtype"] == dtype
            ),
        }
        for dtype in grid["dtypes"]
    ]
    summary = {
        "complete": True,
        "cases": len(cases),
        "passed_cases": sum(r["passed"] for r in cases),
        "comparisons": len(metric_rows),
        "passed_comparisons": sum(m["passed"] for m in metric_rows),
        "execution_errors": 0,
        "groups": groups,
        "maximum_identity_relative_l2": max(
            m["relative_l2"] for m in metric_rows if m["kind"] == "scaling_identity"
        ),
        "native_synthetic_updates": len(cases) * 2 * grid["steps"],
        "model_training_updates": 0,
        "new_task_scores": 0,
        "limits": declared["limits"],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in [
        ("cells", cell_rows),
        ("comparisons", metric_rows),
        ("groups", groups),
        ("first-direction-illustrations", illustrate(grid)),
    ]:
        csv_file(RESULTS / f"{name}.csv", rows)
    artifacts = [
        measurement_path,
        *files,
        RESULTS / "summary.json",
        *RESULTS.glob("*.csv"),
    ]
    audited = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        **summary,
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                STUDY / "plan.json",
                STUDY / "reference.py",
                STUDY / "run.py",
            ]
        },
        "artifacts_sha256": {str(p.relative_to(ROOT)): digest(p) for p in artifacts},
        "evidence": (
            "All represented inputs, every Decimal state and all comparison errors "
            "were reconstructed; all raw/source archive bytes and complete "
            "18-case coverage were checked."
        ),
    }
    (RESULTS / "complete-audit.json").write_text(json.dumps(audited, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
