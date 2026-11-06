"""Check existing RMS forward and directional gradients against independent formulas."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import torch

from diff_gpt.domain.model.rms_norm import RMSNorm

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"


def digest(path: Path) -> str:
    """Bind the observations to the checked source and declared grid."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison(actual: float, expected: float, limit: float) -> dict:
    """Keep nonfinite and zero-reference failures visible without invalid JSON."""
    finite = math.isfinite(actual)
    error = (
        abs(actual - expected) / abs(expected)
        if finite and expected != 0
        else (0.0 if actual == expected == 0 else None)
    )
    return {
        "actual": actual if finite else str(actual),
        "reference": expected,
        "relative_error": error,
        "passed": error is not None and error <= limit,
    }


def main() -> None:
    """Run the complete fixed CPU grid, with FP32 epsilon in both precisions."""
    plan_path = STUDY / "runtime-plan.json"
    plan = json.loads(plan_path.read_text())
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            message = f"Declared runtime-check source changed: {name}"
            raise RuntimeError(message)
    eps_fraction = Fraction(
        plan["epsilon"]["numerator"], plan["epsilon"]["denominator"]
    )
    if Fraction(torch.finfo(torch.float32).eps) != eps_fraction:
        message = "Declared epsilon differs from the actual FP32 native epsilon"
        raise RuntimeError(message)
    eps = float(eps_fraction)
    rows = []
    for width in plan["widths"]:
        for requested in plan["amplitudes"]:
            for dtype_name in plan["dtypes"]:
                dtype = getattr(torch, dtype_name)
                maps = {}
                for count, name in ((1, "single"), (2, "double")):
                    amplitude = torch.tensor(requested, dtype=dtype, requires_grad=True)
                    represented = amplitude.item()
                    norm = RMSNorm(width)
                    norm.eps = eps
                    value = amplitude * torch.ones(width, dtype=dtype)
                    for _ in range(count):
                        value = norm(value)
                    observed = value.mean()
                    derivative = torch.autograd.grad(observed, amplitude)[0].item()
                    if name == "single":
                        energy = represented**2 + eps
                        numerator = eps
                    else:
                        energy = (1 + eps) * represented**2 + eps**2
                        numerator = eps**2
                    root = math.sqrt(energy)
                    maps[name] = {
                        "value": comparison(
                            observed.item(),
                            represented / root,
                            plan["relative_error_limit"],
                        ),
                        "directional_derivative": comparison(
                            derivative,
                            numerator / root**3,
                            plan["relative_error_limit"],
                        ),
                    }
                row = {
                    "width": width,
                    "requested_amplitude": requested,
                    "represented_amplitude": represented,
                    "dtype": dtype_name,
                    "maps": maps,
                    "passed": all(
                        measurement["passed"]
                        for values in maps.values()
                        for measurement in values.values()
                    ),
                }
                rows.append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
    if len(rows) != plan["expected_cases"]:
        message = "Incomplete declared RMS runtime-check grid"
        raise RuntimeError(message)
    finite_errors = [
        measurement["relative_error"]
        for row in rows
        for values in row["maps"].values()
        for measurement in values.values()
        if measurement["relative_error"] is not None
    ]
    result = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failures": [row for row in rows if not row["passed"]],
        "maximum_finite_relative_error": max(finite_errors),
        "epsilon_exact": plan["epsilon"],
        "single_zero_derivative_approx": 1 / math.sqrt(eps),
        "double_zero_derivative_exact": {
            "numerator": eps_fraction.denominator,
            "denominator": eps_fraction.numerator,
        },
        "rows": rows,
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (Path(__file__), plan_path)
        },
        "torch": str(torch.__version__),
        "optimizer_updates": 0,
        "limitations": plan["limitations"],
        "command": (
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python "
            "-m research.rms_composition_guarantees.check_runtime"
        ),
    }
    (RESULTS / "runtime.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if result["failures"]:
        message = "Existing RMS implementation disagrees with a declared formula check"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
