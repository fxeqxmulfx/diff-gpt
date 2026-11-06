"""Check physical matrix gradients against independent finite-epsilon formulas."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from diff_gpt.domain.model.rms_norm import RMSNorm

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"


def sha256(path: Path) -> str:
    """Bind every observation to exact source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual: torch.Tensor, reference: torch.Tensor, limit: float) -> dict:
    """Use the full tensor; require exact zeros for an identically zero reference."""
    measured = actual.detach().double().flatten()
    expected = reference.detach().double().flatten()
    finite = torch.isfinite(measured).all().item()
    if not torch.isfinite(expected).all().item():
        message = "Nonfinite independent reference"
        raise ValueError(message)
    magnitude = expected.abs().max().item()
    difference = (measured - expected).abs().max().item() if finite else None
    error = (
        difference / magnitude
        if finite and magnitude > 0
        else (0.0 if finite and difference == 0 else None)
    )
    return {
        "shape": list(actual.shape),
        "actual_first_four": [
            x if math.isfinite(x) else str(x) for x in measured[:4].tolist()
        ],
        "reference_first_four": expected[:4].tolist(),
        "reference_max_abs": magnitude,
        "max_absolute_error": difference,
        "relative_max_error": error,
        "finite": finite,
        "passed": error is not None and error <= limit,
    }


def measure(width: int, amplitude: float, scale: float, dtype: str, grid: dict) -> dict:
    """Compare independent physical parameter leaves, with no optimizer update."""
    kind = getattr(torch, dtype)
    eps = grid["epsilon"]["numerator"] / grid["epsilon"]["denominator"]
    limit = grid["relative_error_limit"]
    v = torch.tensor([1, 1, -1, -1], dtype=kind).repeat(width // 4)
    tangent = torch.tensor([1, -1, 1, -1], dtype=kind).repeat(width // 4)
    vector = torch.tensor(amplitude, dtype=kind) * v
    matrix = torch.stack([vector / 2, -vector / 2], dim=-1)
    physical = (scale * matrix).detach().requires_grad_()
    counterpart = matrix.detach().requires_grad_()
    fixed_input = torch.tensor(grid["linear_input"], dtype=kind)
    norm, changed_epsilon = RMSNorm(width), RMSNorm(width)
    norm.eps, changed_epsilon.eps = eps, eps / scale**2
    left, right = (
        norm(physical @ fixed_input),
        changed_epsilon(counterpart @ fixed_input),
    )
    represented = vector[0].item()
    scaled = (physical @ fixed_input)[0].detach().item()
    if scaled != represented * scale:
        message = "Declared power-of-two rescaling was not exact"
        raise ValueError(message)
    if (v * tangent).sum().item() != 0:
        message = "Declared directions are not orthogonal"
        raise ValueError(message)
    roots = {
        "physical": math.sqrt(scaled**2 + eps),
        "counterpart": math.sqrt(represented**2 + eps / scale**2),
    }
    radial = {
        "physical": eps / roots["physical"] ** 3,
        "counterpart": (eps / scale**2) / roots["counterpart"] ** 3,
    }
    checks = {
        "physical_forward": compare(
            left, (scaled / roots["physical"]) * v.double(), limit
        ),
        "counterpart_forward": compare(
            right, (represented / roots["counterpart"]) * v.double(), limit
        ),
        "forward_scaling_identity": compare(left, right, limit),
    }
    upstreams = [
        ("radial", v, 1.0, 0.0),
        ("tangent", tangent, 0.0, 1.0),
        ("mixed", (v + tangent) / 2, 0.5, 0.5),
    ]
    for name, upstream, radial_part, tangent_part in upstreams:
        actual_left = torch.autograd.grad(
            left, physical, grad_outputs=upstream, retain_graph=True
        )[0]
        actual_right = torch.autograd.grad(
            right, counterpart, grad_outputs=upstream, retain_graph=True
        )[0]
        for side, actual in [("physical", actual_left), ("counterpart", actual_right)]:
            reference_vjp = (
                radial_part * radial[side] * v.double()
                + (tangent_part / roots[side]) * tangent.double()
            )
            reference = reference_vjp[:, None] * fixed_input.double()[None, :]
            checks[f"{side}_{name}_parameter_gradient"] = compare(
                actual, reference, limit
            )
        checks[f"{name}_inverse_scale_identity"] = compare(
            actual_left, actual_right.double() / scale, limit
        )
    forward_ratio = (
        scale * math.sqrt(represented**2 + eps) / roots["physical"]
        if represented
        else None
    )
    return {
        "width": width,
        "requested_amplitude": amplitude,
        "represented_amplitude": represented,
        "scaled_represented_amplitude": scaled,
        "scale": scale,
        "dtype": dtype,
        "checks": checks,
        "passed": all(row["passed"] for row in checks.values()),
        "analytic_same_epsilon_forward_amplitude_ratio": forward_ratio,
        "physical_radial_coefficient": radial["physical"],
        "physical_tangent_coefficient": 1 / roots["physical"],
        "physical_radial_to_tangent_ratio": eps / (scaled**2 + eps),
        "unscaled_radial_to_tangent_ratio": eps / (represented**2 + eps),
    }


def main() -> None:
    """Require the frozen declaration and checked proofs before all 96 cases."""
    plan_path = STUDY / "plan.json"
    declared = json.loads(plan_path.read_text())
    committed = subprocess.run(  # noqa: S603 -- Fixed local git read.
        ["/usr/bin/git", "show", f"HEAD:{plan_path.relative_to(ROOT)}"],
        capture_output=True,
        check=True,
    ).stdout
    if committed != plan_path.read_bytes():
        message = "Commit the declaration before the numerical check"
        raise ValueError(message)
    proof_path = RESULTS / "guarantees.json"
    proof = json.loads(proof_path.read_text())
    if (
        proof["exit_code"] != 0
        or proof["statements"] != len(declared["proof_targets"])
        or proof["warnings"] != 0
        or proof["extra_axioms"]
        or proof["plan_sha256"] != sha256(plan_path)
    ):
        message = "Complete the declared proofs before the numerical check"
        raise ValueError(message)
    for path, digest in {
        **declared["sources_sha256"],
        **proof["source_sha256"],
    }.items():
        if sha256(ROOT / path) != digest:
            raise ValueError(path)
    grid = declared["numerical_grid"]
    eps = grid["epsilon"]["numerator"] / grid["epsilon"]["denominator"]
    if eps != torch.finfo(torch.float32).eps:
        message = "Changed native FP32 epsilon"
        raise ValueError(message)
    rows = [
        measure(width, amplitude, scale, dtype, grid)
        for width in grid["widths"]
        for amplitude in grid["amplitudes"]
        for scale in grid["positive_scales"]
        for dtype in grid["dtypes"]
    ]
    if len(rows) != grid["expected_cases"]:
        message = "Incomplete numerical grid"
        raise ValueError(message)
    errors = [
        check["relative_max_error"]
        for row in rows
        for check in row["checks"].values()
        if check["relative_max_error"] is not None
    ]
    record = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "checks": sum(len(row["checks"]) for row in rows),
        "maximum_finite_relative_error": max(errors),
        "failures": [row for row in rows if not row["passed"]],
        "rows": rows,
        "torch": str(torch.__version__),
        "optimizer_updates": 0,
        "sources_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [Path(__file__), plan_path, proof_path]
        },
        "limits": declared["limits"],
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    attempt = RESULTS / f"runtime-attempt-{stamp}.json"
    attempt.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    record["attempt"] = str(attempt.relative_to(ROOT))
    record["attempt_sha256"] = sha256(attempt)
    (RESULTS / "runtime.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: record[key]
                for key in [
                    "complete",
                    "cases",
                    "passed",
                    "checks",
                    "maximum_finite_relative_error",
                ]
            }
        )
    )
    if record["failures"]:
        message = "Retained failures in the declared initialization-scale check"
        raise RuntimeError(message)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
