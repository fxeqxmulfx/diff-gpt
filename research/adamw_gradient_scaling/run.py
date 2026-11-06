"""Execute the complete frozen synthetic AdamW grid without model training."""

from __future__ import annotations

import itertools
import json
import math
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.adamw_gradient_scaling.reference import decimal_trace, represented_trace
from research.early99_reporting.round2_audit import read, require, verify_hash
from research.stored_rope_guarantees.check import EXTERNAL, ROOT, digest

STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
RUN = ROOT / "runs/adamw-gradient-scaling/optimizer-traces-v1"
FIELDS = ("parameter", "moment", "variance")
EXPECTED_PROOFS, EXPECTED_TESTS = 3, 2
PULSE_START, PULSE_REVERSE, PULSE_END = 16, 32, 48


def readiness() -> dict:
    """Require committed proof, reference checks, inputs and all synthetic sources."""
    declared = read(STUDY / "plan.json")
    for group in ("sources_sha256", "installed_optimizer_sha256"):
        for path, expected in declared[group].items():
            verify_hash(ROOT / path, expected)
    for path, expected in declared["external_sha256"].items():
        verify_hash(EXTERNAL / path, expected)
    proof = read(RESULTS / "guarantees.json")
    require(
        proof["statements"] == EXPECTED_PROOFS
        and proof["exit_code"] == 0
        and proof["warnings"] == 0
        and not proof["extra_axioms"]
        and proof["plan_sha256"] == digest(STUDY / "plan.json"),
        "Incomplete proof",
    )
    for path, expected in proof["source_sha256"].items():
        verify_hash(ROOT / path, expected)
    for path, expected in proof["external_source_sha256"].items():
        verify_hash(EXTERNAL / path, expected)
    verify_hash(ROOT / proof["attempt"], proof["attempt_sha256"])
    suites = list(
        ET.parse(RESULTS / "reference-validation.xml").getroot().iter("testsuite")  # noqa: S314 -- Own local pytest output.
    )
    require(
        sum(int(s.attrib["tests"]) for s in suites) == EXPECTED_TESTS
        and all(
            int(s.attrib.get(key, 0)) == 0
            for s in suites
            for key in ("errors", "failures", "skipped")
        ),
        "Independent reference validation incomplete",
    )
    validated = read(RESULTS / "reference-validation.json")
    require(validated["passed"] == EXPECTED_TESTS, "Changed reference check coverage")
    for path, expected in validated["source_and_evidence_sha256"].items():
        verify_hash(ROOT / path, expected)
    paths = [
        *STUDY.glob("*.py"),
        STUDY / "Bounds.lean",
        STUDY / "plan.json",
        RESULTS / "guarantees.json",
        RESULTS / "reference-validation.xml",
        RESULTS / "reference-validation.txt",
        RESULTS / "reference-validation.json",
        ROOT / "tests/diff_gpt/application/test_adamw_gradient_scaling.py",
        *[ROOT / path for path in proof["source_sha256"]],
        ROOT / proof["attempt"],
    ]
    hashes = {}
    for path in sorted(set(paths)):
        relative = str(path.relative_to(ROOT))
        committed = subprocess.run(  # noqa: S603 -- Fixed local Git and source paths.
            ["/usr/bin/git", "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        require(committed == path.read_bytes(), f"Commit validated source: {path}")
        hashes[relative] = digest(path)
    return hashes


def multiplier(profile: str, step: int) -> float:
    """Implement the three fully declared, externally supplied gradient histories."""
    if profile == "constant":
        return 1.0
    if profile == "alternating":
        return 1.0 if step % 2 == 0 else -0.5
    if profile == "delayed_pulse":
        return float(PULSE_START <= step < PULSE_REVERSE) - float(
            PULSE_REVERSE <= step < PULSE_END
        )
    message = f"Unknown profile: {profile}"
    raise ValueError(message)


def native_trace(
    initial: list,
    gradients: list,
    rates: list,
    grid: dict,
    eps: float,
    dtype: torch.dtype,
) -> list[dict]:
    """Use native non-fused CPU AdamW with supplied gradients and retained state."""
    parameter = torch.nn.Parameter(torch.tensor(initial, dtype=dtype, device="cpu"))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=0.0,
        betas=tuple(grid["betas"]),
        eps=eps,
        weight_decay=grid["weight_decay"],
        foreach=False,
        fused=False,
        capturable=False,
        amsgrad=False,
    )
    history = []
    for clock, (gradient, rate) in enumerate(zip(gradients, rates, strict=True), 1):
        parameter.grad = torch.tensor(gradient, dtype=dtype, device="cpu")
        optimizer.param_groups[0]["lr"] = rate
        optimizer.step()
        state = optimizer.state[parameter]
        require(int(state["step"]) == clock, "Native clock changed")
        history.append(
            {
                "clock": clock,
                "parameter": parameter.detach().tolist(),
                "moment": state["exp_avg"].tolist(),
                "variance": state["exp_avg_sq"].tolist(),
            }
        )
    return history


def errors(actual: list[float], reference: list[float], limit: float) -> dict:
    """Preserve finite, absolute and relative errors, including exact-zero vectors."""
    finite = all(math.isfinite(v) for v in [*actual, *reference])
    if not finite:
        return {
            "finite": False,
            "max_absolute": None,
            "relative_l2": None,
            "passed": False,
        }
    norm = math.sqrt(math.fsum(v * v for v in reference))
    absolute = max(abs(a - b) for a, b in zip(actual, reference, strict=True))
    relative = (
        math.dist(actual, reference) / norm if norm else (0.0 if not absolute else None)
    )
    return {
        "finite": True,
        "max_absolute": absolute,
        "relative_l2": relative,
        "passed": relative is not None and relative <= limit,
    }


def comparisons(row: dict, grid: dict) -> list[dict]:
    """Keep six native/reference and three direct-identity comparisons per step."""
    rows = []
    for step in range(grid["steps"]):
        for arm in ("scaled", "counterpart"):
            for field in FIELDS:
                rows.append(  # noqa: PERF401 -- Keep the comparison axes explicit.
                    {
                        "clock": step + 1,
                        "kind": "decimal_reference",
                        "arm": arm,
                        "field": field,
                        **errors(
                            row["native"][arm][step][field],
                            row["reference"][arm][step][field],
                            grid["relative_error_limit"],
                        ),
                    }
                )
        for field, factor in zip(
            FIELDS, [1.0, row["scale"], row["scale"] ** 2], strict=True
        ):
            rows.append(
                {
                    "clock": step + 1,
                    "kind": "scaling_identity",
                    "arm": "paired",
                    "field": field,
                    **errors(
                        row["native"]["scaled"][step][field],
                        [factor * v for v in row["native"]["counterpart"][step][field]],
                        grid["relative_error_limit"],
                    ),
                }
            )
    return rows


def measure(dtype_name: str, scale: float, profile: str, grid: dict) -> dict:
    """Run one declared synthetic pair, retaining every represented input/output."""
    dtype = getattr(torch, dtype_name)
    rng = torch.get_rng_state().clone()
    initial = torch.tensor(
        grid["initial_parameters"], dtype=dtype, device="cpu"
    ).tolist()
    base = torch.tensor(grid["gradient_base"], dtype=dtype, device="cpu")
    gradients = [(base * multiplier(profile, k)).tolist() for k in range(grid["steps"])]
    supplied = {
        "counterpart": gradients,
        "scaled": [[scale * v for v in values] for values in gradients],
    }
    rates = [
        grid["peak_learning_rate"] * min(1.0, k / grid["warmup_completed_updates"])
        for k in range(grid["steps"])
    ]
    epsilons = {"scaled": grid["epsilon"], "counterpart": grid["epsilon"] / scale}
    row = {
        "dtype": dtype_name,
        "scale": scale,
        "profile": profile,
        "initial": initial,
        "gradients": supplied,
        "rates": rates,
        "epsilons": epsilons,
        "native": {},
        "reference": {},
    }
    for arm, eps in epsilons.items():
        row["native"][arm] = native_trace(
            initial, supplied[arm], rates, grid, eps, dtype
        )
        row["reference"][arm] = represented_trace(
            decimal_trace(
                initial, supplied[arm], rates, grid["betas"], eps, grid["weight_decay"]
            )
        )
    require(torch.equal(rng, torch.get_rng_state()), "Synthetic pair consumed RNG")
    row["rng_unchanged"] = True
    row["comparisons"] = comparisons(row, grid)
    row["passed"] = all(m["passed"] for m in row["comparisons"])
    return row


def safe_json(value: object) -> str:
    """Encode any nonfinite observations explicitly without losing failing vectors."""

    def clean(item: object) -> object:
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        if isinstance(item, dict):
            return {key: clean(v) for key, v in item.items()}
        if isinstance(item, list):
            return [clean(v) for v in item]
        return item

    return json.dumps(clean(value), indent=2, allow_nan=False) + "\n"


def main() -> None:
    """Refuse overwrite, archive committed inputs and retain every declared case."""
    hashes = readiness()
    require(not RUN.exists(), "Preserve the existing measurement; do not rerun")
    RUN.mkdir(parents=True)
    for path in hashes:
        destination = RUN / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    declared = read(STUDY / "plan.json")
    grid = declared["numerical_grid"]
    cells, failures = [], []
    for dtype, scale, profile in itertools.product(
        grid["dtypes"], grid["positive_scales"], grid["profiles"]
    ):
        identifier = f"{dtype}-{scale:g}-{profile}"
        try:
            row = measure(dtype, scale, profile, grid)
            row["identifier"] = identifier
            cells.append(row)
            (RUN / f"{identifier}.json").write_text(safe_json(row))
        except Exception:  # noqa: BLE001 -- Preserve failure and finish declared coverage.
            error = {"identifier": identifier, "traceback": traceback.format_exc()}
            failures.append(error)
            (RUN / f"{identifier}-error.json").write_text(safe_json(error))
    result = {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "torch": torch.__version__,
        "source_sha256": hashes,
        "cases": cells,
        "execution_errors": failures,
        "expected_cases": grid["expected_cases"],
        "limits": declared["limits"],
    }
    (RESULTS / "measurements.json").write_text(safe_json(result))
    print(
        json.dumps(
            {
                "measured_cases": len(cells),
                "passed_cases": sum(r["passed"] for r in cells),
                "execution_errors": len(failures),
            }
        )
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
