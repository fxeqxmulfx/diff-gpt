"""Check every actual fixed-RoPE table using exact rational arithmetic on its bytes."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import torch

from research.current_grokking.data import build
from research.current_grokking.run import verify_model_reference
from research.early99.experiments.factories import baseline_model
from research.early99.protocol import selected, specification, suite
from research.early99_core_ablations.readiness import verify_training_ready
from research.early99_dimension_gain.readiness import require

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
TABLES = ROOT / "runs/stored-rope-guarantees/tables"
EXPECTED_CASES = 35
EXPECTED_STATEMENTS = 4


def digest(path: Path) -> str:
    """Bind each application certificate to its exact code, proof and table bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational(value: Fraction) -> dict:
    """Retain exact numerator/denominator; label the convenience decimal approximate."""
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal_approx": float(value),
    }


def main() -> None:
    """Construct all declared CPU models and preserve buffers, without training."""
    verify_training_ready()
    require(
        verify_model_reference() == suite()["baseline_model_reference"],
        "Original model changed",
    )
    plan = json.loads((STUDY / "plan.json").read_text())
    proof = json.loads((RESULTS / "guarantees.json").read_text())
    require(
        proof["exit_code"] == 0
        and proof["warnings"] == 0
        and proof["statements"] == EXPECTED_STATEMENTS
        and not proof["extra_axioms"]
        and proof["plan_sha256"] == digest(STUDY / "plan.json"),
        "Complete the clean stored-table proof first",
    )
    for name, expected in proof["source_sha256"].items():
        require(digest(ROOT / name) == expected, f"Changed proof source: {name}")
    for name, expected in proof["external_source_sha256"].items():
        require(
            digest(Path(proof["external_project"]) / name) == expected,
            f"Changed proof dependency: {name}",
        )
    for name, expected in plan["local_source_sha256"].items():
        require(digest(ROOT / name) == expected, f"Changed declared source: {name}")
    bound = Fraction(
        plan["pair_energy_bound"]["numerator"], plan["pair_energy_bound"]["denominator"]
    )
    require(bound >= 1, "The bound must also cover leftover coordinates")
    rows, failures = [], []
    maximum = Fraction(0)
    minimum = None
    witness = None
    total = 0
    for row in selected("all"):
        case = specification(row)
        task = build(case, "cpu")
        require(
            task.fingerprints == row["data_fingerprints"],
            f"Changed task context/data: {case.name}",
        )
        torch.manual_seed(0)
        model = baseline_model(task, case, "cpu").model
        c, s = model.rope_cos, model.rope_sin
        shape = [task.context, case.width // case.heads // 2]
        require(
            c.dtype == s.dtype == torch.float32
            and list(c.shape) == list(s.shape) == shape,
            "Unexpected table representation",
        )
        require(
            torch.isfinite(c).all().item() and torch.isfinite(s).all().item(),
            "Nonfinite stored coefficient",
        )
        archive = TABLES / case.name
        archive.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name, table in (("cos", c), ("sin", s)):
            raw = table.contiguous().numpy().tobytes()
            path = archive / f"{name}.bin"
            if path.exists():
                require(
                    path.read_bytes() == raw,
                    f"Previously archived table changed: {path}",
                )
            else:
                path.write_bytes(raw)
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        values = []
        for position, (cosines, sines) in enumerate(
            zip(c.tolist(), s.tolist(), strict=True)
        ):
            for pair, (cosine, sine) in enumerate(zip(cosines, sines, strict=True)):
                # Every binary32 coefficient converts exactly to binary64/Python
                # float, then to its exact rational dyadic. No floating reduction.
                energy = Fraction(cosine) ** 2 + Fraction(sine) ** 2
                values.append(energy)
                if energy > maximum:
                    maximum = energy
                    witness = {
                        "case": case.name,
                        "position": position,
                        "pair": pair,
                        "cos": rational(Fraction(cosine)),
                        "sin": rational(Fraction(sine)),
                        "energy": rational(energy),
                    }
                if energy > bound:
                    failures.append(
                        {
                            "case": case.name,
                            "position": position,
                            "pair": pair,
                            "energy": rational(energy),
                        }
                    )
        low, high = min(values), max(values)
        minimum = low if minimum is None else min(minimum, low)
        total += len(values)
        rows.append(
            {
                "case": case.name,
                "tier": row["tier"],
                "head_width": case.width // case.heads,
                "context": task.context,
                "table_shape": shape,
                "table_pairs": len(values),
                "minimum_pair_energy": rational(low),
                "maximum_pair_energy": rational(high),
                "all_pairs_within_declared_bound": high <= bound,
                "all_pairs_exactly_unit": low == high == 1,
                "table_sha256": hashes,
                "data_fingerprints": task.fingerprints,
            }
        )
        print(
            json.dumps(
                {
                    "case": case.name,
                    "pairs": len(values),
                    "maximum_approx": float(high),
                    "passed": high <= bound,
                }
            ),
            flush=True,
        )
    require(len(rows) == EXPECTED_CASES, "Incomplete model inventory")
    source_paths = [
        Path(__file__),
        STUDY / "plan.json",
        RESULTS / "guarantees.json",
        ROOT / "research/early99_core_ablations/results/validation.json",
    ]
    record = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "expected_models": EXPECTED_CASES,
        "checked_models": len(rows),
        "passed_models": sum(r["all_pairs_within_declared_bound"] for r in rows),
        "table_pairs_checked_including_repeated_shapes": total,
        "stored_coefficients_checked": 2 * total,
        "declared_pair_energy_bound": rational(bound),
        "norm_multiplier_upper_approx": math.sqrt(float(bound)),
        "norm_multiplier_upper_exact": f"sqrt({bound.numerator}/{bound.denominator})",
        "score_multiplier_upper": rational(bound),
        "minimum_pair_energy": rational(minimum),
        "maximum_pair_energy": rational(maximum),
        "maximum_witness": witness,
        "failures": failures,
        "rows": rows,
        "table_encoding": {
            "dtype": "IEEE754 binary32",
            "byte_order": sys.byteorder,
            "order": "C contiguous",
        },
        "torch": str(torch.__version__),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "baseline_model_reference": verify_model_reference(),
        "command": (
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python "
            "-m research.stored_rope_guarantees.audit_tables"
        ),
        "limitations": plan["limitations"],
        "training_updates_issued": 0,
        "application_ready": not failures,
    }
    (RESULTS / "application.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    require(
        not failures, "Stored tables violate the predeclared bound; retain all failures"
    )


if __name__ == "__main__":
    main()
