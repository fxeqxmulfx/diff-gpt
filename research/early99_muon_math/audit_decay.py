"""Map the real decay theorem's scalar conditions to all declared Muon settings."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from research.early99.protocol import selected, specification
from research.early99_muon.factories import Configuration
from research.early99_muon.readiness import verify_training_ready
from research.early99_reporting.round2_audit import require

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"


def digest(path: Path) -> str:
    """Bind the mapping to unchanged scientific and proof sources."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Read declarations only; do not construct, evaluate or train a model."""
    verify_training_ready()
    proof_path = RESULTS / "decay-guarantees.json"
    proof = json.loads(proof_path.read_text())
    require(proof["exit_code"] == 0 and not proof["extra_axioms"], "Invalid proof")
    for path, expected in proof["source_sha256"].items():
        require(digest(ROOT / path) == expected, f"Changed proof source: {path}")
    require(digest(ROOT / proof["output"]) == proof["output_sha256"], "Changed output")
    require(
        digest(RESULTS / "guarantees.json") == proof["prerequisite_record_sha256"],
        "Step-bound prerequisite changed",
    )
    plan_path = ROOT / "research/early99_muon/plan.json"
    plan = json.loads(plan_path.read_text())
    tasks = selected("short") + selected("long")
    rows = []
    for declared in plan["configurations"]:
        config = Configuration(**declared)
        require(config.schedule == "constant", "Reassess the changed schedule")
        for task in tasks:
            case = specification(task)
            peak = config.training_case(case).learning_rate
            require(case.weight_decay > 0 and config.muon_eps > 0, "Nonpositive floor")
            require(0 <= peak * case.weight_decay <= 1, "Decay step exceeds one")
            require(case.warmup > 0, "Reassess warmup-free configuration")
            probes = []
            for completed in [0, 1, case.warmup - 1, case.warmup, case.warmup + 1]:
                rate = config.learning_rate_at(case, completed, task["train_examples"])
                require(0 <= rate <= peak, "Warmup exceeds the declared peak")
                probes.append({"completed": completed, "rate": rate})
            rows.append(
                {
                    "configuration": config.name,
                    "case": task["name"],
                    "tier": task["tier"],
                    "weight_decay": case.weight_decay,
                    "matrix_peak_rate": peak,
                    "peak_times_decay": peak * case.weight_decay,
                    "epsilon": config.muon_eps,
                    "warmup_updates": case.warmup,
                    "warmup_boundary_rates": probes,
                    "cap_above_initial_rms": 1 / (4 * case.weight_decay),
                    "auxiliary_adamw_excluded": config.optimizer_kind == "muon_adamw",
                }
            )
    require(
        len({(r["configuration"], r["case"]) for r in rows}) == len(rows),
        "Duplicate configuration/case mapping",
    )
    files = [
        Path(__file__),
        plan_path,
        proof_path,
        ROOT / "research/early99/suite.json",
        ROOT / "research/early99/experiments/factories.py",
        ROOT / "research/early99_muon/factories.py",
        ROOT / "research/early99_muon/optimizer.py",
        ROOT / "research/early99_muon/results/validation.json",
        ROOT / "research/early99_reporting/round2_audit.py",
    ]
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "case_count": len(tasks),
        "configuration_count": len(plan["configurations"]),
        "verified_scalar_conditions": len(rows),
        "maximum_rate_times_decay": max(r["peak_times_decay"] for r in rows),
        "scope": (
            "The declared constant schedule clips the nonnegative warmup multiplier "
            "at one; all peak rates satisfy rate*decay <= 1 with positive decay "
            "and epsilon. Boundary probes check the unchanged implementation. "
            "The theorem remains conditional on real arithmetic and the stated "
            "singular representation, and excludes auxiliary AdamW. No training "
            "or outcome-based selection is performed by this mapping."
        ),
        "evidence_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "rows": rows,
    }
    (RESULTS / "decay-application.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
