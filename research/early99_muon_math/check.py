"""Compile sibling matrix bounds while preserving the frozen optimizer theorem."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
EXTERNAL = Path("/home/misha/lean_projects/transformer")
CACHE = ROOT / "runs/early99-muon-math/lean"
RESULTS = STUDY / "results"
EXPECTED_THEOREMS = {
    f"Early99MuonMath.{name}"
    for name in (
        "denominator_pos",
        "normalized_coefficients",
        "floorNorm_spectrum",
        "finiteSteps_spectrum",
        "oriented_direction",
        "direction_energy",
        "adjusted_direction_rms",
        "update_rms",
    )
}


def digest(path: Path) -> str:
    """Hash every compiled source and proof output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run under lake env; compile the unchanged prerequisite into a private cache."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    log = RESULTS / f"lean-attempt-{stamp}.txt"
    env = os.environ.copy()
    env["LEAN_PATH"] = str(CACHE) + os.pathsep + env.get("LEAN_PATH", "")
    sources = [ROOT / "research/early99_muon/Guarantees.lean", STUDY / "Bounds.lean"]
    output = []
    for source in sources:
        target = CACHE / source.relative_to(ROOT).with_suffix(".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = ["lean", f"--root={ROOT}", "-o", str(target), str(source)]
        result = subprocess.run(  # noqa: S603 -- Fixed local compiler and source files.
            command, cwd=EXTERNAL, env=env, capture_output=True, text=True, check=False
        )
        output.append(result.stdout + result.stderr)
        log.write_text("\n".join(output))
        print(result.stdout + result.stderr, end="", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    full = "\n".join(output)
    if "warning:" in full or "error:" in full:
        message = "Proof compilation must have no warnings or errors"
        raise RuntimeError(message)
    axioms = {
        name: values.split(", ") if values else []
        for name, values in re.findall(
            r"'(Early99MuonMath\.[^']+)' depends on axioms: \[([^\]]*)\]", full
        )
    }
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    if set(axioms) != EXPECTED_THEOREMS or any(
        set(a) - allowed for a in axioms.values()
    ):
        message = "Missing matrix theorem or nonstandard axiom"
        raise RuntimeError(message)
    frozen = json.loads(
        (ROOT / "research/early99_muon/results/guarantees.json").read_text()
    )
    for path, expected in frozen["source_sha256"].items():
        if digest(ROOT / path) != expected:
            message = "Frozen scalar proof changed"
            raise RuntimeError(message)
    for path, expected in frozen["external_source_sha256"].items():
        if digest(EXTERNAL / path) != expected:
            message = "External proof dependency changed"
            raise RuntimeError(message)
    result = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "exit_code": 0,
        "warnings": 0,
        "statements": len(axioms),
        "axioms": axioms,
        "extra_axioms": [],
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in [*sources, Path(__file__)]
        },
        "output": str(log.relative_to(ROOT)),
        "output_sha256": digest(log),
        "external_project": str(EXTERNAL),
        "external_source_sha256": frozen["external_source_sha256"],
        "prerequisite_record_sha256": digest(
            ROOT / "research/early99_muon/results/guarantees.json"
        ),
        "command": f"lake env python3 {STUDY / 'check.py'}",
        "mapping": (
            "The same epsilon-floor normalization and five finite quintic steps "
            "have squared Frobenius norm <= 25r/16 under a nonnegative singular "
            "representation with orthonormal columns. Transposing the normalized "
            "matrix before the steps and transposing back preserves the direction. "
            "For positive matrix dimensions and r <= min(rows, columns), the "
            "implemented 0.2sqrt(max(shape)) multiplier gives RMS <= 1/4. "
            "The displacement from the decayed parameter has RMS <= rate/4 "
            "for rate >= 0."
        ),
        "limitations": (
            "Real arithmetic with explicit singular-representation assumptions. "
            "Does not include FP32 casts/product rounding, auxiliary AdamW, or "
            "the separate weight-decay displacement. Does not prove transformer "
            "convergence, numerical robustness of every model block, or earlier "
            "exact held-out accuracy."
        ),
    }
    (RESULTS / "guarantees.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
