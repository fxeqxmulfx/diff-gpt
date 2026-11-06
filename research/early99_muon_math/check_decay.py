"""Check parameter-growth bounds without replacing the frozen step-bound record."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
EXTERNAL = Path("/home/misha/lean_projects/transformer")
CACHE = ROOT / "runs/early99-muon-math/lean"
RESULTS = STUDY / "results"
EXPECTED = {
    f"Early99MuonDecay.{name}"
    for name in (
        "energy_convex",
        "rms_convex_bound",
        "step_bound",
        "trajectory_bound",
    )
}


def digest(path: Path) -> str:
    """Bind the record to the compiled source and complete compiler output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Use lake's environment and a workspace cache, leaving external sources alone."""
    prerequisite_path = RESULTS / "guarantees.json"
    prerequisite = json.loads(prerequisite_path.read_text())
    for path, expected in prerequisite["source_sha256"].items():
        if digest(ROOT / path) != expected:
            message = f"Step-bound source changed: {path}"
            raise RuntimeError(message)
    for path, expected in prerequisite["external_source_sha256"].items():
        if digest(EXTERNAL / path) != expected:
            message = f"External proof dependency changed: {path}"
            raise RuntimeError(message)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    log = RESULTS / f"decay-attempt-{stamp}.txt"
    env = os.environ.copy()
    env["LEAN_PATH"] = str(CACHE) + os.pathsep + env.get("LEAN_PATH", "")
    compiler = shutil.which("lean")
    if compiler is None:
        message = "Run this checker under lake env"
        raise RuntimeError(message)
    sources = [
        ROOT / "research/early99_muon/Guarantees.lean",
        STUDY / "Bounds.lean",
        STUDY / "DecayBounds.lean",
    ]
    output = []
    for source in sources:
        target = CACHE / source.relative_to(ROOT).with_suffix(".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(  # noqa: S603 -- Fixed compiler and local sources.
            [compiler, f"--root={ROOT}", "-o", str(target), str(source)],
            cwd=EXTERNAL,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        output.append(result.stdout + result.stderr)
        log.write_text("\n".join(output))
        print(result.stdout + result.stderr, end="", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    full = "\n".join(output)
    axioms = {
        name: values.split(", ") if values else []
        for name, values in re.findall(
            r"'(Early99MuonDecay\.[^']+)' depends on axioms: \[([^\]]*)\]", full
        )
    }
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    if (
        "warning:" in full
        or "error:" in full
        or set(axioms) != EXPECTED
        or any(set(values) - allowed for values in axioms.values())
    ):
        message = "Incomplete or unclean decay proof"
        raise RuntimeError(message)
    record = {
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
        "prerequisite_record_sha256": digest(prerequisite_path),
        "external_project": str(EXTERNAL),
        "external_source_sha256": prerequisite["external_source_sha256"],
        "command": f"lake env python3 {STUDY / 'check_decay.py'}",
        "mapping": (
            "For nonnegative rates with rate*decay <= 1 and positive decay and "
            "epsilon, every real matrix iterate has RMS <= "
            "max(initial RMS, 1/(4*decay)). Each direction uses the actual "
            "five-step quintic and epsilon-floor normalization, with an explicit "
            "orthonormal singular representation and nonnegative coefficients. "
            "Zero rates during warmup and zero singular coefficients are allowed."
        ),
        "limitations": (
            "Does not include floating-point rounding or auxiliary AdamW. "
            "The all-iterate cap requires positive decay and rate*decay <= 1; "
            "it is not a claim of convergence, generalization or faster learning. "
            "A fixed r-column singular representation is assumed at each step; "
            "the actual rank may be smaller."
        ),
    }
    (RESULTS / "decay-guarantees.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
