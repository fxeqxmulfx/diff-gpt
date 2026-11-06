"""Check finite-epsilon LayerNorm and affine bounds without changing the library."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from research.stored_rope_guarantees.check import EXTERNAL, external_dependencies
from research.stored_rope_guarantees.check import digest as sha256

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
CACHE = ROOT / "runs/layernorm-affine-guarantees/lean"
EXPECTED = {
    "LayerNormAffine.center_energy",
    "LayerNormAffine.center_nonexpansive",
    "LayerNormAffine.layerNorm_radius",
    "LayerNormAffine.layerNorm_lipschitz",
    "LayerNormAffine.layerNorm_shift",
    "LayerNormAffine.diagonal_bound",
    "LayerNormAffine.affine_rms_bounds",
    "LayerNormAffine.affine_layerNorm_bounds",
}


def main() -> None:
    """Preserve every attempt and require eight statements with standard axioms."""
    declared = json.loads((STUDY / "plan.json").read_text())
    for name, digest in declared["sources_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(name)
    compiler = shutil.which("lean")
    if compiler is None:
        message = "Invoke under lake env"
        raise RuntimeError(message)
    for name, digest in declared["external_sha256"].items():
        if sha256(EXTERNAL / name) != digest:
            raise ValueError(name)
    sources = [
        ROOT / "research/normalization_guarantees/RMSLipschitz.lean",
        STUDY / "Centering.lean",
        STUDY / "Affine.lean",
    ]
    dependencies = external_dependencies(sources)
    environment = os.environ.copy()
    environment["LEAN_PATH"] = (
        str(CACHE) + os.pathsep + environment.get("LEAN_PATH", "")
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    attempt_path = RESULTS / f"lean-attempt-{stamp}.json"
    attempt = {
        "source_text": {
            str(path.relative_to(ROOT)): path.read_text() for path in sources
        },
        "commands": [],
    }
    for source in sources:
        target = CACHE / source.relative_to(ROOT).with_suffix(".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [compiler, f"--root={ROOT}", "-o", str(target), str(source)]
        completed = subprocess.run(  # noqa: S603 -- Fixed compiler and local paths.
            command,
            cwd=EXTERNAL,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        attempt["commands"].append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        attempt_path.write_text(json.dumps(attempt, indent=2) + "\n")
        print(completed.stdout + completed.stderr, end="", flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    full = "\n".join(row["stdout"] + row["stderr"] for row in attempt["commands"])
    axioms = {
        name: values.split(", ") if values else []
        for name, values in re.findall(
            r"'(LayerNormAffine\.[^']+)' depends on axioms: \[([^\]]*)\]", full
        )
    }
    if (
        "warning:" in full
        or "error:" in full
        or set(axioms) != EXPECTED
        or any(
            set(values) - {"propext", "Classical.choice", "Quot.sound"}
            for values in axioms.values()
        )
        or dependencies != external_dependencies(sources)
    ):
        message = "Incomplete, unclean or changing proof"
        raise RuntimeError(message)
    record = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "exit_code": 0,
        "statements": len(axioms),
        "warnings": 0,
        "extra_axioms": [],
        "axioms": axioms,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                *sources,
                Path(__file__),
                ROOT / "research/stored_rope_guarantees/check.py",
            ]
        },
        "plan_sha256": sha256(STUDY / "plan.json"),
        "external_project": str(EXTERNAL),
        "external_source_sha256": dependencies,
        "attempt": str(attempt_path.relative_to(ROOT)),
        "attempt_sha256": sha256(attempt_path),
        "limitations": declared["limits"],
    }
    (RESULTS / "guarantees.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
