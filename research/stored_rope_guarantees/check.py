"""Compile the stored-table score bounds and preserve every compiler attempt."""

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
RESULTS = STUDY / "results"
EXTERNAL = Path("/home/misha/lean_projects/transformer")
CACHE = ROOT / "runs/stored-rope-guarantees/lean"
EXPECTED = {
    f"StoredRoPE.{name}"
    for name in (
        "energy_bound",
        "norm_bound",
        "score_bound",
        "ssmax_score_bound",
    )
}


def digest(path: Path) -> str:
    """Hash the exact bytes checked by the compiler."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_dependencies(sources: list[Path]) -> dict:
    """Pin the transitive local Lean sources and the Mathlib/toolchain manifest."""
    records = {}
    pending = list(sources)
    while pending:
        source = pending.pop()
        for module in re.findall(
            r"^\s*(?:public )?import (Transformer[\w.]*)",
            source.read_text(),
            re.MULTILINE,
        ):
            relative = f"src/{module.replace('.', '/')}.lean"
            if relative not in records:
                path = EXTERNAL / relative
                records[relative] = digest(path)
                pending.append(path)
    for relative in ("lake-manifest.json", "lean-toolchain"):
        records[relative] = digest(EXTERNAL / relative)
    return records


def main() -> None:
    """Use lake env with an isolated workspace cache; never edit the proof library."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    log = RESULTS / f"lean-attempt-{stamp}.txt"
    compiler = shutil.which("lean")
    if compiler is None:
        message = "Run this checker under lake env"
        raise RuntimeError(message)
    env = os.environ.copy()
    env["LEAN_PATH"] = str(CACHE) + os.pathsep + env.get("LEAN_PATH", "")
    plan = json.loads((STUDY / "plan.json").read_text())
    for name, expected in plan["local_source_sha256"].items():
        if digest(ROOT / name) != expected:
            message = f"Declared source changed: {name}"
            raise RuntimeError(message)
    for name, expected in plan["external_source_sha256"].items():
        if digest(EXTERNAL / name) != expected:
            message = f"Declared external source changed: {name}"
            raise RuntimeError(message)
    sources = [STUDY / "Bounds.lean"]
    external = external_dependencies(sources)
    output = []
    for source in sources:
        target = CACHE / source.relative_to(ROOT).with_suffix(".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(  # noqa: S603 -- Fixed compiler and local source paths.
            [compiler, f"--root={ROOT}", "-o", str(target), str(source)],
            cwd=EXTERNAL,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        output.append(result.stdout + result.stderr)
        log.write_text("\n".join(output))
        print(output[-1], end="", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    full = "\n".join(output)
    axioms = {
        name: values.split(", ") if values else []
        for name, values in re.findall(
            r"'(StoredRoPE\.[^']+)' depends on axioms: \[([^\]]*)\]", full
        )
    }
    allowed = {"propext", "Classical.choice", "Quot.sound"}
    if (
        "warning:" in full
        or "error:" in full
        or set(axioms) != EXPECTED
        or any(set(values) - allowed for values in axioms.values())
        or external != external_dependencies(sources)
    ):
        message = "Incomplete, unclean or changing stored-table proof"
        raise RuntimeError(message)
    record = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
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
        "external_source_sha256": external,
        "command": f"lake env python3 {STUDY / 'check.py'}",
        "hypotheses": (
            "Real arithmetic, epsilon > 0, d > 0, M >= 1; "
            "each query/key table pair has squared energy <= M; N=n+1."
        ),
        "mapping": plan["proof"],
        "limitations": plan["limitations"],
        "plan_sha256": digest(STUDY / "plan.json"),
    }
    (RESULTS / "guarantees.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
