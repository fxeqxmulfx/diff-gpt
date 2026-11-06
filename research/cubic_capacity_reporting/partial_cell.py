"""Audit one completed capacity cell and its paired existing-size control."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.cubic_capacity.factories import (
    ROOT,
    RUNS,
    STUDY,
    configuration,
    plan,
    resolve,
    task_row,
)
from research.cubic_capacity.readiness import sha256, verify_training_ready
from research.cubic_capacity_reporting.audit import audit_cell
from research.cubic_capacity_reporting.report import paired_ratio
from research.early99_reporting.round2_audit import read, require, verify_hash


def audit_pair(name: str, seed: int) -> dict:
    """Retain a single seed without claiming group or full-study completion."""
    validation = verify_training_ready()
    search, spec = plan(), resolve(name)
    require(seed in search["seeds"], "Undeclared seed")
    root, row = RUNS / name, task_row(spec)
    archived = read(root / "plan.json")
    require(
        archived["study_plan_sha256"] == sha256(STUDY / "plan.json")
        and archived["specification"] == spec
        and archived["configuration"] == configuration(spec).__dict__
        and archived["task"] == row
        and archived["source_sha256"] == validation["source_sha256"]
        and archived["execution"]
        == {
            "device": "cuda",
            "compiled_loss": True,
            "cuda_graphs": True,
            "durable_per_update_journal": True,
        },
        "Changed declared execution",
    )
    for path, expected in archived["source_sha256"].items():
        verify_hash(root / "experiment_source" / path, expected)
    current = next(s for s in search["shapes"] if s["reuse_existing"])
    shape = next(s for s in search["shapes"] if s["name"] == spec["size"])
    baseline = audit_cell(
        ROOT / search["controls"][spec["arm"]], search["task"], seed, spec, current
    )
    candidate = audit_cell(root, row, seed, spec, shape)
    return {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "complete_study": False,
        "complete_group": False,
        "new_cells_audited_in_this_record": 1,
        "reused_control_cells_audited_in_this_record": 1,
        "expected_new_cells": search["new_training_cells"],
        "configuration": name,
        "seed": seed,
        "rows": [baseline, candidate],
        "paired_comparison": paired_ratio(baseline, candidate, "capacity"),
        "fixed75_selection_changed": False,
        "remaining_queue_changed": False,
        "limits": (
            "One paired seed only; the three-seed contrast and full27-cell study "
            "remain incomplete. Preserve every declared run and failure. Model "
            "shape changes parameters and initialized tensors. Historical runtime "
            "ratios are omitted because durable-journal overhead differs."
        ),
        "source_and_evidence_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("audit.py"),
                Path(__file__).with_name("report.py"),
                STUDY / "plan.json",
                STUDY / "results/validation.json",
                root / "plan.json",
            ]
        },
    }


def main() -> None:
    """Use a distinct partial filename that cannot satisfy full-study gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    result = audit_pair(args.name, args.seed)
    path = STUDY / "results" / f"partial-cell-{args.name}-seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["paired_comparison"]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
