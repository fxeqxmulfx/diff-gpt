"""Audit a completed three-seed capacity contrast without closing the full study."""

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
from research.cubic_capacity_reporting.report import (
    contrast_metrics,
    group_metrics,
    paired_ratio,
    write_csv,
)
from research.early99_reporting.round2_audit import read, require, verify_hash


def audit_group(name: str) -> dict:
    """Require every seed in one predeclared contrast, including all budget misses."""
    validation = verify_training_ready()
    search, spec = plan(), resolve(name)
    shape = next(s for s in search["shapes"] if s["name"] == spec["size"])
    current = next(s for s in search["shapes"] if s["reuse_existing"])
    root, row = RUNS / name, task_row(spec)
    require(
        all(
            (root / row["name"] / f"seed{seed}/early99_result.json").is_file()
            for seed in search["seeds"]
        ),
        "Wait for all three seeds; do not report a selected subset",
    )
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
    for path, digest in archived["source_sha256"].items():
        verify_hash(root / "experiment_source" / path, digest)
    baseline = [
        audit_cell(
            ROOT / search["controls"][spec["arm"]], search["task"], seed, spec, current
        )
        for seed in search["seeds"]
    ]
    candidate = [audit_cell(root, row, seed, spec, shape) for seed in search["seeds"]]
    pairs = [
        paired_ratio(a, b, "capacity") for a, b in zip(baseline, candidate, strict=True)
    ]
    return {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "complete_group": True,
        "complete_study": False,
        "configuration": name,
        "new_cells_in_this_group": len(candidate),
        "reused_controls_in_this_group": len(baseline),
        "expected_new_cells_full_study": search["new_training_cells"],
        "groups": [group_metrics(baseline), group_metrics(candidate)],
        "contrast": contrast_metrics(pairs),
        "paired_cells": pairs,
        "cells": [*baseline, *candidate],
        "task": search["task"],
        "scope": search["scope"],
        "fixed75_selection_changed": False,
        "new_long_suite_authorized": False,
        "remaining_queue_changed": False,
        "limits": (
            "One completed predeclared capacity contrast. Other sizes, rates and "
            "architectures remain pending. Preserve all 27 new cells; this partial "
            "report cannot close the full study or select a benchmark winner. "
            "Changed model shape changes initialization and parameter count. "
            "Historical runtime ratios are omitted because journal overhead differs."
        ),
        "source_and_evidence_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
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
    """Write separate partial artifacts; never create full-study completion files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    result = audit_group(args.name)
    destination = STUDY / "results" / f"partial-group-{args.name}.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(destination.with_suffix(".csv"), result["paired_cells"])
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "complete_group",
                    "complete_study",
                    "configuration",
                    "contrast",
                ]
            }
        )
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
