"""Replay all 50 complete-model numerical records and verify every fingerprint."""

from __future__ import annotations

import itertools
from pathlib import Path

import torch

from research.inverse_width_stack_stress.run import (
    OUTPUT,
    PLAN,
    RESULTS,
    cell,
    prepare,
    read,
    sha256,
)
from research.transformer_stress.common import ROOT, clean, require, write_json


def main() -> None:
    """Reproduce the actual forward/backward observations without any training."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    plan = read(PLAN)
    manifest = prepare(plan)
    summary = read(OUTPUT / "summary.json")
    require(read(RESULTS / "summary.json") == summary, "Published summary differs")
    require(read(RESULTS / "manifest.json") == manifest, "Published manifest differs")
    settings = read(ROOT / "research/transformer_stress/plan.json")["components"]
    expected = {
        (layers, context, intervention, magnitude)
        for (layers, context), (intervention, magnitude) in itertools.product(
            settings["full_stack_shapes_layers_context"],
            settings["full_stack_interventions"],
        )
        if intervention in plan["interventions"]
    }
    keys = [
        tuple(r[k] for k in ("layers", "context", "intervention", "magnitude"))
        for r in summary["cells"]
    ]
    require(
        len(keys) == len(set(keys)) == plan["expected_cases"] and set(keys) == expected,
        "Wrong full grid",
    )
    files = {}
    for row in summary["cells"]:
        layers, context, intervention, magnitude = (
            row[k] for k in ("layers", "context", "intervention", "magnitude")
        )
        path = OUTPUT / f"L{layers}-N{context}-{intervention}-{magnitude}.json"
        require(read(path) == row, "Summary differs from raw numerical cell")
        replay = clean(cell(settings, plan, layers, context, intervention, magnitude))
        require(replay == row, f"Numerical replay differs: {path}")
        files[str(path.relative_to(ROOT))] = sha256(path)
    for label, field in (("forward_passes", "forward_pass"), ("full_passes", "passed")):
        require(
            summary[label] == sum(r[field] for r in summary["cells"]),
            "Pass count differs",
        )
    require(summary["completed"] == len(keys), "Completion count differs")
    excluded = {
        (layers, context, intervention, magnitude)
        for (layers, context), (intervention, magnitude) in itertools.product(
            settings["full_stack_shapes_layers_context"],
            settings["full_stack_interventions"],
        )
        if intervention not in plan["interventions"]
    }
    actual_excluded = [
        tuple(r[k] for k in ("layers", "context", "intervention", "magnitude"))
        for r in summary["inapplicable_original_cells"]
    ]
    require(
        len(actual_excluded)
        == len(set(actual_excluded))
        == plan["inapplicable_original_cases"]
        and set(actual_excluded) == excluded,
        "Inapplicable cases differ",
    )
    audit = {
        "replayed_forward_backward_cells": len(keys),
        "all_raw_records_exactly_reproduced": True,
        "source_files_verified": len(manifest["source_sha256"]),
        "initial_and_quantized_parameters_data_fingerprints_reproduced": True,
        "summary_sha256": sha256(RESULTS / "summary.json"),
        "manifest_sha256": sha256(RESULTS / "manifest.json"),
        "raw_cells_sha256": files,
        "auditor_sha256": sha256(Path(__file__)),
        "scope": (
            "Repeatability/source/grid audit with the declared FP64-copy reference; "
            "not an independent arbitrary-precision proof."
        ),
    }
    write_json(RESULTS / "audit.json", audit)
    print({k: v for k, v in audit.items() if k != "raw_cells_sha256"})


if __name__ == "__main__":
    main()
