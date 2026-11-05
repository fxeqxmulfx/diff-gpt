"""Verify provenance, the complete numerical grid, and scalar reference results."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from research.transformer_gradient_stress.run import scalar_checks
from research.transformer_stress.common import ROOT, require, write_json


def main() -> None:
    """Audit the recorded whole-model probes without claiming a second execution."""
    output = ROOT / "runs/transformer-stress/gradient-v1"
    manifest = json.loads((output / "manifest.json").read_text())
    plan = manifest["plan"]
    for name, expected in manifest["source_hashes"].items():
        for path in (ROOT / name, output / "sources" / name):
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"Source changed: {path}",
            )
    scalars = json.loads((output / "scalar.json").read_text())
    full = json.loads((output / "full-stack.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    require(scalars == scalar_checks(plan), "Scalar replay differs")
    scalar_keys = [(row["parameter_dtype"], row["raw_parameter"]) for row in scalars]
    require(
        scalar_keys
        == list(
            itertools.product(
                plan["scalar_parameter_dtypes"], plan["scalar_parameters"]
            )
        ),
        "Scalar grid incomplete or reordered",
    )
    settings = manifest["parent_component_settings"]
    expected_full = [
        (*shape, *intervention)
        for shape, intervention in itertools.product(
            settings["full_stack_shapes_layers_context"],
            settings["full_stack_interventions"],
        )
    ]
    require(
        [(r["layers"], r["context"], r["intervention"], r["magnitude"]) for r in full]
        == expected_full,
        "Full-stack grid incomplete or reordered",
    )
    require(summary["scalar_cases"] == len(scalars), "Scalar count changed")
    require(summary["full_stack_cases"] == len(full), "Stack count changed")
    require(
        summary["all_forwards_identical"]
        == all(r["forward_identical"] for r in [*scalars, *full]),
        "Forward summary differs",
    )
    require(
        summary["all_full_stack_gradients_finite"]
        == all(r["all_gradients_finite"] for r in full),
        "Gradient summary differs",
    )
    for label, records, field in [
        ("corrected_scalar_gradient", scalars, "corrected_relative_gradient_error"),
        ("parent_scalar_gradient", scalars, "parent_relative_gradient_error"),
        (
            "other_parameter_gradient",
            full,
            "other_parameter_gradient_max_relative_error",
        ),
    ]:
        require(
            summary[f"maximum_{label}_relative_error"]
            == max(r[field] for r in records),
            f"Incorrect summary: {label}",
        )
    recovered = sum(
        r["old_frequency_gradient_norm"] == 0 and r["new_frequency_gradient_norm"] > 0
        for r in full
    )
    require(
        recovered == summary["full_stack_frequency_zero_recoveries"],
        "Recovery count differs",
    )
    accepted = (
        all(r["forward_identical"] for r in [*scalars, *full])
        and all(r["all_gradients_finite"] for r in full)
        and max(r["corrected_relative_gradient_error"] for r in scalars)
        <= plan["maximum_scalar_relative_gradient_error"]
        and max(r["other_parameter_gradient_max_relative_error"] for r in full)
        <= plan["maximum_other_parameter_relative_gradient_error"]
    )
    require(accepted == summary["all_declared_checks_pass"], "Acceptance differs")
    audit = {
        "source_files_verified": len(manifest["source_hashes"]),
        "scalar_cases_reproduced": len(scalars),
        "full_stack_records_verified": len(full),
        "full_stack_reexecuted_by_audit": False,
        "all_expected_cases_retained": True,
        "summary_recomputed_from_all_records": True,
        "all_declared_checks_pass": accepted,
        "record_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("scalar.json", "full-stack.json", "summary.json")
        },
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
