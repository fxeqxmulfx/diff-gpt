"""Measure all declared initialized shapes with the previously validated observer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.cubic_usual_gain.readiness import require_committed, sha256
from research.current_grokking.data import build, tensor_digest
from research.current_grokking.protocol import Case
from research.early99.experiments.factories import (
    baseline_model,
    peri_rms_scaled_ssmax_model,
)
from research.early99.experiments.initialization import parameter_metadata
from research.early99_dimension_gain.factories import dimension_model
from research.early99_reporting.round2_audit import read, require, verify_hash
from research.transformer_path_diagnostic.diagnose import observe, state_fingerprints

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
FACTORIES = {
    "original": baseline_model,
    "normalized_inverse": dimension_model,
    "normalized_usual": peri_rms_scaled_ssmax_model,
}


def verify_inputs() -> dict:
    """Bind all reused model, observer and validation sources to committed bytes."""
    declared = read(STUDY / "plan.json")
    for group in ["source_sha256", "evidence_sha256"]:
        for path, digest in declared[group].items():
            verify_hash(ROOT / path, digest)
    require_committed(
        [
            STUDY / "plan.json",
            Path(__file__),
            STUDY / "__init__.py",
            *[ROOT / p for p in declared["source_sha256"]],
            *[ROOT / p for p in declared["evidence_sha256"]],
        ]
    )
    return declared


def main() -> None:
    """Complete all 36 states, preserving unmodified weights and all observed sites."""
    declared = verify_inputs()
    destination = RESULTS / "measurements.json"
    require(not destination.exists(), "Measurements already exist; do not overwrite")
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    rows = []
    for shape in declared["shapes"]:
        case = Case(**shape["case"])
        task = build(case, "cpu")
        require(
            task.fingerprints == declared["task"]["data_fingerprints"],
            "Changed task data",
        )
        batch = task.inputs(torch.tensor(declared["training_indices"]))
        for seed in declared["seeds"]:
            expected, expected_rng = None, None
            for arm in declared["arms"]:
                torch.manual_seed(seed)
                model = FACTORIES[arm](task, case, "cpu")
                fingerprints = state_fingerprints(model.model)["parameters"]
                rng = torch.get_rng_state().clone()
                if expected is None:
                    expected, expected_rng = fingerprints, rng
                require(
                    fingerprints == expected
                    and list(fingerprints) == list(expected)
                    and torch.equal(rng, expected_rng),
                    "Changed same-size initialization/order/RNG",
                )
                metadata = parameter_metadata(model.model)
                require(
                    metadata["parameters"] == shape["unique_parameters"],
                    "Changed capacity",
                )
                require(
                    model.model.token_embedding_table.weight
                    is model.model.lm_head.weight,
                    "Untied weights",
                )
                measurement = observe(model, task, batch, declared["probe_seed"])
                rows.append(
                    {
                        "size": shape["name"],
                        "arm": arm,
                        "seed": seed,
                        "layers": case.layers,
                        "width": case.width,
                        "heads": case.heads,
                        "batch_sha256": tensor_digest(batch),
                        **metadata,
                        **measurement,
                    }
                )
                print(
                    json.dumps(
                        {
                            "size": shape["name"],
                            "arm": arm,
                            "seed": seed,
                            "sites": len(measurement["executed_sites"]),
                            "exact_parity": True,
                        }
                    ),
                    flush=True,
                )
    require(
        len(rows) == declared["state_batch_measurements"],
        "Incomplete initialization audit",
    )
    require(
        len({row["batch_sha256"] for row in rows}) == 1
        and len({row["probe_sha256"] for row in rows}) == 1,
        "Unpaired input/probe",
    )
    verify_inputs()
    record = {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "state_batch_measurements": len(rows),
        "objective_measurements": 2 * len(rows),
        "scientific_optimizer_updates": 0,
        "new_heldout_scores": False,
        "plan_sha256": sha256(STUDY / "plan.json"),
        "measurement_source_sha256": sha256(Path(__file__)),
        "rows": rows,
        "limits": declared["analysis_limits"],
    }
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
