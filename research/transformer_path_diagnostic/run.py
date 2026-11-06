"""Measure the declared old/new states using fixed training examples only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.current_grokking.data import build, tensor_digest
from research.early99.experiments.factories import baseline_model
from research.early99.experiments.initialization import parameter_metadata
from research.early99.protocol import selected, specification
from research.early99_dimension_gain.factories import dimension_model
from research.early99_reporting.round2_audit import require, verify_hash
from research.transformer_path_diagnostic.diagnose import observe, state_fingerprints

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).parent
RESULTS = STUDY / "results"
VALIDATION_TESTS = 5


def digest(path: Path) -> str:
    """Hash a complete source or checkpoint without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(plan: dict) -> None:
    """Verify every pinned input and immutable model dependency."""
    for group in ["existing_artifact_sha256", "existing_source_sha256"]:
        for path, expected in plan[group].items():
            verify_hash(ROOT / path, expected)


def main() -> None:
    """Run all 60 state/batch measurements after a committed validation freeze."""
    require(Path.cwd().resolve() == ROOT, "Run from the repository root")
    record_path = RESULTS / "validation.json"
    record = json.loads(record_path.read_text())
    require(
        record["passed"] == VALIDATION_TESTS and record["failed"] == 0,
        "Complete all five diagnostic checks first",
    )
    for path, expected in record["files_sha256"].items():
        verify_hash(ROOT / path, expected)
    for path in [record_path, *STUDY.glob("*.py"), STUDY / "plan.json"]:
        committed = subprocess.run(  # noqa: S603 -- Exact local committed source.
            ["/usr/bin/git", "show", f"HEAD:{path.relative_to(ROOT)}"],
            check=True,
            capture_output=True,
        ).stdout
        require(
            committed == path.read_bytes(),
            f"Commit validated diagnostic before measurements: {path}",
        )
    output_path = RESULTS / "measurements.json"
    require(not output_path.exists(), "Diagnostic measurements already exist")
    plan = json.loads((STUDY / "plan.json").read_text())
    validate_inputs(plan)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    task_row = next(row for row in selected("long") if row["name"] == plan["task"])
    case = specification(task_row)
    task = build(case, "cpu")
    require(task.fingerprints == task_row["data_fingerprints"], "Changed task data")
    batches = [
        task.inputs(
            torch.arange(
                index * plan["training_batches"]["size"],
                (index + 1) * plan["training_batches"]["size"],
            )
        )
        for index in range(plan["training_batches"]["count"])
    ]
    rows = []
    for seed in plan["seeds"]:
        initial_parameters = None
        initial_rng = None
        states = [("initial", arm, None) for arm in plan["initial_arms"]]
        states += [
            ("final", arm, ROOT / directory / plan["task"] / f"seed{seed}")
            for arm, directory in plan["trained_runs"].items()
        ]
        for stage, arm, directory in states:
            torch.manual_seed(seed)
            factory = (
                dimension_model if arm.startswith("normalized") else baseline_model
            )
            model = factory(task, case, "cpu")
            initialized = state_fingerprints(model.model)["parameters"]
            initial_metadata = parameter_metadata(model.model)
            if initial_parameters is None:
                initial_parameters, initial_rng = (
                    initialized,
                    torch.get_rng_state().clone(),
                )
            require(
                initialized == initial_parameters
                and list(initialized) == list(initial_parameters)
                and torch.equal(initial_rng, torch.get_rng_state()),
                "Initial parameters/order or random draws differ",
            )
            checkpoint_info = None
            if directory is not None:
                saved = torch.load(
                    directory / "checkpoint.pt", map_location="cpu", weights_only=True
                )
                result = json.loads((directory / "early99_result.json").read_text())
                require(
                    initial_metadata["initial_parameters_sha256"]
                    == result["initial_parameters_sha256"],
                    "Regenerated initialization differs from original run",
                )
                require(
                    saved["completed"] == result["completed"]["updates"],
                    "Checkpoint/result exposure mismatch",
                )
                require(
                    saved["metadata"]["data_fingerprints"] == task.fingerprints,
                    "Checkpoint data differ",
                )
                model.model.load_state_dict(saved["model"], strict=True)
                checkpoint_info = {
                    "directory": str(directory.relative_to(ROOT)),
                    "completed": result["completed"],
                    "status": result["status"],
                    "first99_epochs": result["first99_epochs"],
                    "historical_final_heldout_accuracy": result[
                        "final_heldout_accuracy"
                    ],
                    "checkpoint_sha256": digest(directory / "checkpoint.pt"),
                }
            for batch_index, batch in enumerate(batches):
                measurement = observe(
                    model, task, batch, plan["probe_seed"] + batch_index
                )
                rows.append(
                    {
                        "stage": stage,
                        "arm": arm,
                        "seed": seed,
                        "batch": batch_index,
                        "batch_sha256": tensor_digest(batch),
                        "checkpoint": checkpoint_info,
                        **measurement,
                    }
                )
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "arm": arm,
                            "seed": seed,
                            "batch": batch_index,
                            "sites": len(measurement["executed_sites"]),
                            "task_loss": measurement["objectives"]["task_loss"][
                                "value"
                            ],
                            "exact_parity": True,
                        }
                    ),
                    flush=True,
                )
    expected = (
        len(plan["seeds"])
        * (len(plan["initial_arms"]) + len(plan["trained_runs"]))
        * len(batches)
    )
    require(len(rows) == expected, "Incomplete diagnostic grid")
    validate_inputs(plan)
    result = {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "state_batch_measurements": len(rows),
        "objective_measurements": 2 * len(rows),
        "optimizer_updates": 0,
        "new_heldout_scores": False,
        "selection_uses_this_diagnostic": False,
        "plan_sha256": digest(STUDY / "plan.json"),
        "validation_sha256": digest(record_path),
        "torch": str(torch.__version__),
        "rows": rows,
        "limitations": plan["limitations"],
    }
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
