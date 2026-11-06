"""Audit every declared capacity cell from immutable sources and native states."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.cubic_capacity.factories import (
    ROOT,
    RUNS,
    STUDY,
    configuration,
    factories,
    plan,
    task_row,
)
from research.cubic_capacity.readiness import sha256, verify_training_ready
from research.current_grokking.data import build
from research.current_grokking.protocol import Case
from research.early99.experiments.compare import read_cell
from research.early99.experiments.initialization import parameter_metadata
from research.early99.protocol import exposure
from research.early99_execution.journal import verify_saved_journal
from research.early99_reporting.round2_audit import read, require, verify_hash

RESULTS = STUDY / "results"
TARGET = 0.99
OPTIMIZER_EPS = 1e-8


def verify_observations(history: list[dict], row: dict, completed: int) -> None:
    """Require complete exact oracles, fixed cadence and first-crossing termination."""
    interval = row["case"]["observe"]
    expected = list(range(0, completed + 1, interval))
    if completed % interval:
        expected.append(completed)
    require([h["step"] for h in history] == expected, "Observation cadence changed")
    for observation in history:
        for split, count in [
            ("train", row["train_examples"]),
            ("validation", row["heldout_examples"]),
        ]:
            metrics = observation[split]
            require(
                metrics["examples"] == count
                and 0 <= metrics["exact"] <= count
                and math.isfinite(metrics["loss"])
                and math.isclose(
                    metrics["sequence_accuracy"],
                    metrics["exact"] / count,
                    rel_tol=0,
                    abs_tol=1e-12,
                ),
                "Incomplete or inconsistent exact oracle",
            )
    crossings = [
        h["step"] for h in history if h["validation"]["sequence_accuracy"] >= TARGET
    ]
    require(
        completed == (crossings[0] if crossings else row["case"]["max_updates"]),
        "Did not stop at the first crossing or the original budget",
    )


def execution_bounds(directory: Path, result: dict, row: dict) -> dict:
    """Retain genuine work and explicit uncertainty for any unjournaled recovery."""
    resumptions = result["metadata"].get("resumptions", [])
    lower, upper = 0, 0
    journal = result.get("execution_journal")
    if journal is not None:
        verified = verify_saved_journal(
            directory,
            result["completed"],
            row["train_examples"],
            row["case"]["batch"],
            resumptions,
            journal,
        )
        lower = verified["extra_examples_lower"]
        upper = verified["extra_examples_upper"]
    elif resumptions:
        upper = None
    return {
        "journaled": journal is not None,
        "resumptions": len(resumptions),
        "replay_examples_lower": lower,
        "replay_examples_upper": upper,
        "replay_epochs_lower": lower / row["train_examples"],
        "replay_epochs_upper": upper / row["train_examples"]
        if upper is not None
        else None,
    }


def verify_checkpoint(
    directory: Path,
    result: dict,
    history: list[dict],
    expected_model: torch.nn.Module,
    peak_rate: float,
) -> None:
    """Check native state completeness, tensor shapes, tying, moments and final rate."""
    saved = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    require(
        saved["completed"] == result["completed"]
        and saved["history"] == history
        and saved["metadata"] == result["metadata"],
        "Checkpoint, history and result disagree",
    )
    expected = expected_model.state_dict()
    actual = saved["model"]
    require(actual.keys() == expected.keys(), "Checkpoint model fields changed")
    for name, value in actual.items():
        require(
            value.shape == expected[name].shape
            and value.dtype == expected[name].dtype
            and torch.isfinite(value).all().item(),
            f"Invalid model tensor: {name}",
        )
    require(
        torch.equal(actual["token_embedding_table.weight"], actual["lm_head.weight"]),
        "Tied lookup/readout values disagree",
    )
    optimizer = saved["optimizer"]
    require(len(optimizer["param_groups"]) == 1, "Unexpected AdamW parameter groups")
    group = optimizer["param_groups"][0]
    indices = group["params"]
    parameters = list(expected_model.parameters())
    require(
        len(indices) == len(set(indices)) == len(parameters)
        and set(optimizer["state"]) == set(indices),
        "Duplicated or missing optimizer states",
    )
    case = result["metadata"]["case"]
    expected_rate = peak_rate * min(1.0, (result["completed"] - 1) / case["warmup"])
    require(
        tuple(group["betas"]) == (0.9, 0.98)
        and group["eps"] == OPTIMIZER_EPS
        and group["weight_decay"] == case["weight_decay"]
        and group["fused"]
        and group["capturable"]
        and math.isclose(float(group["lr"]), expected_rate, rel_tol=1e-7),
        "Optimizer settings changed",
    )
    for index, parameter in zip(indices, parameters, strict=True):
        state = optimizer["state"][index]
        first, second = state["exp_avg"], state["exp_avg_sq"]
        require(
            int(state["step"]) == result["completed"]
            and first.shape == second.shape == parameter.shape
            and first.dtype == second.dtype == torch.float32
            and torch.isfinite(first).all().item()
            and torch.isfinite(second).all().item()
            and (second >= 0).all().item(),
            "Invalid final AdamW moments or update count",
        )


def audit_cell(root: Path, row: dict, seed: int, spec: dict, shape: dict) -> dict:
    """Reconstruct one complete cell; a larger model is an explicit intervention."""
    directory = root / row["name"] / f"seed{seed}"
    require(not (directory / "error.json").exists(), f"Unresolved error: {directory}")
    record, metadata = read_cell(root, row, seed)
    require(metadata is not None, f"Unfinished cell: {directory}")
    result = read(directory / "result.json")
    history = [
        json.loads(line)
        for line in (directory / "history.jsonl").read_text().splitlines()
    ]
    require(
        metadata == read(directory / "protocol.json")
        and result["endpoint"] == history[-1],
        "Protocol or endpoint changed",
    )
    saved_record = read(directory / "early99_result.json")
    require(
        all(saved_record[k] == v for k, v in record.items()), "Changed first-99 report"
    )
    verify_observations(history, row, result["completed"])
    require(
        metadata["seed"] == seed
        and metadata["batch_seed"] == seed + 10000
        and metadata["parameters"] == shape["unique_parameters"]
        and metadata["observe_updates"] == row["case"]["observe"]
        and metadata["optimizer"] == "torch.optim.AdamW"
        and metadata["optimizer_betas"] == [0.9, 0.98]
        and metadata["optimizer_eps"] == OPTIMIZER_EPS
        and metadata["optimizer_fused"]
        and metadata["compiled_loss"]
        and metadata["warmup_uses_completed_updates"]
        and metadata["device"] == "cuda",
        "Unplanned training or capacity metadata",
    )
    case = Case(**row["case"])
    task = build(case, "cpu")
    require(
        task.fingerprints == metadata["data_fingerprints"]
        and task.metadata == metadata["data_metadata"]
        and task.selection == row["selection"],
        "Data, split or task rule changed",
    )
    torch.manual_seed(seed)
    initial = factories(spec).model(task, case, "cpu").model
    initial_metadata = parameter_metadata(initial)
    require(
        all(metadata[key] == value for key, value in initial_metadata.items()),
        "Executed initialization differs from unchanged construction",
    )
    if not shape["reuse_existing"]:
        declared = next(
            item
            for item in read(RESULTS / "inventory.json")["rows"]
            if item["name"] == spec["name"] and item["seed"] == seed
        )
        require(
            all(declared[key] == value for key, value in initial_metadata.items()),
            "Initialization differs from the committed pretraining inventory",
        )
        require(
            metadata["experiment"] == configuration(spec).metadata(case),
            "Changed experiment",
        )
        require(result.get("execution_journal") is not None, "Missing durable journal")
        require(
            saved_record["capacity_specification"] == spec
            and saved_record["separate_capacity_study"]
            and saved_record["parameters"] == metadata["parameters"]
            and saved_record["experiment"] == metadata["experiment"],
            "Capacity record changed",
        )
    for source, expected in metadata["sources"].items():
        verify_hash(directory / "source" / source, expected)
    for resumed in metadata.get("resumptions", []):
        require(
            resumed["sources"] == metadata["sources"]
            and resumed["device"] == "cuda"
            and resumed["compiled_loss"],
            "Resumption changed execution",
        )
        for source, expected in resumed["sources"].items():
            verify_hash(directory / resumed["directory"] / "source" / source, expected)
    peak_rate = case.learning_rate * spec["learning_rate_multiplier"]
    verify_checkpoint(directory, result, history, initial, peak_rate)
    work = execution_bounds(directory, result, row)
    canonical = record["completed"]["epochs"]
    low = canonical + work["replay_epochs_lower"]
    high = (
        canonical + work["replay_epochs_upper"]
        if work["replay_epochs_upper"] is not None
        else None
    )
    passed = record["status"] == "passed"
    first_train99 = next(
        (h["step"] for h in history if h["train"]["sequence_accuracy"] >= TARGET), None
    )
    require(first_train99 == result["first_train99"], "Training crossing disagrees")
    artifacts = [
        directory / name
        for name in [
            "checkpoint.pt",
            "result.json",
            "early99_result.json",
            "protocol.json",
            "history.jsonl",
        ]
    ]
    artifacts.extend(directory.glob("*journal*.json"))
    artifacts.append(root / "plan.json")
    return {
        "size": shape["name"],
        "arm": spec["arm"],
        "seed": seed,
        "directory": str(directory.relative_to(ROOT)),
        "reused": shape["reuse_existing"],
        "layers": case.layers,
        "width": case.width,
        "heads": case.heads,
        "head_width": case.width // case.heads,
        "parameters": metadata["parameters"],
        "peak_learning_rate": peak_rate,
        "status": record["status"],
        "completed_updates": result["completed"],
        "canonical_epochs": canonical,
        "actual_epochs_lower": low,
        "actual_epochs_upper": high,
        "first99_epochs_lower": low if passed else None,
        "first99_epochs_upper": high if passed else None,
        "first99_epochs": low if passed and low == high else None,
        "canonical_first99_epochs": record["first99_epochs"],
        "canonical_previous_observed_epochs": (record["first99"] or {}).get(
            "previous_observed_epochs"
        ),
        "first_train99_updates": first_train99,
        "first_train99_canonical_epochs": exposure(
            first_train99, row["train_examples"], case.batch
        )["epochs"]
        if first_train99 is not None
        else None,
        "final_train_accuracy": history[-1]["train"]["sequence_accuracy"],
        "final_heldout_accuracy": record["final_heldout_accuracy"],
        "maximum_observed_heldout_accuracy": max(
            h["validation"]["sequence_accuracy"] for h in history
        ),
        "seconds_last_run_segment": record["seconds_last_run_segment"],
        "uninterrupted_runtime": not metadata.get("resumptions"),
        "initial_parameters_sha256": metadata["initial_parameters_sha256"],
        "torch": metadata["torch"],
        **work,
        "artifacts_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in artifacts
        },
    }


def audit() -> dict:
    """Verify the declared 27 new and nine reused cells without retraining."""
    validation = verify_training_ready()
    search = plan()
    progress = read(RUNS / "progress.json")
    expected_children = {
        (spec["name"], seed)
        for spec in search["configurations"]
        for seed in search["seeds"]
    }
    require(
        progress["phase"] == "training_finished_requires_audit"
        and len(progress["children"]) == len(expected_children)
        and {(row["name"], row["seed"]) for row in progress["children"]}
        == expected_children
        and all(
            row.get("exit_code") == 0 or row.get("reused_completed")
            for row in progress["children"]
        ),
        "The complete dispatcher has not finished successfully",
    )
    rows = []
    for shape in search["shapes"]:
        for arm in search["arms"]:
            spec = next(
                s
                for s in search["configurations"]
                if s["arm"] == arm["name"]
                and s["size"] == ("depth" if shape["reuse_existing"] else shape["name"])
            )
            if shape["reuse_existing"]:
                root = ROOT / search["controls"][arm["name"]]
                row = search["task"]
            else:
                root = RUNS / spec["name"]
                row = task_row(spec)
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
                    "Archived experiment plan changed",
                )
                for source, expected in archived["source_sha256"].items():
                    verify_hash(root / "experiment_source" / source, expected)
            rows.extend(
                audit_cell(root, row, seed, spec, shape) for seed in search["seeds"]
            )
    require(
        len(rows) == search["new_training_cells"] + search["reused_current_size_cells"],
        "Incomplete study",
    )
    for shape in search["shapes"]:
        for seed in search["seeds"]:
            paired = [
                r for r in rows if r["size"] == shape["name"] and r["seed"] == seed
            ]
            require(
                len(paired) == len(search["arms"])
                and len({r["initial_parameters_sha256"] for r in paired}) == 1,
                "Same-size initializations differ",
            )
    return {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "new_training_cells": sum(not r["reused"] for r in rows),
        "reused_controls": sum(r["reused"] for r in rows),
        "rows": rows,
        "plan_sha256": sha256(STUDY / "plan.json"),
        "validation_sha256": sha256(RESULTS / "validation.json"),
        "dispatcher_progress_sha256": sha256(RUNS / "progress.json"),
        "auditor_sha256": sha256(Path(__file__)),
        "checked": [
            "sources",
            "initialization",
            "data",
            "capacity",
            "native_optimizer_and_moments",
            "fixed_cadence_and_exact_oracles",
            "first_crossing_or_budget",
            "all_executed_work",
        ],
        "limits": search["scope"],
    }


if __name__ == "__main__":
    torch.set_num_threads(1)
    result = audit()
    (RESULTS / "complete-audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["complete", "new_training_cells", "reused_controls"]
            }
        )
    )
