"""Audit the gain-only experiment and compare all three seeds with every control."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.cubic_capacity_reporting.audit import (
    execution_bounds,
    verify_checkpoint,
    verify_observations,
)
from research.cubic_usual_gain.factories import (
    ROOT,
    RUNS,
    STUDY,
    case_for,
    configuration,
    factories,
    plan,
)
from research.cubic_usual_gain.readiness import sha256, verify_training_ready
from research.current_grokking.data import build
from research.early99.experiments.compare import read_cell
from research.early99.experiments.initialization import parameter_metadata
from research.early99_optim_reporting.metrics import (
    geometric_mean_bounds,
    paired_cell,
    task_metrics,
)
from research.early99_reporting.round2_audit import (
    read,
    require,
    verify_csv,
    verify_hash,
)

RESULTS = STUDY / "results"
OPTIMIZER_EPS = 1e-8


def audit_seed(seed: int, inventory: dict) -> dict:
    """Check the actual complete native checkpoint and all first-crossing evidence."""
    search, case, cfg = plan(), case_for(), configuration()
    root = RUNS / cfg.name
    directory = root / case.name / f"seed{seed}"
    require(
        not (directory / "error.json").exists(),
        f"Unresolved execution error: {directory}",
    )
    record, metadata = read_cell(root, search["task"], seed)
    require(metadata is not None, f"Missing scientific result: {directory}")
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
    archived_record = read(directory / "early99_result.json")
    require(
        all(archived_record[key] == value for key, value in record.items()),
        "Changed first-99 summary",
    )
    require(
        archived_record["separate_gain_study"]
        and archived_record["experiment"] == metadata["experiment"]
        and archived_record["parameters"] == metadata["parameters"],
        "Incorrect gain-study record",
    )
    verify_observations(history, search["task"], result["completed"])
    require(
        metadata["experiment"] == cfg.metadata(case)
        and metadata["seed"] == seed
        and metadata["batch_seed"] == seed + 10000
        and metadata["observe_updates"] == case.observe
        and metadata["device"] == "cuda"
        and metadata["compiled_loss"]
        and metadata["optimizer"] == "torch.optim.AdamW"
        and metadata["optimizer_betas"] == list(cfg.betas)
        and metadata["optimizer_eps"] == OPTIMIZER_EPS
        and metadata["optimizer_fused"]
        and metadata["warmup_uses_completed_updates"],
        "Executed model, optimizer or schedule changed",
    )
    task = build(case, "cpu")
    require(
        task.metadata == metadata["data_metadata"]
        and task.fingerprints == metadata["data_fingerprints"]
        and task.selection == search["task"]["selection"],
        "Changed data or oracle",
    )
    torch.manual_seed(seed)
    initial = factories().model(task, case, "cpu").model
    initialized = parameter_metadata(initial)
    require(
        all(
            metadata[key] == inventory[key] == value
            for key, value in initialized.items()
        ),
        "Unpaired initialization or changed capacity",
    )
    require(
        initial.token_embedding_table.weight is initial.lm_head.weight, "Untied model"
    )
    verify_checkpoint(
        directory, result, history, initial, cfg.training_case(case).learning_rate
    )
    for source, digest in metadata["sources"].items():
        verify_hash(directory / "source" / source, digest)
    for resumed in metadata.get("resumptions", []):
        require(
            resumed["sources"] == metadata["sources"]
            and resumed["device"] == "cuda"
            and resumed["compiled_loss"],
            "Changed recovery sources or execution",
        )
        for source, digest in resumed["sources"].items():
            verify_hash(directory / resumed["directory"] / "source" / source, digest)
    require("execution_journal" in result, "Missing genuine-update journal")
    work = execution_bounds(directory, result, search["task"])
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
    return {
        "seed": seed,
        "status": record["status"],
        "completed": record["completed"],
        "canonical_first99_epochs": record["first99_epochs"],
        "final_train_accuracy": history[-1]["train"]["sequence_accuracy"],
        "final_heldout_accuracy": record["final_heldout_accuracy"],
        "maximum_observed_heldout_accuracy": max(
            h["validation"]["sequence_accuracy"] for h in history
        ),
        "seconds_last_run_segment": record["seconds_last_run_segment"],
        "parameters": metadata["parameters"],
        **work,
        "artifacts_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in artifacts
        },
    }


def compare_control(control: Path, candidate: Path, search: dict) -> dict:
    """Reuse the strict same-size pairing and censor-aware actual-exposure metrics."""
    require(search["seeds"] == [0, 1, 2], "Changed three-seed comparison")
    pairs = [
        paired_cell(search["task"], seed, control, candidate)
        for seed in search["seeds"]
    ]
    for row in pairs:
        require(
            row["baseline_status"] in {"passed", "budget_exhausted"}
            and row["candidate_status"] in {"passed", "budget_exhausted"},
            "Missing comparison cell",
        )
        # Historical controls have different journaling overhead. Keep their
        # recorded times but do not imply a controlled wall-clock comparison.
        row["seconds_ratio"] = None
    low, high = geometric_mean_bounds(pairs, "epoch_ratio_lower", "epoch_ratio_upper")
    return {
        "expected_cells": len(search["seeds"]),
        "baseline_directory": str(control.relative_to(ROOT)),
        "candidate_directory": str(candidate.relative_to(ROOT)),
        "task_metrics": task_metrics(search["task"]["name"], pairs),
        "geometric_mean_epoch_ratio_lower": low,
        "geometric_mean_epoch_ratio_upper": high,
        "geometric_mean_epoch_ratio": low if low == high else None,
        "runs": pairs,
    }


def main() -> None:
    """Close the one-task gain study without changing the fixed75 selection."""
    validation = verify_training_ready()
    search, cfg = plan(), configuration()
    root = RUNS / cfg.name
    progress = read(RUNS / "progress.json")
    require(
        progress["phase"] == "training_finished_requires_audit"
        and len(progress["children"]) == len(search["seeds"])
        and {r["seed"] for r in progress["children"]} == set(search["seeds"])
        and all(
            r.get("exit_code") == 0 or r.get("reused_completed")
            for r in progress["children"]
        ),
        "Incomplete dispatcher",
    )
    archived = read(root / "plan.json")
    require(
        archived["study_plan_sha256"] == sha256(STUDY / "plan.json")
        and archived["configuration"] == cfg.__dict__
        and archived["task"] == search["task"]
        and archived["source_sha256"] == validation["source_sha256"]
        and archived["execution"]
        == {
            "device": "cuda",
            "compiled_loss": True,
            "cuda_graphs": True,
            "durable_per_update_journal": True,
        },
        "Changed frozen experiment",
    )
    for source, digest in archived["source_sha256"].items():
        verify_hash(root / "experiment_source" / source, digest)
    inventory = {r["seed"]: r for r in read(RESULTS / "inventory.json")["rows"]}
    audits = [audit_seed(seed, inventory[seed]) for seed in search["seeds"]]
    reports = {
        name: compare_control(ROOT / directory, root, search)
        for name, directory in search["controls"].items()
    }
    require(
        len(
            {
                json.dumps(
                    [
                        {k: v for k, v in row.items() if k.startswith("candidate_")}
                        for row in report["runs"]
                    ],
                    sort_keys=True,
                )
                for report in reports.values()
            }
        )
        == 1,
        "Candidate differs across control reports",
    )
    paths = []
    for name, report in reports.items():
        path = RESULTS / f"comparison-{name}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        paths.append(path)
        path = path.with_suffix(".csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(report["runs"][0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(report["runs"])
        verify_csv(path, report["runs"])
        paths.append(path)
    record = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "new_cells": len(audits),
        "reused_controls": len(search["controls"]) * len(search["seeds"]),
        "passed": sum(r["status"] == "passed" for r in audits),
        "audits": audits,
        "scope": search["scope"],
        "fixed75_selection_changed": False,
        "new_long_suite_requested": False,
        "timing_limits": (
            "Historical controls predate the new durable execution journal. "
            "Recorded segment times are retained; runtime ratios are omitted."
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                Path(__file__),
                ROOT / "research/cubic_capacity_reporting/audit.py",
                ROOT / "research/early99_optim_reporting/metrics.py",
                ROOT / "research/early99/experiments/compare.py",
            ]
        },
        "evidence_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                *paths,
                root / "plan.json",
                RUNS / "progress.json",
                STUDY / "plan.json",
                RESULTS / "validation.json",
                ROOT / "research/cubic_capacity/results/complete-audit.json",
            ]
        },
    }
    (RESULTS / "complete-audit.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: record[key]
                for key in ["complete", "new_cells", "reused_controls", "passed"]
            }
        )
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
