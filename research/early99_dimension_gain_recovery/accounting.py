"""Bind the one interrupted trajectory to preserved checkpoints and segment clocks."""

import json
from pathlib import Path

import torch

from research.early99.protocol import exposure
from research.early99_dimension_gain_reporting.run_screens import ROOT, RUNS
from research.early99_reporting.round2_audit import read, require, verify_hash

TARGET = 0.99
RECOVERY = Path(__file__).with_name("recovery.json")


def declaration() -> dict:
    """Verify the original choice, launch, completed cells and interrupted backups."""
    record = read(RECOVERY)
    original = ROOT / "research/early99_dimension_gain/results"
    for path, field in [
        (original / "shared-selection.json", "frozen_selection_sha256"),
        (original / "long-launch.json", "original_launch_sha256"),
        (
            ROOT / "research/early99_dimension_gain_reporting/run_long.py",
            "original_launcher_sha256",
        ),
        (RUNS / record["selected_candidate"] / "plan.json", "candidate_plan_sha256"),
    ]:
        verify_hash(path, record[field])
    for cell in record["completed_long_cells_preserved"]:
        for path, digest in cell["files_sha256"].items():
            verify_hash(ROOT / path, digest)
    for path, digest in record["interrupted_cell"]["backup_files_sha256"].items():
        verify_hash(ROOT / path, digest)
    return record


def check_cell(cell: Path, metadata: dict) -> bool:
    """Permit exactly the declared restored checkpoint; reject other resumptions."""
    record = read(RECOVERY)
    interrupted = record["interrupted_cell"]
    expected = RUNS / record["selected_candidate"] / interrupted["case"] / "seed0"
    resumes = metadata.get("resumptions", [])
    if cell != expected:
        require(not resumes, f"Undeclared resumption: {cell}")
        return False
    require(len(resumes) == 1, "Expected exactly one declared resumption")
    resumed = resumes[0]
    require(
        resumed["completed"] == interrupted["saved_completed_updates"]
        and resumed["device"] == metadata["device"] == "cuda"
        and resumed["compiled_loss"] is True,
        "Resume changed execution or checkpoint position",
    )
    require(resumed["sources"] == metadata["sources"], "Resume sources changed")
    for source, digest in resumed["sources"].items():
        verify_hash(cell / resumed["directory"] / "source" / source, digest)
    return True


def trajectory_accounting() -> dict:
    """Preserve discarded observations and report canonical and actual exposure."""
    record = declaration()
    interrupted = record["interrupted_cell"]
    cell = RUNS / record["selected_candidate"] / interrupted["case"] / "seed0"
    backups = interrupted["backup_files_sha256"]
    backup = ROOT / next(path for path in backups if path.endswith("checkpoint.pt"))
    old = torch.load(backup, map_location="cpu", weights_only=True)
    before = ROOT / next(path for path in backups if path.endswith("history.jsonl"))
    lost_history = [json.loads(line) for line in before.read_text().splitlines()]
    history = [
        json.loads(line) for line in (cell / "history.jsonl").read_text().splitlines()
    ]
    result = read(cell / "result.json")
    require(check_cell(cell, result["metadata"]), "Missing recovered cell")
    saved_updates = interrupted["saved_completed_updates"]
    last_updates = interrupted["last_observed_completed_updates"]
    require(old["completed"] == saved_updates, "Backup checkpoint position changed")
    require(lost_history[-1]["step"] == last_updates, "Lost observation changed")
    require(history[: len(old["history"])] == old["history"], "Restored prefix changed")
    require(
        all(row["validation"]["sequence_accuracy"] < TARGET for row in lost_history),
        "A crossing was lost before recovery",
    )
    archived = list(cell.glob("history-before-resume-*.jsonl"))
    require(len(archived) == 1, "Unexpected history archive count")
    require(
        archived[0].read_bytes() == before.read_bytes(), "Discarded history changed"
    )
    train = interrupted["saved_exposure"]["train_examples"]
    batch = interrupted["saved_exposure"]["batch"]
    saved_exposure = exposure(saved_updates, train, batch)
    observed_exposure = exposure(last_updates, train, batch)
    require(saved_exposure == interrupted["saved_exposure"], "Saved exposure changed")
    require(
        observed_exposure == interrupted["last_observed_exposure"],
        "Raw exposure changed",
    )
    canonical = exposure(result["completed"], train, batch)
    extra = observed_exposure["examples_seen"] - saved_exposure["examples_seen"]
    old_seconds = lost_history[-1]["seconds"]
    require(
        old_seconds == interrupted["last_observed_seconds"]
        and old["history"][-1]["seconds"] == interrupted["saved_history_seconds"],
        "Saved segment clock changed",
    )
    return {
        "case": interrupted["case"],
        "seed": interrupted["seed"],
        "resumptions": 1,
        "saved_updates": saved_updates,
        "discarded_updates_replayed": last_updates - saved_updates,
        "discarded_examples_replayed": extra,
        "discarded_epochs_replayed": extra / train,
        "canonical_completed_exposure": canonical,
        "actual_executed_examples_lower_bound": canonical["examples_seen"] + extra,
        "actual_executed_epochs_lower_bound": canonical["epochs"] + extra / train,
        "exposure_caveat": (
            "The last durable observation proves 50 discarded updates. "
            "Any further updates before process loss were unobserved; "
            "total actual work is therefore a lower bound, not an exact total."
        ),
        "original_segment_last_observed_seconds": old_seconds,
        "resumed_segment_seconds": result["seconds"],
        "sum_of_observed_segment_seconds": old_seconds + result["seconds"],
        "unobserved_time_before_process_loss": "Unknown; not included in segment sum",
        "first99_lost_before_resume": False,
        "first99_actual_executed_epochs_lower_bound": (
            read(cell / "early99_result.json")["first99_epochs"] + extra / train
            if read(cell / "early99_result.json")["first99_epochs"] is not None
            else None
        ),
        "old_history_rows_retained": len(lost_history),
        "restored_history_prefix_rows": len(old["history"]),
    }
