"""Verify the restored x state and bound exposure without treating replay as free."""

import json
from pathlib import Path

import torch

from research.early99.protocol import exposure
from research.early99_dimension_gain.readiness import sha256
from research.early99_reporting.round2_audit import read, require, verify_hash

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = Path(__file__).with_name("recovery.json")
TARGET = 0.99


def declaration() -> dict:
    """Verify every completed cell, first-screen report, plan and backup."""
    record = read(DECLARATION)
    for cell in record["completed_cells_preserved"]:
        for path, digest in cell["files_sha256"].items():
            verify_hash(ROOT / path, digest)
    for path, digest in record["completed_first_screen_reports_sha256"].items():
        verify_hash(ROOT / path, digest)
    for name, digest in record["candidate_plans_sha256"].items():
        verify_hash(ROOT / "runs/early99/candidates" / name / "plan.json", digest)
    verify_hash(
        ROOT / "research/early99_schedule_free/plan.json",
        record["frozen_search_plan_sha256"],
    )
    for path, digest in record["interrupted_cell"]["backup_files_sha256"].items():
        verify_hash(ROOT / path, digest)
    return record


def check_resumption(cell: Path, metadata: dict) -> bool:
    """Allow exactly the declared checkpoint and unchanged execution sources."""
    interrupted = read(DECLARATION)["interrupted_cell"]
    expected = (
        ROOT
        / "runs/early99/candidates"
        / interrupted["candidate"]
        / interrupted["case"]
        / f"seed{interrupted['seed']}"
    )
    resumes = metadata.get("resumptions", [])
    if cell != expected:
        require(not resumes, f"Undeclared resumption: {cell}")
        return False
    require(len(resumes) == 1, "Expected exactly one declared resumption")
    resumed = resumes[0]
    require(
        resumed["completed"] == interrupted["saved_updates"]
        and resumed["device"] == metadata["device"] == "cuda"
        and resumed["compiled_loss"] is True
        and resumed["sources"] == metadata["sources"],
        "Resumption changed checkpoint or execution sources",
    )
    for path, digest in resumed["sources"].items():
        verify_hash(cell / resumed["directory"] / "source" / path, digest)
    return True


def trajectory_accounting() -> dict:
    """Retain canonical epochs and report a lower bound on actual executed work."""
    record = declaration()
    interrupted = record["interrupted_cell"]
    cell = (
        ROOT
        / "runs/early99/candidates"
        / interrupted["candidate"]
        / interrupted["case"]
        / f"seed{interrupted['seed']}"
    )
    backups = interrupted["backup_files_sha256"]
    checkpoint = ROOT / next(p for p in backups if p.endswith("checkpoint.pt"))
    old_history_path = ROOT / next(p for p in backups if p.endswith("history.jsonl"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    old_history = [
        json.loads(line) for line in old_history_path.read_text().splitlines()
    ]
    history = [
        json.loads(line) for line in (cell / "history.jsonl").read_text().splitlines()
    ]
    result = read(cell / "result.json")
    measured = read(cell / "early99_result.json")
    require(check_resumption(cell, result["metadata"]), "Missing declared resumption")
    require(
        saved["completed"] == interrupted["saved_updates"]
        and saved["optimizer"]["param_groups"][0]["training"] is False
        and saved["optimizer"]["param_groups"][0]["step"] == saved["completed"],
        "Backup is not the declared averaged x state",
    )
    require(
        history[: len(saved["history"])] == saved["history"], "Restored prefix changed"
    )
    require(
        old_history[-1]["step"] == interrupted["last_observed_updates"]
        and old_history[-1]["seconds"] == interrupted["last_observed_seconds"]
        and all(row["validation"]["sequence_accuracy"] < TARGET for row in old_history),
        "Discarded observations or crossings changed",
    )
    archives = list(cell.glob("history-before-resume-*.jsonl"))
    require(
        len(archives) == 1
        and archives[0].read_bytes() == old_history_path.read_bytes(),
        "Discarded history archive changed",
    )
    before, observed = (
        interrupted["saved_exposure"],
        interrupted["last_observed_exposure"],
    )
    count, batch = before["train_examples"], before["batch"]
    require(
        exposure(saved["completed"], count, batch) == before
        and exposure(old_history[-1]["step"], count, batch) == observed,
        "Declared exposure changed",
    )
    extra = observed["examples_seen"] - before["examples_seen"]
    canonical = exposure(result["completed"], count, batch)
    return {
        "candidate": interrupted["candidate"],
        "case": interrupted["case"],
        "seed": interrupted["seed"],
        "saved_updates": saved["completed"],
        "last_observed_updates_before_loss": old_history[-1]["step"],
        "known_discarded_updates": old_history[-1]["step"] - saved["completed"],
        "known_discarded_examples": extra,
        "known_discarded_epochs": extra / count,
        "canonical_completed_exposure": canonical,
        "actual_executed_examples_lower_bound": canonical["examples_seen"] + extra,
        "actual_executed_epochs_lower_bound": canonical["epochs"] + extra / count,
        "canonical_first99_epochs": measured["first99_epochs"],
        "actual_first99_epochs_lower_bound": (
            measured["first99_epochs"] + extra / count
            if measured["first99_epochs"] is not None
            else None
        ),
        "actual_exposure_upper_bound": None,
        "unobserved_work_before_loss": "Unknown; actual exposure is a lower bound.",
        "discarded_recorded_crossings": 0,
        "original_segment_last_observed_seconds": old_history[-1]["seconds"],
        "resumed_segment_seconds": result["seconds"],
        "sum_observed_segment_seconds": old_history[-1]["seconds"] + result["seconds"],
        "unobserved_time_before_loss": "Unknown; excluded from observed segment sum.",
        "declaration_sha256": sha256(DECLARATION),
    }
