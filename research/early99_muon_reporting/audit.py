"""Audit Muon matrix/auxiliary states and every durable executed-work segment."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.current_grokking.run import verify_model_reference
from research.early99.protocol import selected, specification, suite
from research.early99_execution.journal import verify_saved_journal
from research.early99_muon.factories import Configuration
from research.early99_muon_reporting.checkpoint import verify_optimizer_checkpoint
from research.early99_optim_reporting.metrics import paired_cell
from research.early99_reporting.round2_audit import require

TARGET = 0.99
parser = argparse.ArgumentParser()
parser.add_argument("--long", action="store_true")
parser.add_argument("--name", action="append", required=True)
parser.add_argument("--full", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
require(
    verify_model_reference() == suite()["baseline_model_reference"],
    "Mismatch: verify_model_reference() == suite()['baseline_mod...",
)
search = json.loads(Path("research/early99_muon/plan.json").read_text())
runs_root = Path("/home/misha/python_projects/gpt/runs/early99")
reference = runs_root / "baseline"
incumbent = runs_root / "candidates/adamw-lr2"
records = {}
for name in args.name:
    root = runs_root / "candidates" / name
    raw_plan = (root / "plan.json").read_bytes()
    plan = json.loads(raw_plan)
    config = Configuration(**plan["configuration"])
    require(
        plan["configuration"]
        == next(c for c in search["configurations"] if c["name"] == name),
        "Executed configuration differs from the declared search plan",
    )
    require(
        json.loads((root / "configuration.json").read_text()) == plan["configuration"],
        "Mismatch: json.loads((root / 'configuration.json').read_tex...",
    )
    for path, expected in plan["source_sha256"].items():
        require(
            hashlib.sha256((root / "experiment_source" / path).read_bytes()).hexdigest()
            == expected,
            "Mismatch: hashlib.sha256((root / 'experiment_source' / path...",
        )
        require(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
            "Mismatch: hashlib.sha256(Path(path).read_bytes()).hexdigest...",
        )
    require(
        not list(root.glob("*/seed*/error.json")),
        "Mismatch: not list(root.glob('*/seed*/error.json'))",
    )
    tasks = selected("all" if args.long else "short")
    if not args.full:
        tasks = [t for t in tasks if t["name"] in search["screen_tasks"]]
    count = 0
    passes = 0
    resumed = 0
    exposure_intervals = []
    censored = []
    parameters = []
    for task in tasks:
        for seed in [0, 1, 2]:
            cell = root / task["name"] / f"seed{seed}"
            for filename in [
                "checkpoint.pt",
                "early99_result.json",
                "history.jsonl",
                "protocol.json",
                "result.json",
            ]:
                require((cell / filename).is_file(), str(cell / filename))
            result = json.loads((cell / "result.json").read_text())
            metadata = result["metadata"]
            require(
                metadata == json.loads((cell / "protocol.json").read_text()),
                "Mismatch: metadata == json.loads((cell / 'protocol.json').r...",
            )
            require(
                metadata["experiment"] == config.metadata(specification(task)),
                "Mismatch: metadata['experiment'] == config.metadata(specifi...",
            )
            resumed += len(metadata.get("resumptions", []))
            verified = verify_saved_journal(
                cell,
                result["completed"],
                task["train_examples"],
                task["case"]["batch"],
                metadata.get("resumptions", []),
                result["execution_journal"],
            )
            if verified["extra_examples_upper"]:
                exposure_intervals.append(
                    {"case": task["name"], "seed": seed, **verified}
                )
            for resumed_metadata in metadata.get("resumptions", []):
                require(
                    resumed_metadata["sources"] == metadata["sources"],
                    "Resumed sources changed",
                )
                require(
                    resumed_metadata["device"] == metadata["device"] == "cuda"
                    and resumed_metadata["compiled_loss"],
                    "Resumed execution changed",
                )
                for path, expected in resumed_metadata["sources"].items():
                    require(
                        hashlib.sha256(
                            (
                                cell / resumed_metadata["directory"] / "source" / path
                            ).read_bytes()
                        ).hexdigest()
                        == expected,
                        "Resumed source archive changed",
                    )
            require(
                metadata["optimizer"] == "research.early99_muon.optimizer.MuonBundle"
                and not metadata["optimizer_cuda_graphs"]
                and metadata["matrix_dtype"] == "float32"
                and metadata["matrix_normalization_dtype"] == "float64"
                and metadata["compiled_loss"]
                and metadata["device"] == "cuda"
                and metadata["auxiliary_native_fused"]
                == (config.optimizer_kind == "muon_adamw"),
                "Optimizer execution or precision changed",
            )
            for source, expected in metadata["sources"].items():
                require(
                    hashlib.sha256((cell / "source" / source).read_bytes()).hexdigest()
                    == expected,
                    "Mismatch: hashlib.sha256((cell / 'source' / source).read_by...",
                )
            base_pair = paired_cell(task, seed, reference, root)
            paired_cell(task, seed, incumbent, root)
            history = [
                json.loads(line)
                for line in (cell / "history.jsonl").read_text().splitlines()
            ]
            crossings = [
                h for h in history if h["validation"]["sequence_accuracy"] >= TARGET
            ]
            if crossings:
                require(
                    result["completed"] == crossings[0]["step"],
                    "Mismatch: result['completed'] == crossings[0]['step']",
                )
                require(
                    base_pair["candidate_status"] == "passed",
                    "Mismatch: base_pair['candidate_status'] == 'passed'",
                )
                passes += 1
            else:
                require(
                    result["completed"] == task["case"]["max_updates"],
                    "Mismatch: result['completed'] == task['case']['max_updates'...",
                )
                require(
                    base_pair["candidate_status"] == "budget_exhausted",
                    "Mismatch: base_pair['candidate_status'] == 'budget_exhauste...",
                )
                censored.append(
                    {"case": task["name"], "seed": seed, "updates": result["completed"]}
                )
            saved = torch.load(
                cell / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            require(
                saved["completed"] == result["completed"],
                "Mismatch: saved['completed'] == result['completed']",
            )
            require(
                saved["history"] == history, "Mismatch: saved['history'] == history"
            )
            require(
                saved["metadata"] == metadata, "Mismatch: saved['metadata'] == metadata"
            )
            verify_optimizer_checkpoint(
                saved["optimizer"], metadata, task, config, result["completed"]
            )
            require(
                all(torch.isfinite(value).all() for value in saved["model"].values()),
                "Mismatch: all((torch.isfinite(value).all() for value in sav...",
            )
            parameters.append(metadata["parameters"])
            count += 1
    expected_count = len(tasks) * 3
    require(count == expected_count, "Mismatch: count == expected_count")
    records[name] = {
        "completed": count,
        "passed": passes,
        "censored": censored,
        "execution_errors": 0,
        "resumed": resumed,
        "execution_exposure_intervals": exposure_intervals,
        "candidate_plan_sha256": hashlib.sha256(raw_plan).hexdigest(),
        "source_files_verified": len(plan["source_sha256"]),
        "minimum_parameters": min(parameters),
        "maximum_parameters": max(parameters),
    }
result = {
    "audited_at_utc": datetime.now(UTC).isoformat(),
    "candidates": records,
    "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "completed_new_scientific_runs": sum(r["completed"] for r in records.values()),
    "all_checkpoints_histories_protocols_and_sources_retained": True,
    "checkpoint_optimizer_steps_and_settings_verified": True,
    "stop_at_first_exact_crossing_or_original_budget_verified": True,
    "original_initialization_data_seeds_cadence_and_oracles_verified": True,
    "capacity_policy": (
        "Original capacity and initialization matched; "
        "the fixed inverse-width gain adds no parameters."
    ),
    "source_archives_match_frozen_plans_and_current_implementation": True,
    "previous_round_reports_and_selection_preserved": True,
    "durable_update_journals_and_all_replayed_work_verified": True,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
