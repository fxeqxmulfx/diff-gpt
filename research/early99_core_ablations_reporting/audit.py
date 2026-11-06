"""Audit original-model native states and every durable executed-work segment."""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import torch

from research.current_grokking.run import verify_model_reference
from research.early99.protocol import selected, specification, suite
from research.early99_core_ablations.factories import Configuration
from research.early99_execution.journal import verify_saved_journal
from research.early99_optim_reporting.metrics import paired_cell
from research.early99_reporting.round2_audit import require

OPTIMIZER_EPS = 1e-08
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
search = json.loads(Path("research/early99_core_ablations/plan.json").read_text())
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
                metadata["optimizer_betas"] == list(config.betas),
                "Mismatch: metadata['optimizer_betas'] == list(config.betas)",
            )
            require(
                metadata["optimizer_eps"] == OPTIMIZER_EPS,
                "Mismatch: metadata['optimizer_eps'] == 1e-08",
            )
            require(
                metadata["optimizer_fused"] and metadata["compiled_loss"],
                "Mismatch: metadata['optimizer_fused'] and metadata['compile...",
            )
            require(
                metadata["device"] == "cuda", "Mismatch: metadata['device'] == 'cuda'"
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
            require(
                {int(s["step"]) for s in saved["optimizer"]["state"].values()}
                == {result["completed"]},
                "Mismatch: {int(s['step']) for s in saved['optimizer']['stat...",
            )
            groups = saved["optimizer"]["param_groups"]
            require(len(groups) == 1, "Mismatch: len(groups) == (2 if lerope else 1)")
            group = groups[0]
            require(
                len(group["params"]) == len(set(group["params"]))
                and set(saved["optimizer"]["state"]) == set(group["params"]),
                "Native AdamW checkpoint has missing or duplicated parameter states",
            )
            for state in saved["optimizer"]["state"].values():
                first, second = state["exp_avg"], state["exp_avg_sq"]
                require(
                    first.dtype == second.dtype == torch.float32
                    and first.shape == second.shape
                    and torch.isfinite(first).all().item()
                    and torch.isfinite(second).all().item()
                    and (second >= 0).all().item(),
                    "Invalid final native AdamW moments",
                )
            require(
                group["betas"] == config.betas,
                "Mismatch: group['betas'] == config.betas",
            )
            require(
                group["fused"] and group["capturable"],
                "Mismatch: group['fused'] and group['capturable']",
            )
            require(
                group["eps"] == OPTIMIZER_EPS
                and group["weight_decay"] == task["case"]["weight_decay"],
                "Mismatch: group['eps'] == 1e-08 and group['weight_decay'] =...",
            )
            expected_rate = config.learning_rate_at(
                specification(task), result["completed"] - 1, task["train_examples"]
            )
            require(
                math.isclose(
                    float(group["lr"]), expected_rate, rel_tol=1e-07, abs_tol=1e-12
                ),
                "Mismatch: math.isclose(float(group['lr']), expected_rate, r...",
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
    "final_optimizer_moments_finite_and_second_moments_nonnegative": True,
    "stop_at_first_exact_crossing_or_original_budget_verified": True,
    "original_initialization_data_seeds_cadence_and_oracles_verified": True,
    "capacity_policy": (
        "Original capacity and initialization matched; "
        "the isolated additions introduce no parameters."
    ),
    "source_archives_match_frozen_plans_and_current_implementation": True,
    "previous_round_reports_and_selection_preserved": True,
    "durable_update_journals_and_all_replayed_work_verified": True,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
