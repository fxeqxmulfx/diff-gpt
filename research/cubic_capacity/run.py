"""Run the entire declared capacity grid in isolated native-training processes."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from research.cubic_capacity.factories import (
    ROOT,
    RUNS,
    STUDY,
    case_for,
    configuration,
    plan,
    resolve,
    task_row,
)
from research.cubic_capacity.readiness import (
    sha256,
    source_hashes,
    verify_training_ready,
)
from research.cubic_capacity.trainer import run as train
from research.current_grokking.data import build
from research.current_grokking.run import write
from research.early99.run import summarize
from research.early99_reporting.round2_audit import require, verify_hash


def prepare_output(specification: dict) -> Path:
    """Archive every source and reject changes to an already prepared experiment."""
    output = RUNS / specification["name"]
    sources = source_hashes()
    archived_plan = {
        "study_plan_sha256": sha256(STUDY / "plan.json"),
        "specification": specification,
        "configuration": configuration(specification).__dict__,
        "task": task_row(specification),
        "source_sha256": sources,
        "execution": {
            "device": "cuda",
            "compiled_loss": True,
            "cuda_graphs": True,
            "durable_per_update_journal": True,
        },
    }
    destination = output / "plan.json"
    if destination.exists():
        require(
            json.loads(destination.read_text()) == archived_plan,
            "Capacity settings or sources changed",
        )
        for path, expected in sources.items():
            verify_hash(output / "experiment_source" / path, expected)
        return output
    output.mkdir(parents=True, exist_ok=True)
    for relative in sources:
        path = output / "experiment_source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    write(destination, archived_plan)
    return output


def run_cell(name: str, seed: int) -> None:
    """Train or continue exactly one declared seed, retaining every failure artifact."""
    require(seed in plan()["seeds"], "Undeclared model seed")
    require(torch.cuda.is_available(), "Scientific capacity training requires CUDA")
    verify_training_ready()
    specification = resolve(name)
    output = prepare_output(specification)
    case = case_for(specification)
    directory = output / case.name / f"seed{seed}"
    lock_path = output / f"seed{seed}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(
            not (directory / "error.json").exists(),
            "Preserved execution error requires an explicit recovery audit",
        )
        result_path = directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            task = build(case, "cpu")
            row = task_row(specification)
            require(
                task.fingerprints == row["data_fingerprints"]
                and len(task.splits["train"]) == row["train_examples"]
                and task.selection == row["selection"],
                "Capacity preflight changed data or oracle",
            )
            result = train(
                case,
                seed,
                output,
                "cuda",
                specification=specification,
                resume=(directory / "checkpoint.pt").exists(),
            )
        require(
            result["metadata"]["experiment"]
            == configuration(specification).metadata(case),
            "Executed settings differ from capacity plan",
        )
        history = [
            json.loads(line)
            for line in (directory / "history.jsonl").read_text().splitlines()
        ]
        record = summarize(task_row(specification), seed, result, history)
        record.update(
            parameters=result["metadata"]["parameters"],
            experiment=result["metadata"]["experiment"],
            capacity_specification=specification,
            separate_capacity_study=True,
        )
        write(directory / "early99_result.json", record)
        print(
            json.dumps(
                {
                    "name": name,
                    "seed": seed,
                    "status": record["status"],
                    "first99_epochs": record["first99_epochs"],
                }
            ),
            flush=True,
        )


def main() -> None:
    """Own one dispatcher lock and wait for every declared child process to finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--cell")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "Run from repository root")
    if args.cell:
        try:
            run_cell(args.cell, args.seed)
        except Exception:
            specification = resolve(args.cell)
            error = (
                RUNS
                / specification["name"]
                / plan()["task"]["name"]
                / f"seed{args.seed}"
                / "error.json"
            )
            error.parent.mkdir(parents=True, exist_ok=True)
            if not error.exists():
                write(
                    error,
                    {
                        "name": args.cell,
                        "seed": args.seed,
                        "traceback": traceback.format_exc(),
                    },
                )
            raise
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_training_ready()
        outcomes = []
        for specification in plan()["configurations"]:
            output = prepare_output(specification)
            for seed in plan()["seeds"]:
                directory = output / plan()["task"]["name"] / f"seed{seed}"
                if (directory / "early99_result.json").exists():
                    outcomes.append(
                        {
                            "name": specification["name"],
                            "seed": seed,
                            "reused_completed": True,
                        }
                    )
                    continue
                write(
                    RUNS / "progress.json",
                    {
                        "phase": "training",
                        "name": specification["name"],
                        "seed": seed,
                        "finished_children": len(outcomes),
                    },
                )
                command = [
                    sys.executable,
                    "-m",
                    "research.cubic_capacity.run",
                    "--cell",
                    specification["name"],
                    "--seed",
                    str(seed),
                ]
                with (output / f"seed{seed}.log").open("a") as stream:
                    completed = subprocess.run(  # noqa: S603 -- Declared local runner.
                        command, check=False, stdout=stream, stderr=subprocess.STDOUT
                    )
                outcomes.append(
                    {
                        "name": specification["name"],
                        "seed": seed,
                        "exit_code": completed.returncode,
                    }
                )
                print(json.dumps(outcomes[-1]), flush=True)
        write(
            RUNS / "progress.json",
            {"phase": "training_finished_requires_audit", "children": outcomes},
        )


if __name__ == "__main__":
    main()
