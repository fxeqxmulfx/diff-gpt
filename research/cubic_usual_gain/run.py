"""Run all three declared usual-gain seeds only after completing the capacity grid."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from research.cubic_usual_gain.factories import (
    ROOT,
    RUNS,
    STUDY,
    case_for,
    configuration,
    plan,
    train,
)
from research.cubic_usual_gain.readiness import (
    sha256,
    source_hashes,
    verify_training_ready,
)
from research.current_grokking.run import write
from research.early99.run import summarize
from research.early99_reporting.round2_audit import read, require, verify_hash


def prepare_output() -> Path:
    """Archive exact settings and sources without changing any prior control."""
    cfg = configuration()
    output = RUNS / cfg.name
    sources = source_hashes()
    archived = {
        "study_plan_sha256": sha256(STUDY / "plan.json"),
        "configuration": cfg.__dict__,
        "task": plan()["task"],
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
        require(read(destination) == archived, "Changed started experiment")
        for path, digest in sources.items():
            verify_hash(output / "experiment_source" / path, digest)
        return output
    output.mkdir(parents=True, exist_ok=True)
    for relative in sources:
        target = output / "experiment_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    write(destination, archived)
    return output


def run_seed(seed: int) -> None:
    """Retain failures, recover only explicit checkpoints and count every update."""
    require(seed in plan()["seeds"], "Undeclared seed")
    require(torch.cuda.is_available(), "Scientific training requires CUDA")
    verify_training_ready()
    root = prepare_output()
    case = case_for()
    directory = root / case.name / f"seed{seed}"
    with (root / f"seed{seed}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(
            not (directory / "error.json").exists(),
            "Retained error requires explicit recovery audit",
        )
        result_path = directory / "result.json"
        result = (
            read(result_path)
            if result_path.exists()
            else train(
                case, seed, root, "cuda", resume=(directory / "checkpoint.pt").exists()
            )
        )
        require(
            result["metadata"]["experiment"] == configuration().metadata(case),
            "Changed executed settings",
        )
        history = [
            json.loads(line)
            for line in (directory / "history.jsonl").read_text().splitlines()
        ]
        record = summarize(plan()["task"], seed, result, history)
        record.update(
            parameters=result["metadata"]["parameters"],
            experiment=result["metadata"]["experiment"],
            separate_gain_study=True,
        )
        write(directory / "early99_result.json", record)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "status": record["status"],
                    "first99_epochs": record["first99_epochs"],
                }
            ),
            flush=True,
        )


def main() -> None:
    """Run fresh processes for all three seeds; never silently omit a failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--seed", type=int)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "Run from repository root")
    verify_training_ready()
    if args.seed is not None:
        try:
            run_seed(args.seed)
        except Exception:
            directory = (
                RUNS / configuration().name / case_for().name / f"seed{args.seed}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory / "error.json").exists():
                write(
                    directory / "error.json",
                    {"seed": args.seed, "traceback": traceback.format_exc()},
                )
            raise
        return
    root = prepare_output()
    with (RUNS / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        outcomes = []
        for seed in plan()["seeds"]:
            directory = root / case_for().name / f"seed{seed}"
            if (directory / "early99_result.json").exists():
                outcomes.append({"seed": seed, "reused_completed": True})
                continue
            write(
                RUNS / "progress.json",
                {"phase": "training", "seed": seed, "finished_children": len(outcomes)},
            )
            with (root / f"seed{seed}.log").open("a") as stream:
                child = subprocess.run(  # noqa: S603 -- Fixed declared local runner.
                    [
                        sys.executable,
                        "-m",
                        "research.cubic_usual_gain.run",
                        "--seed",
                        str(seed),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            outcomes.append({"seed": seed, "exit_code": child.returncode})
            print(json.dumps(outcomes[-1]), flush=True)
        write(
            RUNS / "progress.json",
            {"phase": "training_finished_requires_audit", "children": outcomes},
        )


if __name__ == "__main__":
    main()
