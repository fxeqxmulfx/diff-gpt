"""Explicit baseline training and epoch reports for the frozen three-seed suite."""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

from research.current_grokking import data
from research.current_grokking.protocol import ROOT
from research.current_grokking.run import run as train
from research.current_grokking.run import write
from research.early99.protocol import (
    MANIFEST,
    SEEDS,
    exposure,
    first99,
    selected,
    specification,
    suite,
)

DEFAULT_OUTPUT = ROOT / "runs/early99/baseline"


def prepare_output(output: Path) -> None:
    """Freeze the suite and driver sources before any training starts."""
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        path.name: path.read_bytes() for path in sorted(MANIFEST.parent.glob("*.py"))
    }
    sources[MANIFEST.name] = MANIFEST.read_bytes()
    plan = {
        "suite": suite(),
        "driver_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()
        },
    }
    path = output / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            message = "Suite/driver changed: use a new output directory"
            raise ValueError(message)
        for name, expected in plan["driver_sha256"].items():
            archived = (output / "suite_source" / name).read_bytes()
            if hashlib.sha256(archived).hexdigest() != expected:
                message = f"Archived benchmark source changed: {name}"
                raise ValueError(message)
        return
    directory = output / "suite_source"
    directory.mkdir(exist_ok=True)
    for name, raw in sources.items():
        (directory / name).write_bytes(raw)
    write(path, plan)


def summarize(row: dict, seed: int, result: dict, history: list[dict]) -> dict:
    """Report first-99 epochs without requiring delay or final retention."""
    case = specification(row)
    metadata = result["metadata"]
    if metadata["case"] != row["case"]:
        message = "Executed task settings differ from the frozen benchmark"
        raise ValueError(message)
    if metadata["data_fingerprints"] != row["data_fingerprints"]:
        message = "Executed split fingerprints differ from the frozen benchmark"
        raise ValueError(message)
    measured = result["endpoint"]["train"]
    train_examples = measured.get("examples", measured.get("example_count"))
    if train_examples != row["train_examples"] or metadata["seed"] != seed:
        message = "Training size or seed differs from the declared benchmark cell"
        raise ValueError(message)
    crossing = first99(history, train_examples, case.batch)
    if result["first_target"] != (crossing["updates"] if crossing else None):
        message = "Trainer stopping criterion disagrees with historical first 99%"
        raise ValueError(message)
    return {
        "case": case.name,
        "tier": row["tier"],
        "seed": seed,
        "status": "passed" if crossing else "budget_exhausted",
        "first99": crossing,
        "first99_updates": crossing["updates"] if crossing else None,
        "first99_epochs": crossing["epochs"] if crossing else None,
        "completed": exposure(result["completed"], train_examples, case.batch),
        "final_heldout_accuracy": result["endpoint"]["validation"]["sequence_accuracy"],
        "resumed": bool(metadata.get("resumptions")),
        "seconds_last_run_segment": result["seconds"],
        "data_fingerprints": metadata["data_fingerprints"],
        "initial_parameters_sha256": metadata["initial_parameters_sha256"],
        "model_source_reference": metadata.get("model_source_reference"),
    }


def run_one(row: dict, seed: int, output: Path, device: str, *, compiled: bool) -> dict:
    """Train one fresh baseline or resume its own preserved checkpoint."""
    case = specification(row)
    directory = output / case.name / f"seed{seed}"
    path = directory / "result.json"
    if path.exists():
        result = json.loads(path.read_text())
    else:
        task = data.build(case, "cpu")
        if (
            task.fingerprints != row["data_fingerprints"]
            or len(task.splits["train"]) != row["train_examples"]
            or task.selection != row["selection"]
        ):
            message = "Preflight data differ from the frozen benchmark"
            raise ValueError(message)
        del task
        result = train(
            case,
            seed,
            output,
            device,
            resume=(directory / "checkpoint.pt").exists(),
            compiled=compiled,
        )
    history = [
        json.loads(line)
        for line in (directory / "history.jsonl").read_text().splitlines()
    ]
    record = summarize(row, seed, result, history)
    write(directory / "early99_result.json", record)
    return record


def report(output: Path, tier: str) -> dict:
    """Keep all three seeds and bounded misses visible in task/epoch summaries."""
    rows, tasks = [], []
    for task in selected(tier):
        cells = []
        for seed in SEEDS:
            directory = output / task["name"] / f"seed{seed}"
            path = directory / "early99_result.json"
            error_path = directory / "error.json"
            cell = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "case": task["name"],
                    "tier": task["tier"],
                    "seed": seed,
                    "status": json.loads(error_path.read_text())["kind"]
                    if error_path.exists()
                    else "pending",
                }
            )
            rows.append(cell)
            cells.append(cell)
        solved = [c["first99_epochs"] for c in cells if c["status"] == "passed"]
        complete = len(solved) == len(SEEDS)
        tasks.append(
            {
                "case": task["name"],
                "tier": task["tier"],
                "passed_seeds": len(solved),
                "all_three_passed": complete,
                "median_first99_epochs": statistics.median(solved)
                if complete
                else None,
                "worst_first99_epochs": max(solved) if complete else None,
            }
        )
    passed = [r["first99_epochs"] for r in rows if r["status"] == "passed"]
    all_passed = len(passed) == len(rows)
    mean_epochs = None
    if all_passed:
        mean_epochs = (
            math.exp(sum(math.log(value) for value in passed) / len(passed))
            if all(value > 0 for value in passed)
            else 0.0
        )
    result = {
        "tier": tier,
        "seeds": list(SEEDS),
        "planned_runs": len(rows),
        "passed_runs": len(passed),
        "complete": all(r["status"] in {"passed", "budget_exhausted"} for r in rows),
        "all_passed": all_passed,
        "geometric_mean_first99_epochs": mean_epochs,
        "interpretation": (
            "A single observed 99% exact held-out crossing suffices. Epochs count "
            "actual presented examples, including short tails. Per-task median/worst "
            "and aggregate geometric mean require all expected seeds to pass. "
            "Unsolved cells are budget-censored and are never discarded."
        ),
        "tasks": tasks,
        "runs": rows,
    }
    write(output / f"summary-{tier}.json", result)
    fields = (
        "case",
        "tier",
        "seed",
        "status",
        "first99_updates",
        "first99_epochs",
        "final_heldout_accuracy",
        "resumed",
        "seconds_last_run_segment",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    (output / f"summary-{tier}.csv").write_text(buffer.getvalue())
    return result


def main() -> None:
    """Require an explicit action; plan/report never launch training."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-plan", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--tier", choices=("short", "long", "all"), default="short")
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    prepare_output(args.output)
    if args.write_plan:
        print(json.dumps({"plan": str(args.output / "plan.json"), "seeds": SEEDS}))
        return
    if args.report:
        result = report(args.output, args.tier)
        print(
            json.dumps(
                {k: result[k] for k in ("planned_runs", "passed_runs", "complete")}
            )
        )
        return
    candidates = selected(args.tier)
    if args.case:
        candidates = [row for row in candidates if row["name"] == args.case]
        if not candidates:
            parser.error("Case is absent from the selected frozen tier")
    if args.seed is not None:
        if not args.case:
            parser.error("A single seed is only valid with an explicit case")
        directory = args.output / args.case / f"seed{args.seed}"
        try:
            result = run_one(
                candidates[0],
                args.seed,
                args.output,
                args.device,
                compiled=not args.no_compile,
            )
            print(
                json.dumps(
                    {k: result[k] for k in ("case", "seed", "status", "first99_epochs")}
                )
            )
        except Exception as error:
            directory.mkdir(parents=True, exist_ok=True)
            write(
                directory / "error.json",
                {
                    "kind": "numerical_training_failure"
                    if isinstance(error, FloatingPointError)
                    else "execution_error",
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        return
    for row in candidates:
        for seed in SEEDS:
            command = [
                sys.executable,
                "-m",
                "research.early99.run",
                "--run",
                "--tier",
                args.tier,
                "--case",
                row["name"],
                "--seed",
                str(seed),
                "--output",
                str(args.output),
                "--device",
                args.device,
            ]
            if args.no_compile:
                command.append("--no-compile")
            path = args.output / f"{row['name']}-seed{seed}.log"
            with path.open("a") as handle:
                subprocess.run(  # noqa: S603 -- Frozen task names; argument list, no shell.
                    command, stdout=handle, stderr=subprocess.STDOUT, check=True
                )
    report(args.output, args.tier)


if __name__ == "__main__":
    main()
