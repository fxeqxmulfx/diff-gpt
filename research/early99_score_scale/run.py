"""Archive score-only extensions and reuse the unchanged benchmark trainer/scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from research.current_grokking import data
from research.current_grokking.protocol import ROOT
from research.current_grokking.run import verify_model_reference, write
from research.early99.experiments.trainer import run as train
from research.early99.protocol import MANIFEST, SEEDS, selected, specification, suite
from research.early99.run import DEFAULT_OUTPUT, report, summarize
from research.early99_score_scale.factories import Configuration, configured_factories


def prepare_output(
    output: Path, configuration: Configuration, device: str, *, compiled: bool
) -> dict:
    """Archive the candidate, baseline, task manifest and execution settings."""
    reference = verify_model_reference()
    if reference != suite()["baseline_model_reference"]:
        message = "The public-model reference differs from the frozen suite"
        raise ValueError(message)
    paths = [MANIFEST, Path(__file__).with_name("plan.json")]
    for directory in (
        MANIFEST.parent,
        ROOT / "research/current_grokking",
        Path(__file__).parent,
    ):
        paths.extend(sorted(directory.rglob("*.py")))
    sources = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    plan = {
        "suite": suite(),
        "configuration": asdict(configuration),
        "execution": {
            "device": device,
            "compiled_loss": compiled and device == "cuda",
            "cuda_graphs": device == "cuda",
        },
        "source_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            message = "Candidate settings/sources changed: use a new output directory"
            raise ValueError(message)
        for name, expected in plan["source_sha256"].items():
            if (
                hashlib.sha256(
                    (output / "experiment_source" / name).read_bytes()
                ).hexdigest()
                != expected
            ):
                message = f"Archived experiment source changed: {name}"
                raise ValueError(message)
        return plan
    for name, raw in sources.items():
        path = output / "experiment_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write(output / "configuration.json", asdict(configuration))
    write(output / "plan.json", plan)
    return plan


def run_one(
    row: dict,
    seed: int,
    output: Path,
    device: str,
    configuration: Configuration,
    *,
    compiled: bool,
) -> dict:
    """Check frozen data before using the candidate's explicit factories."""
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
            configuration=configuration,
            factories=configured_factories(configuration),
        )
    if result["metadata"]["experiment"] != configuration.metadata(case):
        message = "Executed candidate differs from its configuration"
        raise ValueError(message)
    history = [
        json.loads(line)
        for line in (directory / "history.jsonl").read_text().splitlines()
    ]
    record = summarize(row, seed, result, history)
    record.update(
        parameters=result["metadata"]["parameters"],
        experiment=result["metadata"]["experiment"],
    )
    write(directory / "early99_result.json", record)
    return record


def main() -> None:
    """Require an explicit action and a completed fresh short baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write-plan", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--report", action="store_true")
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tier", choices=("short", "long"), default="short")
    parser.add_argument("--case", action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    configuration = Configuration(**json.loads(args.configuration.read_text()))
    output = args.output or ROOT / "runs/early99/candidates" / configuration.name
    if output.resolve() == args.baseline.resolve():
        parser.error("Candidate output must be separate from the baseline")
    prepare_output(output, configuration, args.device, compiled=not args.no_compile)
    if args.write_plan:
        print(json.dumps({"plan": str(output / "plan.json")}))
        return
    if args.report:
        result = report(output, args.tier)
        print(
            json.dumps(
                {k: result[k] for k in ("planned_runs", "passed_runs", "complete")}
            )
        )
        return
    baseline = report(args.baseline, "short")
    if not baseline["complete"]:
        parser.error(
            "Complete all 75 fresh short baseline cells before candidate training"
        )
    rows = selected(args.tier)
    if args.case:
        unknown = set(args.case) - {r["name"] for r in rows}
        if unknown:
            parser.error(f"Cases absent from frozen tier: {sorted(unknown)}")
        rows = [row for row in rows if row["name"] in args.case]
    if args.seed is not None:
        if not args.case or len(args.case) != 1:
            parser.error("A single seed requires exactly one explicit case")
        directory = output / rows[0]["name"] / f"seed{args.seed}"
        try:
            record = run_one(
                rows[0],
                args.seed,
                output,
                args.device,
                configuration,
                compiled=not args.no_compile,
            )
            print(
                json.dumps(
                    {k: record[k] for k in ("case", "seed", "status", "first99_epochs")}
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
    errors = []
    for row in rows:
        for seed in SEEDS:
            command = [
                sys.executable,
                "-m",
                "research.early99_score_scale.run",
                "--run",
                "--configuration",
                str(output / "configuration.json"),
                "--output",
                str(output),
                "--baseline",
                str(args.baseline),
                "--tier",
                args.tier,
                "--case",
                row["name"],
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            if args.no_compile:
                command.append("--no-compile")
            with (output / f"{row['name']}-seed{seed}.log").open("a") as handle:
                completed = subprocess.run(  # noqa: S603 -- Fixed module; argument list.
                    command, stdout=handle, stderr=subprocess.STDOUT, check=False
                )
            if completed.returncode:
                errors.append(
                    {
                        "case": row["name"],
                        "seed": seed,
                        "returncode": completed.returncode,
                    }
                )
            report(output, args.tier)
    if errors:
        print(json.dumps({"execution_errors": errors}))
        sys.exit(1)


if __name__ == "__main__":
    main()
