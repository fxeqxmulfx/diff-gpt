"""Run a declared task on the unchanged current GPT with native AdamW."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from lab.infrastructure.benchmarks.samplers import EpochSampler, gather

from diff_gpt.domain.model.gpt import GPT
from research.current_grokking import data
from research.current_grokking.protocol import (
    ARITHMETIC,
    LEAN,
    ROOT,
    cases,
    records,
)
from research.current_grokking.train import Logits, Stepper

FIT_TARGET = 0.99
MIN_PLATEAU_OBSERVATIONS = 2
BASE_OBSERVED_SPLITS = 2
CHECKPOINT_INTERVAL = 1000


def math_finite_measurements(row: dict) -> bool:
    """Require finite proper losses without treating absent ceilings as errors."""
    return all(
        row[name]["loss"] is not None and math.isfinite(row[name]["loss"])
        for name in ("train", "validation")
    )


def verify_model_reference() -> dict:
    """Reject a cohort that silently changes any frozen model source file."""
    reference = json.loads(Path(__file__).with_name("model_reference.json").read_text())
    for relative, expected in reference["files"].items():
        actual = hashlib.sha256(
            (ROOT / "src/diff_gpt" / relative).read_bytes()
        ).hexdigest()
        if actual != expected:
            message = f"Frozen current-model source changed: {relative}"
            raise RuntimeError(message)
    return reference


if TYPE_CHECKING:
    from research.current_grokking.protocol import (
        Case,
    )


def write(path: Path, value: object) -> None:
    """Atomically replace machine-readable progress."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_snapshot(directory: Path) -> dict:
    """Preserve executed model/runner/data code, including external oracles."""
    sources = {}
    groups = [
        ("gpt", ROOT / "src/diff_gpt"),
        ("runner", ROOT / "research/current_grokking"),
        ("lean_lab", LEAN / "python/src/lab"),
    ]
    for prefix, base in groups:
        for source in base.rglob("*.py"):
            relative = Path(prefix) / source.relative_to(base)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = source.read_bytes()
            target.write_bytes(raw)
            sources[str(relative)] = hashlib.sha256(raw).hexdigest()
    source = ARITHMETIC / "src/training/domain/tasks.py"
    target = directory / "source/arithmetic_tasks.py"
    target.write_bytes(source.read_bytes())
    sources["arithmetic_tasks.py"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return sources


def metric(case: Case, values: dict) -> float | None:
    """Use each task's declared metric, retaining raw losses and accuracies."""
    if case.metric.startswith("negative_"):
        return -values[case.metric.removeprefix("negative_")]
    return values.get(case.metric)


def outcome(history: list[dict], case: Case, baseline: float = 0.0) -> dict:
    """Report operational delay and bounded misses without eventual-failure claims."""
    first_fit = first_train99 = first_target = None
    streak = longest = 0
    for row in history:
        train = row["train"].get("sequence_accuracy", row["train"].get("accuracy", 0.0))
        value = metric(case, row["validation"])
        if first_train99 is None and train >= FIT_TARGET:
            first_train99 = row["step"]
        fitted = (
            metric(case, row["train"]) >= case.target
            if case.metric == "negative_mse"
            else train >= FIT_TARGET
        )
        if first_fit is None and fitted:
            first_fit = row["step"]
        if first_target is None and value is not None and value >= case.target:
            first_target = row["step"]
        optimum = 0.0 if case.metric.startswith("negative_") else 1.0
        low = value is not None and value <= baseline + 0.1 * (optimum - baseline)
        if first_target is None and fitted and low:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    classification = (
        (
            "delayed_generalization"
            if longest >= MIN_PLATEAU_OBSERVATIONS
            else "generalized_without_low_plateau"
        )
        if first_target is not None
        else "target_not_reached_within_budget"
    )
    return {
        "first_train99": first_train99,
        "first_fit": first_fit,
        "first_target": first_target,
        "low_plateau_observations": longest,
        "classification": classification,
        "final_target_met": metric(case, history[-1]["validation"]) is not None
        and metric(case, history[-1]["validation"]) >= case.target,
    }


def save_checkpoint(
    path: Path,
    model: Logits,
    stepper: Stepper,
    sampler: EpochSampler,
    completed: int,
    history: list[dict],
    metadata: dict,
) -> None:
    """Save final weights, AdamW moments, sampler and both RNG states."""
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.model.state_dict(),
            "optimizer": stepper.optimizer.state_dict(),
            "sampler": sampler.state(),
            "completed": completed,
            "history": history,
            "metadata": metadata,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state()
            if torch.cuda.is_available()
            else None,
        },
        temporary,
    )
    temporary.replace(path)


def run(
    case: Case,
    seed: int,
    output: Path,
    device: str = "cuda",
    *,
    resume: bool = False,
    compiled: bool = True,
) -> dict:
    """Train one case through its first target plus a fixed follow-up, or budget."""
    directory = output / case.name / f"seed{seed}"
    checkpoint = directory / "checkpoint.pt"
    if directory.exists() and not resume:
        msg = f"Refusing to overwrite {directory}"
        raise FileExistsError(msg)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    reference = verify_model_reference()
    started = time.monotonic()
    task = data.build(case, device)
    interval = case.observe or max(
        1, (len(task.splits["train"]) + case.batch - 1) // case.batch
    )
    torch.manual_seed(seed)
    model = Logits(
        GPT(
            task.vocab,
            case.width,
            block_size=task.context,
            n_head=case.heads,
            n_layer=case.layers,
            use_checkpoint=False,
            pad_vocab_size_to=1,
        ).to(device)
    )
    initial = hashlib.sha256()
    for name, parameter in model.model.named_parameters():
        initial.update(name.encode())
        initial.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    metadata = {
        "case": asdict(case),
        "seed": seed,
        "batch_seed": seed + 10_000,
        "model_class": "diff_gpt.domain.model.gpt.GPT",
        "device": device,
        "torch": str(torch.__version__),
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": [0.9, 0.98],
        "optimizer_eps": 1e-8,
        "optimizer_fused": device == "cuda",
        "compiled_loss": compiled and device == "cuda",
        "warmup_uses_completed_updates": True,
        "source_model_commit": "01f18f86787511279fe52fe41883495fce0e3105",
        "initial_parameters_sha256": initial.hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_fingerprints": task.fingerprints,
        "data_metadata": task.metadata,
        "selection_baseline": task.baseline,
        "model_source_reference": reference,
        "observe_updates": interval,
        "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
        "grokking_definition": (
            "At least two observed fitted-train/near-baseline heldout "
            "states before reaching the declared target; also retain raw "
            "timing and fast generalization."
        ),
    }
    stepper = Stepper(model, task, case, device, compiled=compiled)
    sampler = EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=seed + 10_000
    )
    history, completed = [], 0
    if resume:
        saved = torch.load(checkpoint, weights_only=True, map_location=device)
        for key in (
            "case",
            "seed",
            "initial_parameters_sha256",
            "data_fingerprints",
            "model_class",
        ):
            if saved["metadata"][key] != metadata[key]:
                msg = f"Resume changes {key}"
                raise ValueError(msg)
        model.model.load_state_dict(saved["model"])
        stepper.optimizer.load_state_dict(saved["optimizer"])
        stepper.optimizer.param_groups[0]["lr"] = stepper.rate
        sampler.restore(saved["sampler"])
        completed, history = saved["completed"], saved["history"]
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        metadata = saved["metadata"]
        resumed_sources = directory / f"resume-{time.time_ns()}"
        metadata.setdefault("resumptions", []).append(
            {
                "completed": completed,
                "directory": resumed_sources.name,
                "sources": source_snapshot(resumed_sources),
                "device": device,
                "compiled_loss": compiled and device == "cuda",
            }
        )
        write(directory / "protocol.json", metadata)
        if (directory / "history.jsonl").exists():
            shutil.copy2(
                directory / "history.jsonl",
                directory / f"history-before-resume-{time.time_ns()}.jsonl",
            )
        (directory / "history.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in history)
        )
    else:
        metadata["sources"] = source_snapshot(directory)
        write(directory / "protocol.json", metadata)
    stepper.prepare(sampler.sizes())
    checkpoint_completed = completed if resume else -CHECKPOINT_INTERVAL
    print(
        json.dumps(
            {
                "case": case.name,
                "seed": seed,
                "event": "prepared",
                "completed": completed,
                "seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    history_path = directory / "history.jsonl"
    while True:
        if not history or completed > history[-1]["step"]:
            observed = {name: stepper.evaluate(name) for name in task.observed}
            row = {
                "step": completed,
                "seconds": time.monotonic() - started,
                "train": observed["train"],
                "validation": observed[task.selection],
            }
            if not math_finite_measurements(row):
                message = f"Nonfinite training/validation loss at update {completed}"
                raise FloatingPointError(message)
            if len(observed) > BASE_OBSERVED_SPLITS:
                row["additional_observations"] = observed
            history.append(row)
            with history_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            status = outcome(history, case, getattr(task, "baseline", 0.0))
            write(
                directory / "status.json",
                {"case": case.name, "seed": seed, **row, **status},
            )
            print(
                json.dumps({"case": case.name, "seed": seed, **row, **status}),
                flush=True,
            )
            if completed - checkpoint_completed >= CHECKPOINT_INTERVAL:
                save_checkpoint(
                    checkpoint, model, stepper, sampler, completed, history, metadata
                )
                checkpoint_completed = completed
        status = outcome(history, case, getattr(task, "baseline", 0.0))
        if completed >= case.max_updates or (
            status["first_target"] is not None
            and completed >= status["first_target"] + case.after_target
        ):
            break
        stop = min(case.max_updates, completed + interval)
        model.train()
        for _ in range(completed, stop):
            parts, _ = sampler.next()
            stepper.step(gather(parts), completed)
            completed += 1
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        msg = "Nonfinite final public-GPT parameters"
        raise FloatingPointError(msg)
    steps = [int(state["step"]) for state in stepper.optimizer.state.values()]
    if not steps or any(step != completed for step in steps):
        msg = f"Optimizer steps disagree with completed updates: {steps}, {completed}"
        raise AssertionError(msg)
    save_checkpoint(checkpoint, model, stepper, sampler, completed, history, metadata)
    final = {
        name: stepper.evaluate(name)
        for name in task.splits
        if name.startswith("test") and task.splits[name] is not None
    }
    result = {
        "case": case.name,
        "seed": seed,
        "status": "passed"
        if status["first_target"] is not None
        else "budget_exhausted",
        "completed": completed,
        "seconds": time.monotonic() - started,
        **status,
        "endpoint": history[-1],
        "final_test": final,
        "optimizer_steps_verified": True,
        "parameters_finite": True,
        "metadata": metadata,
    }
    if hasattr(task, "final_metrics"):
        result["final_extra"] = task.final_metrics(model)
    write(directory / "result.json", result)
    return result


def main() -> None:
    """Run an explicit case, or dispatch the frozen matrix in fresh processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--kind", help="Dispatch one declared task family")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/current-grokking")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.write_plan:
        write(
            args.output / "plan.json",
            {
                "model": "current public GPT",
                "optimizer": "ordinary AdamW",
                "cases": records(),
                "seeds": args.seeds,
            },
        )
        print(
            json.dumps({"cases": len(cases()), "path": str(args.output / "plan.json")})
        )
        return
    if args.all:
        for case in cases():
            if args.kind is not None and case.kind != args.kind:
                continue
            for seed in args.seeds:
                directory = args.output / case.name / f"seed{seed}"
                if (directory / "result.json").exists():
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "research.current_grokking.run",
                    "--case",
                    case.name,
                    "--seeds",
                    str(seed),
                    "--output",
                    str(args.output),
                    "--device",
                    args.device,
                ]
                if (directory / "checkpoint.pt").exists():
                    command.append("--resume")
                if args.no_compile:
                    command.append("--no-compile")
                with (args.output / f"{case.name}-seed{seed}.log").open("a") as handle:
                    process = subprocess.run(  # noqa: S603 -- Explicit argument list, no shell, matrix-validated names.
                        command, stdout=handle, stderr=subprocess.STDOUT, check=False
                    )
                print(
                    json.dumps(
                        {
                            "case": case.name,
                            "seed": seed,
                            "returncode": process.returncode,
                        }
                    ),
                    flush=True,
                )
        return
    selected = next((case for case in cases() if case.name == args.case), None)
    if selected is None:
        parser.error("select --case NAME, --all, or --write-plan")
    for seed in args.seeds:
        try:
            result = run(
                selected,
                seed,
                args.output,
                args.device,
                resume=args.resume,
                compiled=not args.no_compile,
            )
            print(
                json.dumps(
                    {
                        "result": {
                            key: result[key]
                            for key in (
                                "case",
                                "seed",
                                "status",
                                "completed",
                                "first_target",
                                "classification",
                            )
                        }
                    }
                ),
                flush=True,
            )
        except Exception as error:
            directory = args.output / selected.name / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            write(
                directory / f"error-{time.time_ns()}.json",
                {
                    "kind": "numerical_training_failure"
                    if isinstance(error, FloatingPointError)
                    else "execution_error_not_learning_failure",
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
