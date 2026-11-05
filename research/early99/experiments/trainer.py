"""
Candidate copy of the verified loop; original baseline remains frozen.

Only construction and experiment metadata differ. The sampler, observations,
checkpointing, first-crossing stopping and endpoint verification are reused.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
from lab.infrastructure.benchmarks.samplers import EpochSampler, gather

from research.current_grokking import data
from research.current_grokking.run import (
    BASE_OBSERVED_SPLITS,
    CHECKPOINT_INTERVAL,
    math_finite_measurements,
    outcome,
    save_checkpoint,
    source_snapshot,
    verify_model_reference,
    write,
)
from research.early99.experiments.factories import (
    Configuration,
    Factories,
    FactoryStepper,
)

if TYPE_CHECKING:
    from pathlib import Path

    from research.current_grokking.protocol import Case

BASELINE_CONFIGURATION = Configuration()
BASELINE_FACTORIES = Factories()


def run(
    case: Case,
    seed: int,
    output: Path,
    device: str = "cuda",
    *,
    resume: bool = False,
    compiled: bool = True,
    configuration: Configuration = BASELINE_CONFIGURATION,
    factories: Factories = BASELINE_FACTORIES,
) -> dict:
    """Reuse the frozen loop with explicit construction and stricter resume checks."""
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
    model = factories.model(task, case, device)
    initial = hashlib.sha256()
    for name, parameter in model.model.named_parameters():
        initial.update(name.encode())
        initial.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    metadata = {
        "case": asdict(case),
        "experiment": configuration.metadata(case),
        "experiment_source_plan": str(output / "plan.json"),
        "seed": seed,
        "batch_seed": seed + 10_000,
        "model_class": "diff_gpt.domain.model.gpt.GPT",
        "device": device,
        "torch": str(torch.__version__),
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": list(configuration.betas),
        "optimizer_eps": 1e-8,
        "optimizer_fused": device == "cuda",
        "compiled_loss": compiled and device == "cuda",
        "warmup_uses_completed_updates": True,
        "source_model_commit": reference["model_commit"],
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
    stepper = FactoryStepper(
        model,
        task,
        case,
        device,
        configuration=configuration,
        factories=factories,
        compiled=compiled,
    )
    sampler = EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=seed + 10_000
    )
    history, completed = [], 0
    if resume:
        saved = torch.load(checkpoint, weights_only=True, map_location=device)
        for key in (
            "case",
            "experiment",
            "device",
            "compiled_loss",
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
