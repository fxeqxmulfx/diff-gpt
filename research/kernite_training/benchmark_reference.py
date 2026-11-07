"""Separate compiled CUDA computation, exhaustive scoring and journal overhead."""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from research.cubic_capacity.factories import (
    case_for,
    configuration,
    factories,
    resolve,
)
from research.current_grokking import run as _bootstrap  # noqa: F401
from research.current_grokking.data import build
from research.early99.experiments.factories import FactoryStepper
from research.early99_execution.journal import UpdateJournal, recorded_step


def run(bundle: Path, directory: Path, *, journaled: bool) -> dict:
    """Time issued updates only after the existing capture-neutral preparation."""
    directory.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((bundle / "manifest.json").read_text())
    specification = resolve(manifest["name"])
    case = case_for(specification)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    started = time.perf_counter()
    task = build(case, "cuda")
    torch.manual_seed(manifest["seed"])
    model = factories(specification).model(task, case, "cuda")
    stepper = FactoryStepper(
        model,
        task,
        case,
        "cuda",
        configuration=configuration(specification),
        factories=factories(specification),
        compiled=True,
    )
    stepper.prepare([96, 256])
    preparation = time.perf_counter() - started
    order = np.fromfile(bundle / "order.u32", dtype="<u4").astype("int64")
    boundaries = np.fromfile(bundle / "boundaries.u32", dtype="<u4")
    batches = [
        torch.from_numpy(order[start:stop].copy())
        for start, stop in pairwise(boundaries)
    ]
    journal = UpdateJournal(directory / "execution.json", 4704, 256)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step, indices in enumerate(batches):
        if journaled:
            recorded_step(journal, indices, step, "cuda", stepper.step)
        else:
            stepper.step(indices, step)
    torch.cuda.synchronize()
    training = time.perf_counter() - started
    started = time.perf_counter()
    train = stepper.evaluate("train")
    validation = stepper.evaluate("validation")
    torch.cuda.synchronize()
    first_scoring = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(10):
        stepper.evaluate("train")
        stepper.evaluate("validation")
    torch.cuda.synchronize()
    scoring = (time.perf_counter() - started) / 10
    report = {
        "backend": "original torch.compile + CUDA graphs + fused native AdamW",
        "journal_per_update": journaled,
        "updates": len(batches),
        "preparation_seconds": preparation,
        "training_seconds": training,
        "milliseconds_per_update": 1000 * training / len(batches),
        "first_full_scoring_seconds": first_scoring,
        "steady_full_scoring_seconds": scoring,
        "train": train,
        "validation": validation,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "scope": ("Execution timing only; no scientific first99 claim"),
    }
    (directory / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run one timing arm without sharing the GPU with another benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--journaled", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.bundle, args.output, journaled=args.journaled)))


if __name__ == "__main__":
    main()
