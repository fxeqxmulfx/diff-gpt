"""Export inputs once; the native training executable does not import Python."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from research.cubic_capacity.factories import (
    case_for,
    configuration,
    factories,
    resolve,
    task_row,
)
from research.current_grokking import run as _bootstrap  # noqa: F401
from research.current_grokking.data import build
from research.current_grokking.run import verify_model_reference
from research.early99.experiments.initialization import parameter_metadata

# The existing runner installs the frozen Lean project's data/sampler import path.
from lab.infrastructure.benchmarks.samplers import EpochSampler, gather  # isort: skip

CONTEXT, VOCAB, HEAD_WIDTH, IGNORE = 2, 97, 32, -100


def export(directory: Path, name: str, seed: int, updates: int) -> dict:
    """Keep original initialization and every sampled training row, without updates."""
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    spec = resolve(name)
    case = case_for(spec)
    if seed not in (0, 1, 2) or not 1 <= updates <= case.max_updates:
        message = "Use a declared seed and a positive budget within the frozen task"
        raise ValueError(message)
    reference = verify_model_reference()
    config = configuration(spec)
    effective = config.training_case(case)
    task = build(case, "cpu")
    if task.fingerprints != task_row(spec)["data_fingerprints"]:
        message = "Export changed the frozen data or oracle"
        raise ValueError(message)
    torch.manual_seed(seed)
    model = factories(spec).model(task, case, "cpu")
    if (
        task.context != CONTEXT
        or task.vocab != VOCAB
        or case.width // case.heads != HEAD_WIDTH
    ):
        message = "The native backend currently supports two-token arithmetic"
        raise ValueError(message)
    files = {}

    def save(name: str, value: torch.Tensor | np.ndarray) -> None:
        array = value.detach().numpy() if isinstance(value, torch.Tensor) else value
        payload = array.tobytes(order="C")
        (directory / name).write_bytes(payload)
        files[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    parameters = []
    offset = 0
    for param_name, value in model.model.named_parameters():
        parameters.append(
            {
                "name": param_name,
                "shape": list(value.shape),
                "offset": offset,
                "count": value.numel(),
            }
        )
        offset += value.numel()
    save(
        "parameters.f32",
        torch.cat([p.detach().flatten() for p in model.parameters()])
        .numpy()
        .astype("<f4"),
    )
    save(
        "rope.f32",
        torch.stack((model.model.rope_cos, model.model.rope_sin)).numpy().astype("<f4"),
    )
    for split in ("train", "validation"):
        rows = task.splits[split].rows
        if not bool((rows[:, 1, 0] == IGNORE).all()):
            message = "Unexpected first-token targets"
            raise ValueError(message)
        save(
            f"{split}.u32",
            torch.cat((rows[:, 0], rows[:, 1, 1:]), dim=1).numpy().astype("<u4"),
        )
    sampler = EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=seed + 10_000
    )
    schedule = []
    boundaries = [0]
    for _ in range(updates):
        parts, _ = sampler.next()
        index = gather(parts).numpy().astype("<u4")
        schedule.append(index)
        boundaries.append(boundaries[-1] + len(index))
    save(
        "order.u32", np.concatenate(schedule) if schedule else np.empty(0, dtype="<u4")
    )
    save("boundaries.u32", np.asarray(boundaries, dtype="<u4"))
    manifest = {
        "version": 1,
        "name": name,
        "seed": seed,
        "width": case.width,
        "layers": case.layers,
        "heads": case.heads,
        "vocab": task.vocab,
        "normalized": spec["architecture"] == "inverse_width",
        "ssmax": spec["architecture"] == "inverse_width",
        "attention_gain": float(model.model.blocks[0].sa.attention_gain().flatten()[0]),
        "learning_rate": effective.learning_rate,
        "weight_decay": effective.weight_decay,
        "warmup": case.warmup,
        "batch": case.batch,
        "updates": updates,
        "observe": case.observe,
        "target": case.target,
        "train_examples": len(task.splits["train"]),
        "validation_examples": len(task.splits["validation"]),
        "parameters": parameters,
        "parameter_count": offset,
        "initialization": parameter_metadata(model.model),
        "data_fingerprints": task.fingerprints,
        "model_source_reference": reference,
        "configuration": config.metadata(case),
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Prepare a reproducible binary input bundle, never train on export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--name", default="cubic-depth-normalized-lr2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=100_000)
    args = parser.parse_args()
    value = export(args.directory, args.name, args.seed, args.updates)
    print(
        json.dumps(
            {
                "directory": str(args.directory),
                "parameters": value["parameter_count"],
                "updates": value["updates"],
            }
        )
    )


if __name__ == "__main__":
    main()
