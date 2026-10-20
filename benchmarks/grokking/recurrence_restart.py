"""Fork a full DiffGPT checkpoint and repeat warmup with the same optimizer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from benchmarks.grokking.div97 import write_json
from benchmarks.grokking.recurrence import (
    RATE_FIELDS,
    ROOT,
    SOURCE_SNAPSHOT,
    Config,
    run,
    verify,
)


def prepare_fork(parent: Path, directory: Path, config: Config) -> dict[str, object]:
    """Retain every iterate and data state; change only the explicit rate settings."""
    snapshot = parent.read_bytes()
    saved = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    original = Config(**saved["config"])
    allowed = {"learning_rate", "epochs", "restart_step", "restart_warmup"}
    if any(
        value != asdict(config)[key]
        for key, value in asdict(original).items()
        if key not in allowed
    ):
        msg = "a warmup fork can change only its rate settings and epoch budget"
        raise ValueError(msg)
    group = saved["optimizer"]["param_groups"][0]
    if original.restart_step:
        msg = "fork from an original constant-rate checkpoint"
        raise ValueError(msg)
    if (
        config.restart_step != group["step"]
        or not config.restart_step
        or config.restart_warmup < 0
        or config.epochs <= saved["epoch"]
    ):
        msg = "restart must use this checkpoint's step and extend its epoch budget"
        raise ValueError(msg)
    updates = math.ceil(saved["train_rows"] / config.batch_size)
    if group["step"] != saved["epoch"] * updates or group["training"]:
        msg = "incomplete optimizer checkpoint"
        raise ValueError(msg)
    if group["betas"] != (config.beta1, config.beta2):
        msg = "checkpoint optimizer betas do not match its config"
        raise ValueError(msg)
    if sorted(saved["sampler_order"]) != list(range(saved["train_rows"])):
        msg = "incomplete data sampler checkpoint"
        raise ValueError(msg)
    for state in saved["optimizer"]["state"].values():
        if any(
            key not in state or not torch.isfinite(state[key]).all()
            for key in ("base", "evaluation", "second")
        ):
            msg = "incomplete or nonfinite optimizer iterates"
            raise ValueError(msg)
    # These benchmark curves contain one numeric CSV record per epoch.
    source_history = (parent.parent / "history.csv").read_bytes()
    shared_history = b"".join(
        source_history.splitlines(keepends=True)[: saved["epoch"] + 2]
    )
    rows = list(csv.DictReader(io.StringIO(shared_history.decode())))
    if [int(row["epoch"]) for row in rows] != list(range(saved["epoch"] + 1)):
        msg = "incomplete parent curve"
        raise ValueError(msg)
    if any(
        float(rows[-1][f"{split}_{key}"]) != saved["metrics"][split][key]
        for split in ("train", "test")
        for key in ("accuracy", "decoded_accuracy")
    ):
        msg = "parent curve and checkpoint disagree"
        raise ValueError(msg)
    # The source has a constant nominal rate. Reconstruct only these past
    # scalar diagnostics; every accuracy/loss observation is copied unchanged.
    total, step, rate = 0.0, 0, 0.0
    for row in rows:
        while step < int(row["step"]):
            step += 1
            ramp = min(1.0, step / original.warmup) if original.warmup else 1.0
            rate = original.learning_rate * math.sqrt(1 - original.beta2**step) * ramp
            total += rate**2
        row.update(
            zip(
                RATE_FIELDS,
                (original.learning_rate, rate, rate**2 / total if total else 0.0),
                strict=True,
            )
        )
    if not math.isclose(total, group["total"], rel_tol=1e-12, abs_tol=1e-15):
        msg = "parent averaging total does not match constant-rate history"
        raise ValueError(msg)
    metadata = json.loads((parent.parent / "metadata.json").read_text())
    provenance = {
        "parent": str(parent.resolve()),
        "parent_sha256": hashlib.sha256(snapshot).hexdigest(),
        "parent_epoch": saved["epoch"],
        "parent_config": asdict(original),
        "parent_source_sha256": metadata["source_sha256"],
        "shared_history_sha256": hashlib.sha256(shared_history).hexdigest(),
        "optimizer_steps_retained": group["step"],
        "averaging_total_retained": group["total"],
        "optimizer_and_sampler_reset": False,
        "config": asdict(config),
    }
    directory.mkdir(parents=True)
    (directory / "parent-checkpoint.pt").write_bytes(snapshot)
    (directory / "shared-history.csv").write_bytes(shared_history)
    with (directory / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved["config"] = asdict(config)
    torch.save(saved, directory / "checkpoint.pt")
    metadata["config"] = asdict(config)
    metadata["parent_source_sha256"] = metadata["source_sha256"]
    metadata["source_sha256"] = {
        name: hashlib.sha256(contents).hexdigest()
        for name, contents in SOURCE_SNAPSHOT.items()
    }
    restart_source = Path(__file__).read_bytes()
    metadata["source_sha256"][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
        restart_source
    ).hexdigest()
    metadata["restart"] = provenance
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "restart.json", provenance)
    (directory / "measured_source.py").write_bytes(
        SOURCE_SNAPSHOT["benchmarks/grokking/recurrence.py"]
    )
    (directory / "measured_field_sampler.py").write_bytes(
        SOURCE_SNAPSHOT["benchmarks/grokking/recurrence_field_sampler.py"]
    )
    (directory / "measured_restart.py").write_bytes(restart_source)
    write_json(
        directory / "status.json",
        {
            "config": asdict(config),
            **{
                key: saved[key]
                for key in (
                    "epoch",
                    "seconds",
                    "metrics",
                    "criterion",
                    "consecutive_epochs_at_99",
                )
            },
        },
    )
    return provenance


def main() -> None:
    """Continue a frozen parent with a deterministic second warmup, without reset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    saved = torch.load(args.parent, map_location="cpu", weights_only=True)
    config = replace(
        Config(**saved["config"]),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        restart_step=saved["optimizer"]["param_groups"][0]["step"],
        restart_warmup=args.warmup,
    )
    del saved
    directory = args.output / config.name()
    if not args.resume:
        print(json.dumps(prepare_fork(args.parent, directory, config)), flush=True)
    run(config, directory, args.device, resume=True)
    print(json.dumps(verify(directory, args.device, full=True)), flush=True)


if __name__ == "__main__":
    main()
