"""
Replay Basis hard recall and stop at the first regressing feature.

Run from the repository root with .venv/bin/python -m research.recall_ablation.run.
The baseline must pass before a feature can run. Training and validation are
the historical streams; final test is never loaded by the feature screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from diff_gpt.application.mqar import IGNORE, MQAR, build_split, fingerprint
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW
from research.recall_ablation.features import (
    ARCHITECTURE_STAGES,
    PREDECESSOR,
    build_model,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "src/diff_gpt/domain/model/gpt.py",
    "src/diff_gpt/domain/model/mini_block.py",
    "src/diff_gpt/domain/model/centered_xsa.py",
    "src/diff_gpt/domain/model/centered_mini.py",
    "src/diff_gpt/domain/model/mha.py",
    "src/diff_gpt/domain/model/rope.py",
    "src/diff_gpt/domain/model/feed_forward.py",
    "src/diff_gpt/domain/model/rms_norm.py",
    "src/diff_gpt/domain/optimizer/schedule_free.py",
    "src/diff_gpt/application/mqar.py",
    "research/recall_ablation/run.py",
    "research/recall_ablation/features.py",
    "research/recall_ablation/confirm.py",
)
TASK = MQAR(symbols=256, pairs=16, overwrites=8, queries=8)
RATE = 3e-4
WARMUP = 50
BATCH = 64
OBSERVE = 50
TARGET = 0.99
TRAIN_ROWS = 20_000


def write_json(path: Path, value: object) -> None:
    """Replace a report atomically, retaining every observation separately."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def split(name: str, count: int) -> tuple[tuple[torch.Tensor, ...], str]:
    """Pack the lab's variable-length examples at their fixed compiled width."""
    examples = build_split(TASK, 64, name, 1, count, min_length=48)
    digest = fingerprint(TASK, 64, name, 1, examples, min_length=48)
    tokens = torch.zeros(count, 64, dtype=torch.long)
    positions = torch.empty(count, 8, dtype=torch.long)
    targets = torch.empty(count, 8, dtype=torch.long)
    for row, (inp, labels) in enumerate(examples):
        tokens[row, : len(inp)] = torch.tensor(inp)
        query = [i for i, label in enumerate(labels) if label != IGNORE]
        positions[row] = torch.tensor(query)
        targets[row] = torch.tensor([labels[i] for i in query])
    return (tokens, positions, targets), digest


def validate_confirmation(metadata: dict, reference: dict) -> None:
    """Require a passed, identical architecture/recipe while allowing new seeds."""
    if (
        reference.get("status") != "passed"
        or reference.get("first99") is None
        or metadata["stage"] != reference.get("stage")
    ):
        msg = "confirmation requires a passed reference for the same stage"
        raise ValueError(msg)
    if metadata["batch_seed"] != metadata["seed"] + 10_000:
        msg = "confirmation batch_seed must remain model seed + 10000"
        raise ValueError(msg)
    seed_specific = {
        "seed",
        "batch_seed",
        "initial_parameters_sha256",
        "parent_initial_parameters_sha256",
        "historical_initial_parameters_sha256",
        "shared_non_mlp_initial_sha256",
        "git_commit",
        "sources",
    }
    for field, value in metadata.items():
        if field not in seed_specific and (
            field not in reference or value != reference[field]
        ):
            msg = f"confirmation changes the accepted reference's {field}"
            raise ValueError(msg)
    for name, digest in reference["sources"].items():
        # Only this driver changes to add the confirmation eligibility branch.
        # Model, data, feature builder and optimizer sources must stay exact.
        if (
            name != "research/recall_ablation/run.py"
            and metadata["sources"].get(name) != digest
        ):
            msg = f"confirmation changes model/data source {name}"
            raise ValueError(msg)


def run(
    args: argparse.Namespace,
    seed: int,
    data: dict,
    *,
    confirmation_reference: dict | None = None,
) -> dict:
    """Run one fresh, matched seed; save full state at its endpoint."""
    directory = args.output / args.stage / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    model, feature_metadata = build_model(args.stage, seed)
    initial = hashlib.sha256()
    for parameter in model.parameters():
        initial.update(parameter.detach().numpy().tobytes())
    settings = {"lr": RATE, "betas": (0.9, 0.98), "weight_decay": 0.1}
    optimizer = (
        torch.optim.AdamW(model.parameters(), **settings)
        if args.stage != "schedule-free"
        else ScheduleFreeAdamW(model.parameters(), **settings, warmup=0)
    )

    def objective(
        tokens: torch.Tensor, positions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits = model(tokens, positions=positions)[0]
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    def prediction(tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return model(tokens, positions=positions)[0]

    started = time.perf_counter()
    train_loss = (
        torch.compile(objective, fullgraph=True, dynamic=False)
        if args.compile
        else objective
    )
    predict = (
        torch.compile(prediction, fullgraph=True, dynamic=False)
        if args.compile
        else prediction
    )
    generator = torch.Generator().manual_seed(seed + 10_000)
    permutation = torch.randperm(TRAIN_ROWS, generator=generator)
    cursor = 0
    train, validation = data["train"], data["validation"]
    metadata = {
        "stage": args.stage,
        "seed": seed,
        "max_updates": args.max_updates,
        "model": "historical GPTMini, 128x6, 4 heads",
        **feature_metadata,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_parameters_sha256": initial.hexdigest(),
        "optimizer": type(optimizer).__name__,
        "learning_rate": RATE,
        "betas": [0.9, 0.98],
        "weight_decay": 0.1,
        "warmup": WARMUP,
        "warmup_convention": "completed_updates, external schedule in both arms",
        "batch": BATCH,
        "epoch_tail": 32,
        "data_seed": 1,
        "batch_seed": seed + 10_000,
        "evaluate_every": OBSERVE,
        "target_sequence_accuracy": TARGET,
        "compiled": args.compile,
        "threads": args.threads,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(  # noqa: S603 -- fixed read-only git command
            [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
        "split_fingerprints": data["fingerprints"],
        "test_loaded": False,
    }
    if confirmation_reference is not None:
        validate_confirmation(metadata, confirmation_reference)
        metadata["experiment_role"] = "fresh_seed_confirmation"
        metadata["confirmation_reference"] = {
            "stage": confirmation_reference["stage"],
            "seed": confirmation_reference["seed"],
            "first99": confirmation_reference["first99"],
            "result_path": confirmation_reference.get("_result_path"),
            "result_sha256": confirmation_reference.get("_result_sha256"),
            "training_signature_sha256": confirmation_reference.get(
                "_training_signature_sha256"
            ),
        }
    elif args.stage != "baseline":
        baseline = json.loads(
            (args.output / "baseline" / f"seed{seed}" / "result.json").read_text()
        )
        if args.stage in ARCHITECTURE_STAGES:
            parent = json.loads(
                (
                    args.output
                    / PREDECESSOR[args.stage]
                    / f"seed{seed}"
                    / "result.json"
                ).read_text()
            )
            if (
                metadata["parent_initial_parameters_sha256"]
                != parent["initial_parameters_sha256"]
            ):
                msg = (
                    "the architectural candidate does not reproduce "
                    "its passing parent's initialization"
                )
                raise RuntimeError(msg)
        if (
            metadata["historical_initial_parameters_sha256"]
            != baseline["initial_parameters_sha256"]
        ):
            msg = "the historical initial weights differ from the recorded control"
            raise RuntimeError(msg)
        matched_fields = [
            "learning_rate",
            "betas",
            "weight_decay",
            "batch_seed",
            "split_fingerprints",
        ]
        if args.stage not in ARCHITECTURE_STAGES:
            matched_fields.extend(("initial_parameters_sha256", "parameters"))
        for field in matched_fields:
            if metadata[field] != baseline[field]:
                msg = f"feature changes the matched baseline's {field}"
                raise RuntimeError(msg)
    for name in SOURCES:
        snapshot = directory / "source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, snapshot)
    write_json(directory / "protocol.json", metadata)
    history = []
    first99 = None
    for completed in range(args.max_updates + 1):
        if completed % OBSERVE == 0 or completed == args.max_updates:
            if isinstance(optimizer, ScheduleFreeAdamW):
                optimizer.eval()
            model.eval()
            correct = exact = 0
            total_loss = 0.0
            with torch.no_grad():
                for start in range(0, 512, 64):
                    tokens, positions, targets = (
                        part[start : start + 64] for part in validation
                    )
                    logits = predict(tokens, positions)
                    hits = logits.argmax(-1) == targets
                    correct += int(hits.sum())
                    exact += int(hits.all(-1).sum())
                    total_loss += float(
                        F.cross_entropy(
                            logits.flatten(0, 1), targets.flatten(), reduction="sum"
                        )
                    )
            row = {
                "step": completed,
                "loss": total_loss / 4096,
                "token_accuracy": correct / 4096,
                "sequence_accuracy": exact / 512,
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            with (directory / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps({"stage": args.stage, "seed": seed, **row}), flush=True)
            if row["sequence_accuracy"] >= TARGET:
                first99 = completed
                break
            if completed == args.max_updates:
                break
            model.train()
            if isinstance(optimizer, ScheduleFreeAdamW):
                optimizer.train()
        if completed == args.max_updates:
            break
        if cursor == TRAIN_ROWS:
            permutation = torch.randperm(TRAIN_ROWS, generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + BATCH]
        cursor += len(indices)
        batch = tuple(part[indices] for part in train)
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(*batch)
        if not bool(torch.isfinite(loss)):
            msg = f"nonfinite training loss at {completed} updates"
            raise FloatingPointError(msg)
        loss.backward()
        optimizer.param_groups[0]["lr"] = RATE * min(1, completed / WARMUP)
        optimizer.step()
    if isinstance(optimizer, ScheduleFreeAdamW):
        optimizer.eval()
    model.eval()
    result = {
        **metadata,
        "first99": first99,
        "completed": completed,
        "endpoint": history[-1],
        "status": "passed" if first99 is not None else "censored",
    }
    write_json(directory / "result.json", result)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed": completed,
            "permutation": permutation,
            "cursor": cursor,
            "sampler_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "protocol": metadata,
        },
        directory / "endpoint.pt",
    )
    return result


def main() -> None:
    """Run one feature, refusing a failed or incompletely covered control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/recall-ablation")
    parser.add_argument(
        "--stage",
        choices=("baseline", "schedule-free", *ARCHITECTURE_STAGES),
        default="baseline",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-updates", type=int, default=4800)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if args.stage != "baseline":
        for seed in args.seeds:
            if args.stage in ARCHITECTURE_STAGES:
                parent = json.loads(
                    (
                        args.output
                        / PREDECESSOR[args.stage]
                        / f"seed{seed}"
                        / "result.json"
                    ).read_text()
                )
                if parent["status"] != "passed":
                    msg = "stop: the preceding architectural feature did not pass"
                    raise RuntimeError(msg)
            baseline = json.loads(
                (args.output / "baseline" / f"seed{seed}" / "result.json").read_text()
            )
            if baseline["status"] != "passed":
                msg = f"baseline seed {seed} did not pass; stop the ablation"
                raise RuntimeError(msg)
            if args.max_updates < baseline["first99"]:
                msg = "the feature budget must cover the baseline's first crossing"
                raise ValueError(msg)
    train, train_hash = split("train", TRAIN_ROWS)
    validation, validation_hash = split("validation", 512)
    data = {
        "train": train,
        "validation": validation,
        "fingerprints": {"train": train_hash, "validation": validation_hash},
    }
    for seed in args.seeds:
        result = run(args, seed, data)
        if result["status"] != "passed":
            print(
                json.dumps(
                    {
                        "stop": "first regression or baseline failure",
                        "stage": args.stage,
                        "seed": seed,
                    }
                ),
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
