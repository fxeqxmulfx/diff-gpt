"""Independently audit complete temporal data and replay verified checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from diff_gpt.application.grokking import recurrence as experiment
from diff_gpt.application.grokking.div97 import digest, load_checkpoint, write_json
from diff_gpt.application.grokking.div97_report import classify, history, replay_browser
from diff_gpt.domain.encoder_decoder import encode

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from diff_gpt.application.ports import BrowserReplay

MASKED_LABEL = -100


def require(condition: object, message: str) -> None:
    """Keep audit checks active even when Python runs with optimization enabled."""
    if not condition:
        raise ValueError(message)


def audit(directory: Path, device: str) -> dict[str, Any]:
    """Check every state, every shift, actual encoder causality and full SF state."""
    saved = load_checkpoint(directory / "checkpoint.pt", "cpu")
    metadata = json.loads((directory / "metadata.json").read_text())
    config = experiment.Config(**saved["config"])
    raw = saved["raw"].numpy()
    train = saved["train_rows"]
    modulus = experiment.P
    expected_size = (modulus - 1) ** 2 if config.task == "division" else modulus**2 - 1
    require(len(raw) == expected_size, "incomplete recurrence state space")
    require(np.unique(raw[:, :2], axis=0).shape[0] == expected_size, "duplicate states")
    require(
        raw.min() >= int(config.task == "division") and raw.max() < modulus,
        "noncanonical observed values",
    )
    require(not np.any(np.all(raw[:, :2] == 0, axis=1)), "absorbing zero state")
    if config.task == "division":
        np.testing.assert_array_equal(raw[:, 2:] * raw[:, 1:-1] % modulus, raw[:, :-2])
    else:
        np.testing.assert_array_equal(
            (raw[:, 2:] - raw[:, 1:-1] - raw[:, :-2]) % modulus, 0
        )
    # A full partition of the state space, closed under one step, cannot split
    # any orbit. This does not reuse the generator's orbit enumeration.
    membership = np.full((modulus, modulus), -1, dtype=np.int64)
    membership[raw[:train, 0], raw[:train, 1]] = 0
    membership[raw[train:, 0], raw[train:, 1]] = 1
    expected_split = (np.arange(len(raw)) >= train).astype(np.int64)
    for position in range(raw.shape[1] - 1):
        np.testing.assert_array_equal(
            membership[raw[:, position], raw[:, position + 1]], expected_split
        )
    data = metadata["data"]
    require(
        not data["synthetic_prefix"] and data["difference_axis"] == "rows",
        "unexpected synthetic prefix or difference axis",
    )
    require(data["columns"] == ["x"], "the task must have one temporal column")
    require(digest(raw) == data["raw_sha256"], "raw data hash mismatch")
    vocabulary = experiment.grid_vocabulary(config)
    domains = np.repeat(experiment.grid_domain(config), len(raw))
    _, grid, full_tokens = encode(
        raw.T.astype(np.float64),
        vocabulary,
        domains,
        config.order,
        tokenizer=config.tokenizer,
    )
    _, observed_grid, observed_tokens = encode(
        raw[:, :-1].T.astype(np.float64),
        vocabulary,
        domains,
        config.order,
        tokenizer=config.tokenizer,
    )
    np.testing.assert_array_equal(grid, observed_grid)
    np.testing.assert_array_equal(full_tokens[:-1], observed_tokens)
    inputs, labels = experiment.tensors(raw, config, "cpu")
    np.testing.assert_array_equal(inputs.numpy().T, observed_tokens)
    selected = labels.numpy() != MASKED_LABEL
    np.testing.assert_array_equal(labels.numpy()[selected], full_tokens[1:].T[selected])
    (group,) = saved["optimizer"]["param_groups"]
    steps_per_epoch = math.ceil(train / config.batch_size)
    require(metadata["optimizer"] == "ScheduleFreeAdamW", "unexpected optimizer")
    require(
        metadata["updates_per_epoch"] == steps_per_epoch,
        "metadata optimizer step count mismatch",
    )
    require(
        group["step"] == saved["epoch"] * steps_per_epoch, "optimizer step mismatch"
    )
    require(not group["training"] and group["total"] > 0, "invalid averaging state")
    require(
        sorted(saved["sampler_order"]) == list(range(train)), "invalid sampler order"
    )
    require(0 <= saved["sampler_state"] < 2**64, "invalid batch sampler RNG state")
    model = experiment.build(config, "cpu")
    model.load_state_dict(saved["model"])
    require(
        sum(parameter.numel() for parameter in model.parameters())
        == metadata["parameters"],
        "model parameter count mismatch",
    )
    for identifier, parameter in zip(group["params"], model.parameters(), strict=True):
        state = saved["optimizer"]["state"][identifier]
        require(state.keys() == {"base", "second"}, "optimizer iterates are not z, v")
        for name in ("base", "second"):
            require(
                state[name].shape == parameter.shape
                and torch.isfinite(state[name]).all(),
                "invalid optimizer tensor",
            )
        require((state["second"] >= 0).all(), "negative second moment")
    rows = history(directory)
    require(
        [row["epoch"] for row in rows] == list(range(saved["epoch"] + 1)),
        "incomplete curve",
    )
    tail = rows[-config.after99 - 1 :]
    require(len(tail) == config.after99 + 1, "not enough stability observations")
    require(
        min(min(row["test_accuracy"], row["test_decoded_accuracy"]) for row in tail)
        >= experiment.TARGET_ACCURACY,
        "test accuracy does not remain above target",
    )
    del model, saved
    verification = experiment.verify(directory, device, full=True)
    result = {
        "directory": str(directory.resolve().relative_to(experiment.ROOT)),
        "config": config.__dict__,
        "raw_rows": len(raw),
        "train_rows": train,
        "test_rows": len(raw) - train,
        "all_state_transitions_checked": True,
        "all_orbit_shifts_disjoint": True,
        "all_encoder_inputs_match_observed_only_encoding": True,
        "all_supervised_tokens_match_actual_encoder": True,
        "full_optimizer_and_sampler_state": True,
        "optimizer_steps": verification["epoch"] * steps_per_epoch,
        "updates_per_epoch": steps_per_epoch,
        "stable_observations": len(tail),
        "minimum_test_accuracy_in_stable_tail": min(
            row["test_accuracy"] for row in tail
        ),
        "criterion": classify(rows),
        "query_criterion": classify(rows, "query_"),
        "verification": verification,
        "sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "checkpoint.pt",
                "history.csv",
                "metadata.json",
                "verification.json",
            )
        },
    }
    write_json(directory / "full-audit.json", result)
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def audit_report(args: argparse.Namespace, *, node: BrowserReplay) -> None:
    """Audit each run of the report, then write the audit report."""
    source = json.loads(args.report.read_text())
    results = []
    for run in source["runs"]:
        result = audit(experiment.ROOT / run["directory"], args.device)
        results.append(result)
        print(
            json.dumps(
                {
                    "audited": result["directory"],
                    "public_correct": result["verification"]["public_correct"],
                    "public_rows": result["verification"]["public_rows"],
                }
            ),
            flush=True,
        )
    report = {"runs": results, "evaluation_batch": args.evaluation_batch}
    if args.browser_module:
        report["browser_criterion_replay"] = replay_browser(
            results, args.browser_module, node=node
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
