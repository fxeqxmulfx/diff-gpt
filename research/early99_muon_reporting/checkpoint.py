"""Validate serialized matrix and auxiliary state against the frozen inventory."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from research.early99.protocol import specification
from research.early99_reporting.round2_audit import require

if TYPE_CHECKING:
    from research.early99_muon.factories import Configuration


def verify_optimizer_checkpoint(
    state: dict, metadata: dict, task: dict, config: Configuration, completed: int
) -> None:
    """Check every unique matrix, issued step, moment, rate and native option."""
    inventory = json.loads(
        Path(
            "research/early99_optimizer_followups/parameter-inventory.json"
        ).read_text()
    )
    original = next(r for r in inventory["rows"] if r["case"] == task["name"])
    shapes = {
        p["name"]: p["shape"]
        for p in [*original["hybrid_muon_matrices"], *original["hybrid_adamw_matrices"]]
    }
    hybrid = config.optimizer_kind == "muon_adamw"
    matrix = (
        [p["name"] for p in original["hybrid_muon_matrices"]]
        if hybrid
        else original["pure_muon_matrices"]
    )
    auxiliary = ["token_embedding_table.weight"] if hybrid else []
    expected = {
        "kind": config.optimizer_kind,
        "muon": [{"name": n, "shape": shapes[n]} for n in matrix],
        "adamw": [{"name": n, "shape": shapes[n]} for n in auxiliary],
    }
    require(
        state["partition"] == metadata["optimizer_partition"] == expected,
        "Partition differs from the frozen all-case inventory",
    )
    require(
        (state["adamw"] is not None) == hybrid, "Pure Muon contains an AdamW fallback"
    )
    require(
        len(set(matrix + auxiliary)) == original["trainable_tensor_count"],
        "Duplicate or missing tied-weight update",
    )
    case = specification(task)
    for label, names in [("muon", matrix), ("adamw", auxiliary)]:
        if not names:
            continue
        optimizer = state[label]
        require(len(optimizer["param_groups"]) == 1, "Unexpected optimizer groups")
        group = optimizer["param_groups"][0]
        require(group["param_names"] == names, "Parameter order differs from inventory")
        require(len(group["params"]) == len(names), "Missing optimizer parameters")
        require(set(optimizer["state"]) == set(group["params"]), "Missing moment state")
        for identifier, name in zip(group["params"], names, strict=True):
            moment = optimizer["state"][identifier]
            keys = {"momentum_buffer"} if label == "muon" else {"exp_avg", "exp_avg_sq"}
            require(set(moment) == keys | {"step"}, "Unexpected moment fields")
            require(int(moment["step"]) == completed, "Issued-update count differs")
            for key in keys:
                value = moment[key]
                require(
                    list(value.shape) == shapes[name]
                    and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()),
                    "Invalid optimizer moment",
                )
            if label == "adamw":
                require(
                    bool((moment["exp_avg_sq"] >= 0).all()), "Negative second moment"
                )
        settings = (
            {
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "ns_coefficients": (3.4445, -4.775, 2.0315),
                "eps": 1e-7,
                "adjust_lr_fn": "match_rms_adamw",
            }
            if label == "muon"
            else {
                "betas": (0.9, 0.98),
                "eps": 1e-8,
                "fused": True,
                "capturable": False,
                "amsgrad": False,
                "maximize": False,
            }
        )
        settings["weight_decay"] = case.weight_decay
        for key, value in settings.items():
            require(group[key] == value, f"Changed {label} setting {key}")
        expected_rate = (
            config.learning_rate_at(case, completed - 1, task["train_examples"])
            if label == "muon"
            else config.auxiliary_rate_at(case, completed - 1)
        )
        require(
            math.isclose(group["lr"], expected_rate, rel_tol=1e-12),
            "Final rate differs from issued update",
        )
