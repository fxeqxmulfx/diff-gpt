"""Diagnose both symmetries in every saved decay cell without further training."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path

import torch

from research.transformer_stress.common import ROOT, require, write_json
from research.transformer_stress_decay.run import (
    OUTPUT,
    PLAN,
    cell_directory,
    key,
    prepare,
    read,
    sha256,
)
from research.transformer_stress_parity.models import ConfiguredNetwork

CONTROL_DECAY = 0.01


def directory(row: dict) -> Path:
    """Locate all reused controls and new cells without selecting by success."""
    activation, teacher = row["activation_variant"], row["teacher"]
    depth, strength, seed = row["branches"], row["strength"], row["seed"]
    if row["weight_decay"] != CONTROL_DECAY:
        return cell_directory(
            activation, row["weight_decay"], teacher, depth, strength, seed
        )
    name = f"d{depth}-a{strength}-seed{seed}"
    if activation == "shifted" and teacher == "original_odd":
        return ROOT / "runs/transformer-stress/rates-v1/lr0.01" / name
    return ROOT / "runs/transformer-stress/parity-v1" / teacher / activation / name


@torch.no_grad()
def components(
    model: ConfiguredNetwork,
    x: torch.Tensor,
    odd: torch.Tensor,
    even: torch.Tensor,
    strength: float,
) -> dict:
    """Decompose a fixed symmetric diagnostic, distinct from the primary metric."""
    plus = (model(x).double() - x.double()) / strength
    minus = (model(-x).double() + x.double()) / strength
    predicted_odd, predicted_even = (plus - minus) / 2, (plus + minus) / 2
    odd_error = (predicted_odd - odd).square().mean()
    even_error = (predicted_even - even).square().mean()
    total = (
        (plus - odd - even).square().mean() + (minus + odd - even).square().mean()
    ) / 2
    energy = odd.square().mean() + even.square().mean()
    require(bool(torch.isfinite(total)) and bool(energy > 0), "Invalid component probe")
    require(
        math.isclose(float(total), float(odd_error + even_error), rel_tol=1e-12),
        "Odd/even decomposition does not reconstruct symmetric error",
    )
    return {
        "odd_error_over_total_teacher_energy": float(odd_error / energy),
        "even_error_over_total_teacher_energy": float(even_error / energy),
        "symmetric_error_over_total_teacher_energy": float(total / energy),
        "teacher_odd_energy": float(odd.square().mean()),
        "teacher_even_energy": float(even.square().mean()),
        "even_fraction_of_squared_error": float(even_error / total),
    }


def main() -> None:
    """Preserve all 144 cells, both splits and full probe/checkpoint provenance."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    plan = read(PLAN)
    prepare(plan)
    summary = read(OUTPUT / "summary.json")
    require(summary["complete"], "Finish the frozen study first")
    old = torch.load(OUTPUT / "data.pt", weights_only=True)
    mixed = torch.load(OUTPUT / "mixed-data.pt", weights_only=True)
    rows = []
    for row in summary["cells"]:
        path = directory(row) / "checkpoint.pt"
        saved = torch.load(path, weights_only=False)
        require(
            all(row.get(k) == v for k, v in saved["result"].items()),
            "Checkpoint result differs from summary",
        )
        require(
            saved["settings"]["weight_decay"] == row["weight_decay"],
            "Checkpoint decay differs",
        )
        settings = dict(saved["settings"], activation_variant=row["activation_variant"])
        model = ConfiguredNetwork(settings, row["branches"], row["variant"])
        model.load_state_dict(saved["model"], strict=True)
        result = {
            k: row[k]
            for k in (
                "activation_variant",
                "weight_decay",
                "teacher",
                "branches",
                "strength",
                "seed",
                "status",
                "first_threshold_updates",
                "final_train_relative_mse",
                "final_heldout_relative_mse",
            )
        }
        result["checkpoint_sha256"] = sha256(path)
        result["checkpoint"] = str(path.relative_to(ROOT))
        for split in ("train", "heldout"):
            odd = old[split]["residual"].double()
            even = mixed[split]["residual"].double() - odd
            if row["teacher"] == "original_odd":
                even = torch.zeros_like(even)
            result[f"{split}_and_reflections"] = components(
                model, old[split]["input"], odd, even, row["strength"]
            )
        rows.append(result)
    require(len(rows) == plan["total_comparison_cells"], "Incomplete diagnosis")
    require(len({key(row) for row in rows}) == len(rows), "Duplicate diagnosed cell")
    groups = []
    for activation, decay, teacher in itertools.product(
        plan["activation_variants"],
        [plan["control_weight_decay"], *plan["new_weight_decays"]],
        plan["teachers"],
    ):
        group = [
            r
            for r in rows
            if (r["activation_variant"], r["weight_decay"], r["teacher"])
            == (activation, decay, teacher)
        ]
        groups.append(
            {
                "activation_variant": activation,
                "weight_decay": decay,
                "teacher": teacher,
                "cells": len(group),
                **{
                    split: {
                        metric: statistics.mean(r[split][metric] for r in group)
                        for metric in group[0][split]
                    }
                    for split in ("train_and_reflections", "heldout_and_reflections")
                },
            }
        )
    result = {
        "scope": (
            "Read-only post-hoc diagnosis of every saved decay cell. No training, "
            "augmentation during training, reselection or replacement "
            "of primary scores."
        ),
        "equation": (
            "Evaluate residual F(x)-x and F(-x)+x, divide by original strength, "
            "and split into their half-difference and half-sum. Symmetric squared "
            "error is the sum of odd and even squared errors."
        ),
        "target": (
            "Odd component is the frozen original tanh residual. Even component is "
            "frozen mixed minus original residual, in FP64, retaining quantization "
            "in the original plus-side label. Reflect these components algebraically; "
            "no original target array changes."
        ),
        "denominator": (
            "Odd target energy plus even target energy on the same split. These "
            "additional symmetric probe metrics are distinct from the original "
            "held-out relative MSE and its threshold."
        ),
        "rows": rows,
        "groups": groups,
        "summary_sha256": sha256(OUTPUT / "summary.json"),
        "audit_sha256": sha256(OUTPUT / "audit.json"),
        "data_sha256": {
            name: sha256(OUTPUT / name) for name in ("data.pt", "mixed-data.pt")
        },
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(
        Path(__file__).with_name("results") / "residual-parity-components.json", result
    )
    print(json.dumps({"cells": len(rows), "groups": groups}))


if __name__ == "__main__":
    main()
