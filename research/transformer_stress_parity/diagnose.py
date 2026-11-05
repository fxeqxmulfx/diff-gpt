"""Decompose frozen parent errors into odd, even and cross terms without training."""

from __future__ import annotations

import hashlib
import json

import torch

from research.transformer_stress.common import ROOT, write_json
from research.transformer_stress_fixes.learning import ShiftedResidualNetwork


def main() -> None:
    """Inspect checkpoints; the auxiliary negative inputs are not training data."""
    torch.set_num_threads(1)
    base = ROOT / "runs/transformer-stress/rates-v1"
    splits = torch.load(base / "data.pt", weights_only=True)
    x = splits["heldout"]["input"]
    rows = []
    for path in sorted(base.glob("lr0.01/*/checkpoint.pt")):
        checkpoint = torch.load(path, weights_only=False)
        result = checkpoint["result"]
        model = ShiftedResidualNetwork(
            checkpoint["settings"], result["branches"], result["variant"]
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        with torch.no_grad():
            positive = (model(x) - x).double()
            negative = (model(-x) + x).double()
            odd, even = (positive - negative) / 2, (positive + negative) / 2
            target = (result["strength"] * splits["heldout"]["residual"]).double()
            denominator = target.square().mean()
            error = odd - target
            rows.append(
                {
                    "checkpoint": str(path.relative_to(ROOT)),
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "depth": result["branches"],
                    "strength": result["strength"],
                    "seed": result["seed"],
                    "heldout_relative_mse": result["final_heldout_relative_mse"],
                    "odd_error": (error.square().mean() / denominator).item(),
                    "even_energy": (even.square().mean() / denominator).item(),
                    "cross_term": (2 * (error * even).mean() / denominator).item(),
                }
            )
    output = {
        "cells": rows,
        "scope": (
            "Post-hoc diagnosis on the original heldout inputs and their negatives; "
            "no training, no changed original metric or selection. The real teacher "
            "is odd. Original FP32 target addition/subtraction introduces small "
            "rounding differences in decomposition totals."
        ),
    }
    write_json(ROOT / "runs/transformer-stress/parity-v1/diagnosis.json", output)
    print(json.dumps({"diagnosed_existing_cells": len(rows)}))


if __name__ == "__main__":
    main()
