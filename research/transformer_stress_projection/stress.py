"""Compare actual projection arithmetic to independent analytic NumPy equations."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import torch

from research.transformer_stress.common import ROOT, require, write_json
from research.transformer_stress_bias.run import read, sha256
from research.transformer_stress_projection.models import project_branch


def reference(value: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the Euclidean ball formula and its analytic interior/exterior VJP."""
    radius = math.sqrt(value.size)
    length = float(np.sqrt(np.dot(value, value)))
    if length < radius:
        return value.copy(), probe.copy()
    unit = value / length
    return radius * unit, (radius / length) * (probe - unit * np.dot(unit, probe))


def check_grid(plan: dict) -> dict:
    """Check every declared dtype/width/scale, including zero and enormous inputs."""
    grid = plan["numeric_grid"]
    generator = torch.Generator().manual_seed(grid["seed"])
    rows = []
    for width, scale, name in itertools.product(
        grid["widths"], grid["scales"], grid["dtypes"]
    ):
        dtype = getattr(torch, name)
        x = (
            (torch.randn(width, generator=generator, dtype=torch.float64) * scale)
            .to(dtype)
            .requires_grad_()
        )
        probe = torch.randn(width, generator=generator, dtype=torch.float64).to(dtype)
        second = (
            torch.randn(width, generator=generator, dtype=torch.float64) * scale
        ).to(dtype)
        output = project_branch(x)
        gradient = torch.autograd.grad((output * probe).sum(), x)[0]
        expected, vjp = reference(x.detach().double().numpy(), probe.double().numpy())
        forward_error = float(
            np.max(np.abs(output.detach().double().numpy() - expected))
        )
        gradient_error = float(np.max(np.abs(gradient.double().numpy() - vjp)))
        radius = output.detach().double().square().mean().sqrt().item()
        distance = torch.linalg.vector_norm(
            x.detach().double() - second.double()
        ).item()
        output_distance = torch.linalg.vector_norm(
            output.detach().double() - project_branch(second).double()
        ).item()
        ratio = output_distance / distance if distance else 0.0
        finite = bool(torch.isfinite(output).all() and torch.isfinite(gradient).all())
        passed = (
            finite
            and forward_error <= grid["forward_absolute_tolerance"]
            and gradient_error <= grid["gradient_absolute_tolerance"]
            and radius <= 1 + grid["nonexpansive_slack"]
            and ratio <= 1 + grid["nonexpansive_slack"]
        )
        rows.append(
            {
                "width": width,
                "scale": scale,
                "dtype": name,
                "finite": finite,
                "forward_absolute_error": forward_error,
                "gradient_absolute_error": gradient_error,
                "output_rms": radius,
                "paired_distance_ratio": ratio,
                "passed": passed,
            }
        )
    own = Path(__file__)
    paths = [own, own.with_name("models.py"), own.with_name("plan.json")]
    return {
        "expected": grid["cases"],
        "passed": sum(r["passed"] for r in rows),
        "maximum_forward_error": max(r["forward_absolute_error"] for r in rows),
        "maximum_gradient_error": max(r["gradient_absolute_error"] for r in rows),
        "maximum_distance_ratio": max(r["paired_distance_ratio"] for r in rows),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "scope": "Isolated projection arithmetic; not full-network validation.",
        "cells": rows,
    }


def main() -> None:
    """Preserve every stress row before requiring that the declared checks pass."""
    torch.set_num_threads(1)
    result = check_grid(read(Path(__file__).with_name("plan.json")))
    write_json(Path(__file__).parent / "results/numeric.json", result)
    require(
        result["passed"] == result["expected"], "Numeric checks failed; retain rows"
    )
    print({k: v for k, v in result.items() if k != "cells"})


if __name__ == "__main__":
    main()
