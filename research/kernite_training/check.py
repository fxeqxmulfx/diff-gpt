"""Independent PyTorch checks of the standalone native output and derivatives."""

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
)
from research.current_grokking import run as _bootstrap  # noqa: F401
from research.current_grokking.data import build

LOSS_TOLERANCE = 1e-4


def check(bundle: Path, output: Path, device: str = "cuda") -> dict:
    """Compare each stored native state locally, plus the complete short trajectory."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    probe = json.loads((output / "probe.json").read_text())
    if (
        hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
        != probe["input_sha256"]
    ):
        message = "Probe used a different input bundle"
        raise ValueError(message)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    spec = resolve(manifest["name"])
    case = case_for(spec)
    task = build(case, device)
    torch.manual_seed(manifest["seed"])
    model = factories(spec).model(task, case, device)
    params = list(model.model.named_parameters())
    initial = torch.cat([p.detach().flatten().cpu() for _, p in params])

    def read(name: str, root: Path = output) -> torch.Tensor:
        return torch.from_numpy(np.fromfile(root / name, dtype="<f4").copy()).to(device)

    if not torch.equal(initial, read("parameters.f32", bundle).cpu()):
        message = "The exported initialization changed"
        raise AssertionError(message)
    order = np.fromfile(bundle / "order.u32", dtype="<u4")
    boundaries = np.fromfile(bundle / "boundaries.u32", dtype="<u4")
    rows = []

    def compare(
        label: str, actual: torch.Tensor, reference: torch.Tensor, tolerance: float
    ) -> None:
        a, b = actual.detach().double(), reference.detach().double()
        error = torch.linalg.vector_norm(a - b).item()
        norm = torch.linalg.vector_norm(b).item()
        relative = error / norm if norm else None
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        passed = finite and (
            relative <= tolerance if relative is not None else error == 0
        )
        rows.append(
            {
                "quantity": label,
                "relative_l2": relative,
                "absolute_l2": error,
                "max_abs": (a - b).abs().max().item(),
                "tolerance": tolerance,
                "passed": passed,
            }
        )

    def compare_parameters(
        label: str, flat: torch.Tensor, values: list[torch.Tensor], tolerance: float
    ) -> None:
        offset = 0
        for (name, parameter), reference in zip(params, values, strict=True):
            actual = flat[offset : offset + parameter.numel()].reshape_as(parameter)
            offset += parameter.numel()
            compare(f"{label}/{name}", actual, reference, tolerance)

    def batch(step: int) -> torch.Tensor:
        start, stop = map(int, boundaries[step : step + 2])
        indices = torch.from_numpy(order[start:stop].astype("int64")).to(device)
        return task.inputs(indices)

    # Local checks hold the input weights fixed at the native state's exact bytes.
    for row in probe["rows"]:
        step = row["step"]
        weights = read(f"step{step}-parameters.f32")
        offset = 0
        with torch.no_grad():
            for _, parameter in params:
                parameter.copy_(
                    weights[offset : offset + parameter.numel()].reshape_as(parameter)
                )
                offset += parameter.numel()
        model.zero_grad(set_to_none=True)
        logits, targets = task.forward(model, batch(step))
        loss = task.loss(logits, targets)
        loss.backward()
        compare(
            f"step{step}/logits",
            read(f"step{step}-logits.f32").reshape_as(logits),
            logits,
            1e-4,
        )
        compare_parameters(
            f"step{step}/gradients",
            read(f"step{step}-gradients.f32"),
            [p.grad for _, p in params],
            1e-3,
        )
        exact = int((logits[:, 1].argmax(-1) == targets[:, 1]).sum())
        rows.append(
            {
                "quantity": f"step{step}/exact",
                "native": row["hits"],
                "reference": exact,
                "passed": row["hits"] == exact,
            }
        )
        rows.append(
            {
                "quantity": f"step{step}/loss",
                "native": row["loss"],
                "reference": loss.item(),
                "passed": abs(row["loss"] - loss.item()) < LOSS_TOLERANCE,
            }
        )

    # Independent optimizer trajectory from the original initialization and batch order.
    offset = 0
    with torch.no_grad():
        for _, parameter in params:
            parameter.copy_(
                initial[offset : offset + parameter.numel()].reshape_as(parameter)
            )
            offset += parameter.numel()
    config = configuration(spec)
    effective = config.training_case(case)
    rate = torch.tensor(0.0, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=effective.weight_decay,
        fused=device == "cuda",
        foreach=False if device == "cpu" else None,
        capturable=device == "cuda",
    )
    for step in range(probe["steps"]):
        rate.fill_(effective.learning_rate * min(1.0, step / case.warmup))
        optimizer.zero_grad(set_to_none=True)
        logits, targets = task.forward(model, batch(step))
        task.loss(logits, targets).backward()
        optimizer.step()
    compare_parameters(
        "trajectory/parameters", read("parameters.f32"), [p for _, p in params], 1e-3
    )
    for filename, state_key in (
        ("first_moment.f32", "exp_avg"),
        ("second_moment.f32", "exp_avg_sq"),
    ):
        compare_parameters(
            f"trajectory/{state_key}",
            read(filename),
            [optimizer.state[p][state_key] for _, p in params],
            1e-3,
        )
    report = {
        "name": manifest["name"],
        "seed": manifest["seed"],
        "steps": probe["steps"],
        "reference_device": device,
        "torch": str(torch.__version__),
        "native": probe,
        "checks": rows,
        "passed": all(row["passed"] for row in rows),
    }
    (output / f"comparison-{device}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    """Keep a machine-readable record, including every failed comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = check(args.bundle, args.output, args.device)
    failures = [r for r in report["checks"] if not r["passed"]]
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(report["checks"]),
                "failures": failures,
            }
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
