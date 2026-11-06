"""Separate captured-update behavior from compiled versus eager rounding."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research/early99_core_ablations/results"


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Report absolute and normwise errors, retaining near-zero coordinates."""
    difference = actual.double() - expected.double()
    return {
        "max_absolute": difference.abs().max().item(),
        "relative_l2": (
            difference.norm() / expected.double().norm().clamp_min(1e-30)
        ).item(),
        "expected_norm": expected.double().norm().item(),
    }


def main() -> None:
    """Keep the exact failed fixture and expose all per-tensor differences."""
    path = ROOT / "tests/diff_gpt/application/test_early99_core_ablations.py"
    spec = importlib.util.spec_from_file_location("core_validation", path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    torch.set_float32_matmul_precision("highest")
    torch.compiler.reset()
    case = helpers.replace(helpers.small_case(), width=16, layers=2)
    config = helpers.Configuration(mechanism="scaled_peri_rms")
    task = helpers.arithmetic(case, "cuda")
    torch.manual_seed(11)
    factory = helpers.configured_factories(config)
    model = factory.model(task, case, "cuda")
    models = {
        "captured": model,
        "compiled": copy.deepcopy(model),
        "eager": copy.deepcopy(model),
    }
    stepper = helpers.FactoryStepper(
        model,
        task,
        case,
        "cuda",
        configuration=config,
        factories=factory,
        journal_path=ROOT
        / "runs/early99-core-ablations/diagnostic/execution-journal.json",
        compiled=True,
    )
    others = {
        name: helpers.Stepper(
            m, task, config.training_case(case), "cuda", compiled=name == "compiled"
        )
        for name, m in models.items()
        if name != "captured"
    }
    sampler = helpers.EpochSampler(
        len(task.splits["train"]), case.batch, wrap=False, seed=10011
    )
    stepper.prepare(sampler.sizes())
    rows = []
    names = list(dict(model.named_parameters()))
    for completed in range(case.max_updates + 1):
        parts, _ = sampler.next()
        indices = helpers.gather(parts)
        stepper.step(indices, completed)
        for name, other in others.items():
            other.rate.fill_(
                config.learning_rate_at(case, completed, len(task.splits["train"]))
            )
            other.update(indices.to("cuda"))
            rows.append(
                {
                    "completed": completed + 1,
                    "reference": name,
                    "parameters": {
                        n: error(p, dict(models[name].named_parameters())[n])
                        for n, p in model.named_parameters()
                    },
                    "moments": {
                        n: {
                            field: error(a[field], b[field])
                            for field in ("exp_avg", "exp_avg_sq")
                        }
                        for n, a, b in zip(
                            names,
                            stepper.optimizer.state.values(),
                            other.optimizer.state.values(),
                            strict=True,
                        )
                    },
                }
            )
    output = {
        "fixture": "unchanged original failed scaled_peri_rms fixture",
        "steps": rows,
    }
    (RESULTS / "cuda-rounding-diagnostic.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    for reference in others:
        group = [r for r in rows if r["reference"] == reference]
        print(
            json.dumps(
                {
                    "reference": reference,
                    "parameter_max_absolute": max(
                        v["max_absolute"]
                        for r in group
                        for v in r["parameters"].values()
                    ),
                    "parameter_max_relative_l2": max(
                        v["relative_l2"]
                        for r in group
                        for v in r["parameters"].values()
                    ),
                    "moment_max_absolute": max(
                        v["max_absolute"]
                        for r in group
                        for fs in r["moments"].values()
                        for v in fs.values()
                    ),
                    "moment_max_relative_l2": max(
                        v["relative_l2"]
                        for r in group
                        for fs in r["moments"].values()
                        for v in fs.values()
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
