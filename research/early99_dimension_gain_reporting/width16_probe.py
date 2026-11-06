"""Compare eager and compiled gradients of the equal-gain width-16 controls."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from research.current_grokking import data
from research.current_grokking.train import Stepper
from research.early99.experiments.initialization import parameter_metadata
from research.early99.protocol import selected, specification
from research.early99_dimension_gain.factories import (
    Configuration,
    configured_factories,
)
from research.early99_dimension_gain.readiness import require, sha256
from research.early99_score_scale.factories import Configuration as Quarter
from research.early99_score_scale.factories import (
    configured_factories as quarter_factories,
)

CASES = (
    "suite-boolean-and-random",
    "gpt-probe-key-value-binding",
    "gpt-probe-linear-transport",
)


def differentiate(
    model: torch.nn.Module, stepper: Stepper, batch: torch.Tensor
) -> tuple[float, dict]:
    """Read the original first-batch loss and gradients without issuing an update."""
    model.zero_grad(set_to_none=True)
    loss = stepper.objective(batch)
    loss.backward()
    return float(loss.detach()), {
        n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
    }


def compare(first: dict, second: dict) -> dict:
    """Retain absolute/relative errors and exact inequality counts."""
    a = torch.cat([v.flatten() for v in first.values()]).double()
    b = torch.cat([v.flatten() for v in second.values()]).double()
    require(
        bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "Nonfinite probe"
    )
    return {
        "maximum_absolute_difference": float((a - b).abs().max()),
        "relative_l2_difference": float((a - b).norm() / a.norm().clamp_min(1e-300)),
        "unequal_elements": int(torch.count_nonzero(a != b)),
        "total_elements": a.numel(),
    }


def main() -> None:
    """Probe the same data and seed at both factories, preserving all run outputs."""
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    rows = []
    for task_row in [r for r in selected("short") if r["name"] in CASES]:
        case = specification(task_row)
        task = data.build(case, "cuda")
        require(
            task.fingerprints == task_row["data_fingerprints"], "Changed probe data"
        )
        models, metadata = {}, {}
        for name, cfg, factory in (
            ("quarter", Quarter(score_multiplier=0.25), quarter_factories),
            ("inverse", Configuration(), configured_factories),
        ):
            torch.manual_seed(0)
            models[name] = factory(cfg).model(task, case, "cuda")
            metadata[name] = parameter_metadata(models[name].model)
        require(metadata["quarter"] == metadata["inverse"], "Unpaired initialization")
        batch = task.inputs(torch.arange(case.batch, device="cuda"), static=True)
        row = {
            "case": case.name,
            "seed": 0,
            "head_width": case.width // case.heads,
            "batch_size": case.batch,
            "initialization": metadata["quarter"],
            "gain": {
                n: m.model.blocks[0].sa.attention_gain().flatten().tolist()
                for n, m in models.items()
            },
        }
        for compiled in (False, True):
            observations, repetitions = {}, {}
            for name, model in models.items():
                stepper = Stepper(model, task, case, "cuda", compiled=compiled)
                observations[name] = differentiate(model, stepper, batch)
                repetitions[name] = differentiate(model, stepper, batch)
                require(
                    not stepper.optimizer.state, "A probe issued an optimizer update"
                )
                require(
                    parameter_metadata(model.model) == metadata[name],
                    "Probe changed weights",
                )
            mode = "compiled" if compiled else "eager"
            row[mode] = {
                "losses": {n: value[0] for n, value in observations.items()},
                "parent_vs_candidate_gradients": compare(
                    observations["quarter"][1], observations["inverse"][1]
                ),
                "repeated_same_factory_gradients": {
                    n: compare(value[1], repetitions[n][1])
                    for n, value in observations.items()
                },
                "repeated_same_factory_loss_equal": {
                    n: value[0] == repetitions[n][0]
                    for n, value in observations.items()
                },
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del models
        torch.cuda.empty_cache()
    result = {
        "scope": (
            "Read-only first-batch seed-0 differentiation of all three width-16 "
            "screen controls. No optimizer updates or benchmark reruns; no causal "
            "attribution of a whole learning trajectory from this local probe."
        ),
        "rows": rows,
        "source_sha256": sha256(Path(__file__)),
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(),
    }
    destination = Path("research/early99_dimension_gain/results/width16-probe.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
