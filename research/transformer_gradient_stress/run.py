"""Compare corrected bounded-frequency gradients to high-precision references."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch

from research.transformer_gradient_stress.frequency import (
    StableFrequencies,
    decimal_reference,
    stable_multiplier,
)
from research.transformer_stress.common import (
    ROOT,
    finite,
    relative_error,
    require,
    source_manifest,
    write_json,
)
from research.transformer_stress.run import PLAN as ORIGINAL_PLAN
from research.transformer_stress_fixes.models import fixed_model
from research.transformer_stress_fixes.run import intervene

PLAN = Path(__file__).with_name("plan.json")
OUTPUT = ROOT / "runs/transformer-stress/gradient-v1"


def scalar_checks(plan: dict) -> list[dict]:
    """Measure the entire declared grid, including representable tiny derivatives."""
    records = []
    for dtype_name in plan["scalar_parameter_dtypes"]:
        dtype = getattr(torch, dtype_name)
        for raw in plan["scalar_parameters"]:
            parameter = torch.tensor(float(raw), dtype=dtype, requires_grad=True)
            wide = parameter.double()
            parent = (4 * (wide / 4).tanh()).exp()
            corrected = stable_multiplier(wide)
            old_gradient = torch.autograd.grad(parent, parameter, retain_graph=True)[0]
            new_gradient = torch.autograd.grad(corrected, parameter)[0]
            ref_value, ref_gradient = decimal_reference(raw)
            records.append(
                {
                    "raw_parameter": raw,
                    "parameter_dtype": dtype_name,
                    "forward_identical": bool(torch.equal(parent, corrected)),
                    "forward_relative_error": abs(corrected.item() - ref_value)
                    / ref_value,
                    "parent_gradient": old_gradient.item(),
                    "corrected_gradient": new_gradient.item(),
                    "reference_gradient": ref_gradient,
                    "parent_relative_gradient_error": abs(
                        old_gradient.item() - ref_gradient
                    )
                    / ref_gradient,
                    "corrected_relative_gradient_error": abs(
                        new_gradient.item() - ref_gradient
                    )
                    / ref_gradient,
                }
            )
    return records


def full_checks(settings: dict) -> list[dict]:
    """Check identical logits and frequency gradients in every full-stack case."""
    rows = []
    for layers, context in settings["full_stack_shapes_layers_context"]:
        for intervention, magnitude in settings["full_stack_interventions"]:
            torch.manual_seed(settings["seed"])
            parent = fixed_model(settings, layers, context, "all_numeric_bounded_rope")
            intervene(parent, intervention, magnitude)
            corrected = copy.deepcopy(parent)
            corrected.lerope = StableFrequencies(corrected.lerope)
            tokens = torch.randint(settings["full_stack_vocab"], (1, context))
            target = (tokens + 1) % settings["full_stack_vocab"]
            old_logits, old_loss = parent(tokens, target)
            new_logits, new_loss = corrected(tokens, target)
            old_loss.backward()
            new_loss.backward()
            old_parameters = dict(parent.named_parameters())
            differences = []
            for name, parameter in corrected.named_parameters():
                require(torch.equal(parameter, old_parameters[name]), "Weights changed")
                if name != "lerope.log_scales":
                    differences.append(
                        relative_error(parameter.grad, old_parameters[name].grad)
                    )
            old_frequency = parent.lerope.log_scales.grad
            new_frequency = corrected.lerope.log_scales.grad
            rows.append(
                {
                    "layers": layers,
                    "context": context,
                    "intervention": intervention,
                    "magnitude": magnitude,
                    "forward_identical": bool(torch.equal(old_logits, new_logits)),
                    "all_gradients_finite": all(
                        finite(p.grad) for p in corrected.parameters()
                    ),
                    "other_parameter_gradient_max_relative_error": max(differences),
                    "old_frequency_gradient_norm": old_frequency.double().norm().item(),
                    "new_frequency_gradient_norm": new_frequency.double().norm().item(),
                    "old_zero_frequency_entries": (old_frequency == 0).sum().item(),
                    "new_zero_frequency_entries": (new_frequency == 0).sum().item(),
                }
            )
    return rows


def main() -> None:
    """Freeze provenance and retain all scalar and whole-model numerical outcomes."""
    plan = json.loads(PLAN.read_text())
    original = json.loads(ORIGINAL_PLAN.read_text())
    torch.set_num_threads(original["threads"])
    torch.set_float32_matmul_precision("highest")
    require(not OUTPUT.exists(), "Do not overwrite the gradient experiment")
    OUTPUT.mkdir(parents=True)
    sources = source_manifest()
    for path in [
        *Path(__file__).parent.glob("*.py"),
        PLAN,
        *ROOT.joinpath("research/transformer_stress_fixes").glob("*.py"),
        ROOT / "research/transformer_stress_fixes/plan.json",
    ]:
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for name in sources:
        target = OUTPUT / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(
        OUTPUT / "manifest.json",
        {
            "source_hashes": sources,
            "plan": plan,
            "parent_component_settings": original["components"],
            "torch": torch.__version__,
            "device": "cpu",
        },
    )
    scalars = scalar_checks(plan)
    write_json(OUTPUT / "scalar.json", scalars)
    full = full_checks(original["components"])
    write_json(OUTPUT / "full-stack.json", full)
    summary = {
        "scalar_cases": len(scalars),
        "full_stack_cases": len(full),
        "all_forwards_identical": all(
            row["forward_identical"] for row in [*scalars, *full]
        ),
        "all_full_stack_gradients_finite": all(
            row["all_gradients_finite"] for row in full
        ),
        "maximum_corrected_scalar_gradient_relative_error": max(
            row["corrected_relative_gradient_error"] for row in scalars
        ),
        "maximum_parent_scalar_gradient_relative_error": max(
            row["parent_relative_gradient_error"] for row in scalars
        ),
        "maximum_other_parameter_gradient_relative_error": max(
            row["other_parameter_gradient_max_relative_error"] for row in full
        ),
        "full_stack_frequency_zero_recoveries": sum(
            row["old_frequency_gradient_norm"] == 0
            and row["new_frequency_gradient_norm"] > 0
            for row in full
        ),
        "scope": plan["scope"],
    }
    summary["all_declared_checks_pass"] = (
        summary["all_forwards_identical"]
        and summary["all_full_stack_gradients_finite"]
        and summary["maximum_corrected_scalar_gradient_relative_error"]
        <= plan["maximum_scalar_relative_gradient_error"]
        and summary["maximum_other_parameter_gradient_relative_error"]
        <= plan["maximum_other_parameter_relative_gradient_error"]
    )
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
