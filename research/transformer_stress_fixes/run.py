"""Run separately frozen numerical fixes and the single-change learning ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

from research.transformer_stress.common import (
    ROOT,
    finite,
    full_stack,
    gradient_norm,
    relative_error,
    require,
    source_manifest,
    write_json,
)
from research.transformer_stress.run import PLAN as PARENT_PLAN
from research.transformer_stress.run import report
from research.transformer_stress_fixes.learning import train_shifted_cell
from research.transformer_stress_fixes.models import fixed_model

PLAN = Path(__file__).with_name("plan.json")


def intervene(model: torch.nn.Module, intervention: str, magnitude: float) -> None:
    """Apply exactly the original fixed interventions to each paired model."""
    with torch.no_grad():
        if intervention == "embedding":
            model.token_embedding_table.weight.mul_(magnitude)
        elif intervention == "frequency":
            model.lerope.log_scales.fill_(magnitude)
        else:
            for block in model.blocks:
                if intervention == "qkv":
                    block.sa.c_attn.weight.mul_(magnitude)
                elif intervention == "swiglu":
                    block.ffwd.c_fc.weight.mul_(magnitude)


def numeric_cell(
    settings: dict,
    layers: int,
    context: int,
    intervention: str,
    magnitude: float,
    variant: str,
) -> dict:
    """Compare to FP64 original equations with the intended unrounded angle map."""
    torch.manual_seed(settings["seed"])
    original = full_stack(settings, layers, context)
    original_state = copy.deepcopy(original.state_dict())
    torch.manual_seed(settings["seed"])
    model = fixed_model(settings, layers, context, variant)
    require(
        all(
            torch.equal(value, model.state_dict()[key])
            for key, value in original_state.items()
        ),
        "A numerical fix changed original initial parameters",
    )
    intervene(original, intervention, magnitude)
    intervene(model, intervention, magnitude)
    reference = original.double()
    # Unlike v1's promoted rounded table, this reference constructs the intended
    # angle in FP64. Evaluate the unchanged parent against this same reference.
    positions = torch.arange(context, dtype=torch.float64)
    head_dim = settings["full_stack_width"] // settings["full_stack_heads"]
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
    reference.lerope.base_angles = torch.outer(
        positions * positions.log1p(), 10000**-exponents
    )
    if variant == "all_numeric_bounded_rope":
        with torch.no_grad():
            reference.lerope.log_scales.copy_(
                4 * (reference.lerope.log_scales / 4).tanh()
            )
    tokens = torch.randint(settings["full_stack_vocab"], (1, context))
    targets = (tokens + 1) % settings["full_stack_vocab"]
    logits, loss = model(tokens, targets)
    loss.backward()
    with torch.no_grad():
        expected = reference(tokens)[0]
    grad = gradient_norm(model)
    return {
        "variant": variant,
        "bounded_rotary_log_frequency": 4
        if variant == "all_numeric_bounded_rope"
        else None,
        "layers": layers,
        "context": context,
        "intervention": intervention,
        "magnitude": magnitude,
        "parameters": sum(p.numel() for p in model.parameters()),
        "all_parameters_fp32": all(
            p.dtype == torch.float32 for p in model.parameters()
        ),
        "tied_parameter_identity": model.token_embedding_table.weight
        is model.lm_head.weight,
        "logits_finite": finite(logits),
        "loss_finite": finite(loss),
        "gradients_finite": math.isfinite(grad),
        "loss": loss.item(),
        "gradient_norm": grad,
        "maximum_logit": logits.detach().abs().max().item(),
        "reference_finite": finite(expected),
        "logits_relative_error": relative_error(logits, expected),
    }


def main() -> None:
    """Preserve source/data provenance and every original failure in a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/transformer-stress/fixes-v1"
    )
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--learning", action="store_true")
    args = parser.parse_args()
    plan, parent = json.loads(PLAN.read_text()), json.loads(PARENT_PLAN.read_text())
    torch.set_num_threads(parent["threads"])
    torch.set_float32_matmul_precision("highest")
    manifest = source_manifest()
    for path in [*Path(__file__).parent.glob("*.py"), PLAN]:
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    output = args.output
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        require(
            json.loads(manifest_path.read_text())["source_hashes"] == manifest,
            "Fix sources changed after freezing",
        )
    else:
        output.mkdir(parents=True, exist_ok=True)
        for name in manifest:
            destination = output / "sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        write_json(
            manifest_path,
            {
                "source_hashes": manifest,
                "plan": plan,
                "parent_plan": parent,
                "torch": torch.__version__,
                "device": "cpu",
            },
        )
        for name in ("data.pt", "data.json"):
            shutil.copyfile(ROOT / "runs/transformer-stress/v1" / name, output / name)
    if args.numerical:
        require(
            not (output / "numerical.json").exists(),
            "Do not overwrite numerical results",
        )
        rows = []
        for variant in ["control", *plan["numerical_variants"]]:
            for layers, context in parent["components"][
                "full_stack_shapes_layers_context"
            ]:
                for intervention, magnitude in parent["components"][
                    "full_stack_interventions"
                ]:
                    rows.append(
                        numeric_cell(
                            parent["components"],
                            layers,
                            context,
                            intervention,
                            magnitude,
                            variant,
                        )
                    )
            write_json(output / "numerical-progress.json", {"rows": rows})
            print(json.dumps({"numerical_variant_complete": variant}), flush=True)
        write_json(
            output / "numerical.json",
            {
                "scope": plan["numerical_scope"],
                "reference_note": (
                    "FP64 original equations with unrounded FP64 base angle. "
                    "Parent recomputed under this same reference; v1's separate "
                    "rounded-base reference remains preserved. The bounded arm "
                    "uses its explicitly declared 4*tanh(alpha/4) equation."
                ),
                "rows": rows,
            },
        )
    if args.learning:
        settings = parent["residual_learning"]
        splits = torch.load(output / "data.pt", weights_only=True)
        for depth in settings["branches"]:
            for strength in settings["teacher_strengths"]:
                for seed in settings["seeds"]:
                    for kind in settings["variants"]:
                        directory = (
                            output
                            / "learning"
                            / f"{kind}-d{depth}-a{strength}-seed{seed}"
                        )
                        if (directory / "result.json").exists():
                            continue
                        row = train_shifted_cell(
                            settings, splits, kind, depth, strength, seed, directory
                        )
                        print(
                            json.dumps(
                                {
                                    key: row[key]
                                    for key in (
                                        "variant",
                                        "branches",
                                        "strength",
                                        "seed",
                                        "status",
                                        "first_threshold_updates",
                                        "final_heldout_relative_mse",
                                    )
                                }
                            ),
                            flush=True,
                        )
                        report(output, parent)
                        summary_path = output / "learning-summary.json"
                        summary = json.loads(summary_path.read_text())
                        summary["scope"] = plan["learning_scope"]
                        summary["feedforward"] = plan["learning_variant"]
                        write_json(summary_path, summary)


if __name__ == "__main__":
    main()
