"""Preserve all applicable original stress cases for the complete new architecture."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import torch

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.rms_norm import RMSNorm
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)
from research.early99.experiments.scalable_softmax import install_scalable_softmax
from research.early99_dimension_gain.factories import install_inverse_width
from research.transformer_stress.common import (
    EPS,
    ROOT,
    digest,
    finite,
    relative_error,
    require,
    source_manifest,
    write_json,
)

PLAN = Path(__file__).with_name("plan.json")
OUTPUT = ROOT / "runs/transformer-stress/inverse-width-stack-v1"
RESULTS = PLAN.parent / "results"
CANDIDATE_PLAN = (
    ROOT
    / "runs/early99/candidates/peri-rms-scaled-ssmax-inverse-width-lr2"
    / "plan.json"
)


def read(path: Path) -> dict:
    """Read a preserved JSON record without importing unrelated experiments."""
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    """Bind each source and record to its exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervene(model: GPT, name: str, magnitude: float) -> None:
    """Apply the original non-frequency interventions without changing controls."""
    require(
        name in {"control", "embedding", "qkv", "swiglu"}, "Unsupported intervention"
    )
    with torch.no_grad():
        if name == "embedding":
            model.token_embedding_table.weight.mul_(magnitude)
        for block in model.blocks:
            if name == "qkv":
                block.sa.c_attn.weight.mul_(magnitude)
            elif name == "swiglu":
                block.ffwd.c_fc.weight.mul_(magnitude)


def model_for(settings: dict, layers: int, context: int) -> GPT:
    """Use production operations and the same original stress initializer."""
    model = GPT(
        settings["full_stack_vocab"],
        settings["full_stack_width"],
        block_size=context,
        n_head=settings["full_stack_heads"],
        n_layer=layers,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    )
    install_peri_rms(model)
    install_embedding_rms(model)
    install_residual_scaling(model)
    install_scalable_softmax(model)
    old = digest(dict(model.named_parameters()))
    rng = torch.get_rng_state().clone()
    install_inverse_width(model)
    require(old == digest(dict(model.named_parameters())), "Wrapper changed parameters")
    require(torch.equal(rng, torch.get_rng_state()), "Wrapper consumed RNG")
    require(
        model.token_embedding_table.weight is model.lm_head.weight, "Tied weight lost"
    )
    for block in model.blocks:
        require(block.sa.use_qk_norm, "Q/K RMS missing")
        require(
            torch.equal(
                block.sa.attention_gain(),
                torch.full_like(block.sa.attention_gain(), 1 / block.sa.head_dim),
            ),
            "Wrong gain",
        )
    for module in model.modules():
        if isinstance(module, RMSNorm):
            require(not list(module.parameters()), "RMS gained parameters")
            module.eps = EPS
    return model


def gradients(actual: GPT, expected: GPT) -> tuple[list[dict], float, bool]:
    """Compare every named parameter; missing gradients remain explicit failures."""
    first, second = dict(actual.named_parameters()), dict(expected.named_parameters())
    require(list(first) == list(second), "Reference parameter names/order differ")
    records, error_sq, ref_sq = [], 0.0, 0.0
    all_finite = True
    for name, parameter in first.items():
        value, reference = parameter.grad, second[name].grad
        present = value is not None and reference is not None
        valid = present and finite(value) and finite(reference)
        all_finite = all_finite and valid
        if present:
            difference = value.double() - reference.double()
            error_sq += difference.square().sum().item()
            ref_sq += reference.double().square().sum().item()
        records.append(
            {
                "name": name,
                "both_gradients_present": present,
                "actual_finite": value is not None and finite(value),
                "reference_finite": reference is not None and finite(reference),
                "relative_l2_error": relative_error(value, reference)
                if valid
                else None,
            }
        )
    ratio = (
        math.sqrt(error_sq) / max(math.sqrt(ref_sq), 1e-30) if all_finite else math.inf
    )
    return records, ratio, all_finite


def cell(
    settings: dict,
    plan: dict,
    layers: int,
    context: int,
    intervention: str,
    magnitude: float,
) -> dict:
    """Apply no updates; compare full logits, CE and every parameter gradient."""
    torch.manual_seed(settings["seed"])
    actual = model_for(settings, layers, context)
    initial = digest(dict(actual.named_parameters()))
    intervene(actual, intervention, magnitude)
    quantized = digest(dict(actual.named_parameters()))
    reference = copy.deepcopy(actual).double()
    require(
        reference.token_embedding_table.weight is reference.lm_head.weight,
        "Reference lost tying",
    )
    tokens = torch.randint(settings["full_stack_vocab"], (1, context))
    targets = (tokens + 1) % settings["full_stack_vocab"]
    logits, loss = actual(tokens, targets)
    loss.backward()
    expected, ref_loss = reference(tokens, targets)
    ref_loss.backward()
    grad_rows, grad_error, grad_finite = gradients(actual, reference)
    forward_finite = finite(logits) and finite(loss)
    reference_finite = finite(expected) and finite(ref_loss)
    logit_error = relative_error(logits, expected)
    loss_error = abs(loss.item() - ref_loss.item()) / max(abs(ref_loss.item()), 1e-30)
    threshold = plan["relative_error_threshold"]
    forward_pass = forward_finite and reference_finite and logit_error <= threshold
    passed = (
        forward_pass
        and grad_finite
        and grad_error <= threshold
        and loss_error <= threshold
    )
    require(
        quantized == digest(dict(actual.named_parameters())),
        "A numerical probe updated weights",
    )
    return {
        "layers": layers,
        "context": context,
        "intervention": intervention,
        "magnitude": magnitude,
        "head_width": settings["full_stack_width"] // settings["full_stack_heads"],
        "parameters": sum(p.numel() for p in actual.parameters()),
        "initial_parameters_sha256": initial,
        "quantized_parameters_sha256": quantized,
        "tokens_targets_sha256": digest({"tokens": tokens, "targets": targets}),
        "tied_parameter_identity": actual.token_embedding_table.weight
        is actual.lm_head.weight,
        "all_parameters_fp32": all(
            p.dtype == torch.float32 for p in actual.parameters()
        ),
        "logits_finite": finite(logits),
        "loss_finite": finite(loss),
        "gradients_finite": grad_finite,
        "reference_finite": reference_finite,
        "loss": loss.item(),
        "reference_loss": ref_loss.item(),
        "logits_relative_error": logit_error,
        "loss_relative_error": loss_error,
        "global_gradient_relative_error": grad_error,
        "gradient_comparisons": grad_rows,
        "forward_pass": forward_pass,
        "passed": passed,
    }


def prepare(plan: dict) -> dict:
    """Bind all frozen production dependencies and the committed diagnostic."""
    for name, expected in plan["evidence_sha256"].items():
        require(sha256(ROOT / name) == expected, "Declared evidence changed")
    scientific = read(CANDIDATE_PLAN)
    sources = dict(scientific["source_sha256"])
    for name, expected in sources.items():
        require(
            sha256(ROOT / name) == expected, f"Frozen candidate source changed: {name}"
        )
    sources.update(source_manifest())
    paths = [
        *PLAN.parent.glob("*.py"),
        PLAN,
    ]
    for p in paths:
        relative = str(p.relative_to(ROOT))
        committed = subprocess.run(  # noqa: S603 -- Inspect exact local committed bytes.
            ["/usr/bin/git", "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        require(committed == p.read_bytes(), "Commit diagnostic before execution")
        sources[relative] = sha256(p)
    record = {
        "plan": plan,
        "source_sha256": sources,
        "torch": torch.__version__,
        "candidate_plan_sha256": sha256(CANDIDATE_PLAN),
    }
    manifest = OUTPUT / "manifest.json"
    if manifest.exists():
        require(read(manifest) == record, "Existing diagnostic changed")
        for name, expected in sources.items():
            require(sha256(OUTPUT / "sources" / name) == expected, "Archive changed")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        for name in sources:
            target = OUTPUT / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        write_json(manifest, record)
    return record


def run() -> None:
    """Keep all 50 outcomes and mark unsupported learned-frequency cases explicitly."""
    plan = read(PLAN)
    prepare(plan)
    settings = read(ROOT / "research/transformer_stress/plan.json")["components"]
    rows, inapplicable = [], []
    for layers, context in settings["full_stack_shapes_layers_context"]:
        for intervention, magnitude in settings["full_stack_interventions"]:
            if intervention not in plan["interventions"]:
                inapplicable.append(
                    {
                        "layers": layers,
                        "context": context,
                        "intervention": intervention,
                        "magnitude": magnitude,
                        "reason": "Fixed RoPE has no learned-frequency parameter",
                    }
                )
                continue
            path = OUTPUT / f"L{layers}-N{context}-{intervention}-{magnitude}.json"
            if path.exists():
                row = read(path)
            else:
                row = cell(settings, plan, layers, context, intervention, magnitude)
                write_json(path, row)
                row = read(path)
            rows.append(row)
            print(
                {
                    k: row[k]
                    for k in (
                        "layers",
                        "context",
                        "intervention",
                        "magnitude",
                        "forward_pass",
                        "passed",
                    )
                },
                flush=True,
            )
    require(len(rows) == plan["expected_cases"], "Missing applicable cell")
    require(
        len(inapplicable) == plan["inapplicable_original_cases"],
        "Wrong inapplicable cases",
    )
    summary = {
        "expected": plan["expected_cases"],
        "completed": len(rows),
        "forward_passes": sum(r["forward_pass"] for r in rows),
        "full_passes": sum(r["passed"] for r in rows),
        "nonfinite_actual_cells": sum(
            not (r["logits_finite"] and r["loss_finite"] and r["gradients_finite"])
            for r in rows
        ),
        "ordinary_controls": [
            {
                k: r[k]
                for k in (
                    "layers",
                    "context",
                    "forward_pass",
                    "passed",
                    "logits_relative_error",
                    "global_gradient_relative_error",
                )
            }
            for r in rows
            if r["intervention"] == "control"
        ],
        "inapplicable_original_cells": inapplicable,
        "scope": plan["scope"],
        "cells": rows,
    }
    write_json(OUTPUT / "summary.json", summary)
    RESULTS.mkdir(exist_ok=True)
    for name in ("manifest.json", "summary.json"):
        shutil.copyfile(OUTPUT / name, RESULTS / name)
    print(
        {
            k: v
            for k, v in summary.items()
            if k not in {"cells", "inapplicable_original_cells"}
        }
    )


def main() -> None:
    """Use one CPU thread and preserve the active GPU training experiment."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    with OUTPUT.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()


if __name__ == "__main__":
    main()
