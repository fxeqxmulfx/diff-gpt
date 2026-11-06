"""Nonmutating paired forward, gradient-path and component measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.rms_norm import RMSNorm
from research.current_grokking.data import tensor_digest
from research.early99_reporting.round2_audit import require
from research.transformer_path_diagnostic.metrics import (
    cosine,
    decomposition,
    norm,
    probe_like,
    ratio,
    stats,
)
from research.transformer_path_diagnostic.trace import Trace

if TYPE_CHECKING:
    from research.current_grokking.interfaces import Task
    from research.current_grokking.train import Logits

SATURATION_LOW = 0.01
SATURATION_HIGH = 0.99


def state_fingerprints(model: torch.nn.Module) -> dict:
    """Include persistent and nonpersistent buffers, with deduplicated parameters."""
    return {
        kind: {name: tensor_digest(value) for name, value in members}
        for kind, members in [
            ("parameters", model.named_parameters()),
            ("buffers", model.named_buffers()),
        ]
    }


def parameter_metrics(model: torch.nn.Module, gradients: dict) -> dict:
    """Retain every unique parameter and the fused Q/K/V or gate/value partitions."""
    result = {}
    for name, parameter in model.named_parameters():
        pieces = [(name, parameter, gradients[name])]
        labels = (
            ["q", "k", "v"]
            if name.endswith("c_attn.weight")
            else ["gate", "value"]
            if name.endswith("c_fc.weight")
            else []
        )
        if labels:
            pieces += [
                (name + "." + label, value, gradient)
                for label, value, gradient in zip(
                    labels,
                    parameter.chunk(len(labels)),
                    gradients[name].chunk(len(labels)),
                    strict=True,
                )
            ]
        for label, value, gradient in pieces:
            result[label] = {
                "weight": stats(value),
                "gradient": stats(gradient),
                "gradient_to_weight_l2": ratio(norm(gradient), norm(value)),
            }
    return result


def branch_paths(trace: Trace, objective: str, layers: int) -> list:
    """Resolve the actual constant-prefix guards and each local branch VJP."""
    rows = []
    gradients = trace.gradients[objective]
    for layer in range(layers):
        base = f"blocks.{layer}"
        guard = trace.guards[base][..., None]
        for branch, before, after, output in [
            ("attention", base + ".input", base + ".ln_2.input", base + ".sa.output"),
            (
                "feedforward",
                base + ".ln_2.input",
                base + ".output",
                base + ".ffwd.output",
            ),
        ]:
            raw = trace.live[output]
            applied = torch.where(guard, raw[:, :1], raw)
            require(
                torch.equal(
                    trace.values[before] + applied.detach(), trace.values[after]
                ),
                f"Residual forward reconstruction failed: {base}/{branch}",
            )
            identity = gradients[after]
            contribution = torch.autograd.grad(
                applied, trace.live[before], grad_outputs=identity, retain_graph=True
            )[0]
            rows.append(
                {
                    "layer": layer,
                    "branch": branch,
                    "skip_rms": stats(trace.values[before])["rms"],
                    "branch_rms": stats(applied)["rms"],
                    "branch_to_skip_rms": ratio(
                        norm(applied), norm(trace.values[before])
                    ),
                    "activation_cosine": cosine(applied, trace.values[before]),
                    "gradient_paths": decomposition(
                        identity, contribution, gradients[before]
                    ),
                    "first_gradient_path": "identity",
                    "second_gradient_path": "branch",
                    "guarded_positions": int(guard.sum()),
                }
            )
    return rows


def tying_paths(trace: Trace, gradients: dict, objective: str) -> dict:
    """Separate lookup and readout use of the same weight, without an optimizer step."""
    model = trace.model
    weight = model.token_embedding_table.weight
    require(weight is model.lm_head.weight, "Embedding/readout weights are not tied")
    g = trace.gradients[objective]
    lookup = torch.autograd.grad(
        trace.live["token_embedding_table.lookup.output"],
        weight,
        grad_outputs=g["token_embedding_table.lookup.output"],
        retain_graph=True,
    )[0]
    # Detach the hidden state to isolate only the direct readout use of the weight.
    local_readout = F.linear(trace.live["lm_head.input"].detach(), weight)
    readout = torch.autograd.grad(
        local_readout, weight, grad_outputs=g["lm_head.output"]
    )[0]
    return {
        **decomposition(lookup, readout, gradients["token_embedding_table.weight"]),
        "first_gradient_path": "embedding_lookup",
        "second_gradient_path": "direct_readout",
    }


def rms_paths(trace: Trace, objective: str) -> dict:
    """Measure local RMS pullbacks separately from gradients of shared inputs."""
    rows = {}
    gradients = trace.gradients[objective]
    for name, module in trace.modules.items():
        if not isinstance(module, RMSNorm) or name + ".output" not in gradients:
            continue
        incoming = gradients[name + ".output"]
        outgoing = torch.autograd.grad(
            trace.live[name + ".output"],
            trace.live[name + ".input"],
            grad_outputs=incoming,
            retain_graph=True,
        )[0]
        rows[name] = {
            "output_cotangent_l2": norm(incoming),
            "local_input_cotangent_l2": norm(outgoing),
            "local_vjp_gain": ratio(norm(outgoing), norm(incoming)),
            "input_rms": stats(trace.values[name + ".input"])["rms"],
            "epsilon": module.eps
            if module.eps is not None
            else torch.finfo(trace.live[name + ".input"].dtype).eps,
        }
    return rows


def attention_metrics(trace: Trace, layers: int) -> list:
    """Summarize causal rows per head, including explicit singleton accounting."""
    rows = []
    for layer in range(layers):
        base = f"blocks.{layer}.sa"
        probabilities = trace.values[base + ".softmax.output"].double()
        scores = trace.values[base + ".softmax.input0"].double()
        counts = torch.arange(1, probabilities.shape[-2] + 1).double()
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        for head in range(probabilities.shape[1]):
            p, s, h = probabilities[:, head], scores[:, head], entropy[:, head]
            spread = (
                s.max(-1).values
                - s.masked_fill(~torch.isfinite(s), torch.inf).min(-1).values
            )
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "singleton_rows": p.shape[0],
                    "nonsingleton_rows": p.shape[0] * (p.shape[1] - 1),
                    "normalized_entropy_mean_nonsingleton": (
                        h[:, 1:] / counts[1:].log()
                    )
                    .mean()
                    .item(),
                    "max_probability_mean_nonsingleton": p[:, 1:]
                    .max(-1)
                    .values.mean()
                    .item(),
                    "max_probability_above_099_fraction_nonsingleton": (
                        p[:, 1:].max(-1).values > SATURATION_HIGH
                    )
                    .double()
                    .mean()
                    .item(),
                    "score_spread_mean_nonsingleton": spread[:, 1:].mean().item(),
                    "effective_visible_keys_mean_nonsingleton": h[:, 1:]
                    .exp()
                    .mean()
                    .item(),
                    "row_sum_max_absolute_error": (p.sum(-1) - 1).abs().max().item(),
                }
            )
    return rows


def nonlinear_metrics(trace: Trace, layers: int) -> list:
    """Report XSA retention/orthogonality and the actual SwiGLU gate statistics."""
    rows = []
    for layer in range(layers):
        base = f"blocks.{layer}"
        own = trace.values[base + ".sa.xsa.input2"].double()
        attended = trace.values[base + ".sa.xsa.input0"].double() + own[:, :, :1]
        projected = trace.values[base + ".sa.xsa.output"].double()
        denominator = own.norm(dim=-1) * projected.norm(dim=-1)
        active = denominator > 0
        cosines = (own * projected).sum(-1)[active] / denominator[active]
        gate, value = trace.values[base + ".ffwd.c_fc.output"].chunk(2, dim=-1)
        sigmoid = gate.double().sigmoid()
        rows.append(
            {
                "layer": layer,
                "xsa_projected_to_attended_l2": ratio(norm(projected), norm(attended)),
                "xsa_nonzero_pairs": int(active.sum()),
                "xsa_own_value_abs_cosine_max": cosines.abs().max().item()
                if cosines.numel()
                else None,
                "swiglu_gate": stats(gate),
                "swiglu_value": stats(value),
                "swiglu_product": stats(trace.values[base + ".ffwd.c_proj.input"]),
                "swiglu_sigmoid_below_001_fraction": (sigmoid < SATURATION_LOW)
                .double()
                .mean()
                .item(),
                "swiglu_sigmoid_above_099_fraction": (sigmoid > SATURATION_HIGH)
                .double()
                .mean()
                .item(),
            }
        )
    return rows


def observe(model: Logits, task: Task, batch: torch.Tensor, probe_seed: int) -> dict:
    """Verify exact observer neutrality for both objectives and retain every site."""
    model.eval()
    gpt = model.model
    before, rng = state_fingerprints(gpt), torch.get_rng_state().clone()
    expected, targets = task.forward(model, batch)
    probe = probe_like(expected, probe_seed)
    objectives = {
        "task_loss": task.loss(expected, targets),
        "unit_logit_probe": (expected * probe).sum(),
    }
    reference = {}
    for name, objective in objectives.items():
        model.zero_grad(set_to_none=True)
        objective.backward(retain_graph=True)
        reference[name] = {
            name: p.grad.detach().clone() for name, p in gpt.named_parameters()
        }
    output = {
        "reference_logits_sha256": tensor_digest(expected),
        "targets_sha256": tensor_digest(targets),
        "probe_sha256": tensor_digest(probe),
        "probe_l2": norm(probe),
    }
    with Trace(gpt) as trace:
        actual, actual_targets = task.forward(model, batch)
        require(
            torch.equal(actual, expected) and torch.equal(actual_targets, targets),
            "Instrumentation changed forward values or targets",
        )
        measured = {
            "task_loss": task.loss(actual, targets),
            "unit_logit_probe": (actual * probe).sum(),
        }
        output["objectives"] = {}
        for name, objective in measured.items():
            model.zero_grad(set_to_none=True)
            trace.objective = name
            objective.backward(retain_graph=True)
            trace.objective = None
            parameter_gradients = {
                key: p.grad.detach().clone() for key, p in gpt.named_parameters()
            }
            require(
                torch.equal(objective, objectives[name])
                and all(
                    torch.equal(value, reference[name][key])
                    for key, value in parameter_gradients.items()
                ),
                f"Instrumentation changed objective/parameter gradients: {name}",
            )
            g = trace.gradients[name]
            output["objectives"][name] = {
                "value": objective.item(),
                "sites": {
                    site: {
                        "activation": stats(value),
                        "gradient": stats(g[site]) if site in g else None,
                    }
                    for site, value in trace.values.items()
                },
                "parameters": parameter_metrics(gpt, parameter_gradients),
                "residual_paths": branch_paths(trace, name, gpt.n_layer),
                "tied_weight_paths": tying_paths(trace, parameter_gradients, name),
                "rms_local_paths": rms_paths(trace, name),
            }
        output["attention"] = attention_metrics(trace, gpt.n_layer)
        output["nonlinear"] = nonlinear_metrics(trace, gpt.n_layer)
        output["executed_sites"] = sorted(trace.values)
    model.zero_grad(set_to_none=True)
    require(
        before == state_fingerprints(gpt) and torch.equal(rng, torch.get_rng_state()),
        "Diagnostic mutated parameters, buffers or global RNG",
    )
    output["exact_forward_and_all_parameter_gradient_parity"] = True
    output["unchanged_model_and_rng"] = True
    return output
