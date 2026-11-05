"""Stress the actual transformer operations against independent FP64 equations."""

from __future__ import annotations

import copy
import math

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.centered_xsa import XSA_EPS, centered_projection
from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.rms_norm import rms_norm
from research.early99.experiments.learned_rope import LearnedFrequencies
from research.transformer_stress.common import (
    EPS,
    finite,
    full_stack,
    gradient_norm,
    relative_error,
    rms,
)


def reference_rms(value: torch.Tensor) -> torch.Tensor:
    """Evaluate the finite-epsilon definition in FP64 on the actual FP32 input."""
    value = value.double()
    return value / (value.square().mean(-1, keepdim=True) + EPS).sqrt()


def normalization(settings: dict) -> list[dict]:
    """Check forward magnitude and radial/tangent gradients, including overflow."""
    records = []
    for dimension in settings["dimensions"]:
        base = torch.randn(3, dimension)
        base = base / base.square().mean(-1, keepdim=True).sqrt()
        for scale in settings["rms_scales"]:
            x = (base * scale).requires_grad_()
            output = rms_norm(x, EPS)
            reference_input = x.detach().double().requires_grad_()
            reference = reference_rms(reference_input)
            for direction in ("radial", "random"):
                probe = x.detach() if direction == "radial" else torch.randn_like(base)
                actual_grad = torch.autograd.grad(
                    (output * probe).sum(), x, retain_graph=True
                )[0]
                if direction == "radial":
                    # An exactly parallel upstream has this closed form. Native
                    # FP64 autograd can also cancel its tiny radial derivative.
                    ref_grad = (
                        EPS
                        * reference_input
                        / (reference_input.square().mean(-1, keepdim=True) + EPS).pow(
                            1.5
                        )
                    )
                else:
                    ref_grad = torch.autograd.grad(
                        (reference * probe.double()).sum(),
                        reference_input,
                        retain_graph=True,
                    )[0]
                records.append(
                    {
                        "component": "rms",
                        "dimension": dimension,
                        "scale": scale,
                        "direction": direction,
                        "output_finite": finite(output),
                        "gradient_finite": finite(actual_grad),
                        "output_rms": rms(output),
                        "reference_rms": rms(reference),
                        "forward_relative_error": relative_error(output, reference),
                        "gradient_relative_error": relative_error(
                            actual_grad, ref_grad
                        ),
                        "gradient_absolute_rms_error": rms(
                            actual_grad.double() - ref_grad
                        ),
                        "gradient_rms": rms(actual_grad),
                    }
                )
    return records


def attention_scaling(settings: dict) -> list[dict]:
    """Check SSMax at zero/positive gaps; the concentration condition is explicit."""
    records = []
    for length in settings["ssmax_lengths"]:
        for gap in settings["ssmax_score_gaps"]:
            logits = torch.zeros(length)
            logits[0] = gap
            weights = (logits * math.log(length)).softmax(-1)
            expected = 1 / (1 + (length - 1) * math.exp(-math.log(length) * gap))
            records.append(
                {
                    "component": "ssmax",
                    "length": length,
                    "score_gap_before_log_count": gap,
                    "target_probability": weights[0].item(),
                    "analytic_probability": expected,
                    "absolute_error": abs(weights[0].item() - expected),
                    "entropy": -(
                        weights.double() * weights.double().clamp_min(1e-300).log()
                    )
                    .sum()
                    .item(),
                    "finite": finite(weights),
                    "row_sum": weights.double().sum().item(),
                }
            )
    return records


def exclusive_attention(settings: dict) -> list[dict]:
    """Compare centered XSA to its original clipped equation on quantized values."""
    records = []
    direction = torch.randn(8)
    direction = direction / direction.norm()
    weights = torch.arange(1, 9).float().softmax(0)
    for scale in settings["xsa_scales"]:
        for pattern in ("constant", "nearly_parallel", "common_offset"):
            values = direction.expand(8, -1).clone() * scale
            if pattern == "nearly_parallel":
                values = values + torch.randn_like(values) * scale * 1e-6
            elif pattern == "common_offset":
                values = values + torch.randn_like(values)
            values.requires_grad_()
            own, anchor = values[-1], values[0]
            average = (weights[:, None] * (values - anchor)).sum(0)
            output = centered_projection(average, own - anchor, own)
            ref_values = values.detach().double().requires_grad_()
            ref_own = ref_values[-1]
            unit = ref_own / ref_own.norm().clamp_min(XSA_EPS)
            # FP64 softmax weights sum to one after renormalization; the actual
            # FP32 weights may differ slightly. Use normalized actual weights
            # to isolate the centered/projection calculation from that roundoff.
            ref_weights = weights.double() / weights.double().sum()
            ref_average = (ref_weights[:, None] * ref_values).sum(0)
            reference = ref_average - (ref_average * unit).sum() * unit
            probe = torch.arange(1, 9).float()
            gradient = torch.autograd.grad((output * probe).sum(), values)[0]
            ref_grad = torch.autograd.grad(
                (reference * probe.double()).sum(), ref_values
            )[0]
            records.append(
                {
                    "component": "xsa",
                    "scale": scale,
                    "pattern": pattern,
                    "output_finite": finite(output),
                    "gradient_finite": finite(gradient),
                    "output_rms": rms(output),
                    "reference_rms": rms(reference),
                    "forward_relative_error": relative_error(output, reference),
                    "forward_absolute_rms_error": rms(output.double() - reference),
                    "gradient_relative_error": relative_error(gradient, ref_grad),
                    "error_relative_to_values": rms(output.double() - reference)
                    / max(rms(values), 1e-30),
                    "own_dot_output": (own.detach().double() * output.detach().double())
                    .sum()
                    .item(),
                    "own_norm": own.detach().double().norm().item(),
                    "clipping_epsilon": XSA_EPS,
                }
            )
    return records


def swiglu(settings: dict) -> list[dict]:
    """Measure the unbounded intermediate products hidden behind branch RMS."""
    records = []
    original = FeedForward(16)
    x0 = torch.randn(4, 16)
    for scale in settings["swiglu_weight_scales"]:
        model = copy.deepcopy(original)
        with torch.no_grad():
            model.c_fc.weight.mul_(scale)
        x = x0.clone().requires_grad_()
        pre = rms_norm(x, EPS)
        gate, values = model.c_fc(pre).chunk(2, -1)
        product = F.silu(gate) * values
        raw = model(pre)
        output = rms_norm(raw, EPS) / 2
        reference_model = copy.deepcopy(model).double()
        ref_x = x.detach().double().requires_grad_()
        ref_raw = reference_model(reference_rms(ref_x))
        reference = reference_rms(ref_raw) / 2
        probe = torch.randn_like(output)
        grad = torch.autograd.grad((output * probe).sum(), x)[0]
        ref_grad = torch.autograd.grad((reference * probe.double()).sum(), ref_x)[0]
        records.append(
            {
                "component": "swiglu",
                "input_projection_scale": scale,
                "product_finite": finite(product),
                "product_rms": rms(product),
                "raw_finite": finite(raw),
                "raw_rms": rms(raw),
                "output_finite": finite(output),
                "output_rms": rms(output),
                "gradient_finite": finite(grad),
                "forward_relative_error": relative_error(output, reference),
                "gradient_relative_error": relative_error(grad, ref_grad),
            }
        )
    return records


def rotary(settings: dict) -> list[dict]:
    """Separate near-isometric rotations from accurate intended rotary phases."""
    records = []
    for log_positions in (False, True):
        module = LearnedFrequencies(
            8,
            max(settings["rotary_positions"]) + 1,
            torch.ones(()),
            log_positions=log_positions,
        )
        for log_scale in settings["rotary_log_scales"]:
            with torch.no_grad():
                module.log_scales.fill_(log_scale)
            for position in settings["rotary_positions"]:
                cos, sin = module(position, position + 1)
                frequencies = 10000 ** -(torch.arange(0, 8, 2).double() / 8)
                warped = position * math.log1p(position) if log_positions else position
                angles = warped * frequencies * math.exp(log_scale)
                ref_cos, ref_sin = angles.cos(), angles.sin()
                gradient = torch.autograd.grad((cos + sin).sum(), module.log_scales)[0]
                records.append(
                    {
                        "component": "rotary",
                        "position_map": "p_log_p_plus_1" if log_positions else "p",
                        "position": position,
                        "log_scale": log_scale,
                        "finite": finite(cos) and finite(sin),
                        "gradient_finite": finite(gradient),
                        "maximum_phase_vector_error": torch.sqrt(
                            (cos.double() - ref_cos).square()
                            + (sin.double() - ref_sin).square()
                        )
                        .max()
                        .item(),
                        "maximum_rotation_norm_error": (
                            cos.double().square() + sin.double().square() - 1
                        )
                        .abs()
                        .max()
                        .item(),
                        "maximum_frequency_gradient": gradient.detach()
                        .abs()
                        .max()
                        .item(),
                    }
                )
    return records


def tied_readout(settings: dict) -> list[dict]:
    """Check both gradient paths and distinguish embedding from readout scale."""
    records = []
    original = torch.randn(11, 8)
    tokens = torch.tensor([1, 4, 7])
    for scale in settings["tied_weight_scales"]:
        weight = (original * scale).requires_grad_()
        hidden = rms_norm(F.embedding(tokens, weight), EPS)
        logits = F.linear(hidden, weight)
        # Linear probe avoids overflow from squaring deliberately huge logits.
        probe = torch.linspace(-1, 1, logits.numel()).reshape_as(logits)
        gradient = torch.autograd.grad((logits * probe).sum(), weight)[0]
        lookup = weight.detach().clone().requires_grad_()
        readout = weight.detach().clone().requires_grad_()
        split_logits = F.linear(rms_norm(F.embedding(tokens, lookup), EPS), readout)
        g_lookup, g_readout = torch.autograd.grad(
            (split_logits * probe).sum(), (lookup, readout)
        )
        reference = F.linear(
            reference_rms(F.embedding(tokens, weight.detach().double())),
            weight.detach().double(),
        )
        records.append(
            {
                "component": "tied_readout",
                "weight_scale": scale,
                "embedding_rms": rms(hidden),
                "maximum_absolute_logit": logits.detach().abs().max().item(),
                "logits_finite": finite(logits),
                "gradient_finite": finite(gradient),
                "gradient_path_sum_relative_error": relative_error(
                    gradient, g_lookup + g_readout
                ),
                "readout_gradient_rms": rms(g_readout),
                "lookup_gradient_rms": rms(g_lookup),
                "forward_relative_error": relative_error(logits, reference),
            }
        )
    return records


def stream_mixing() -> list[dict]:
    """Test exact stochastic contraction and finite Sinkhorn, not full mHC."""
    records = []
    averaging = torch.full((2, 2), 0.5, dtype=torch.float64)
    disagreement = torch.tensor([1.0, -1.0], dtype=torch.float64)
    records.append(
        {
            "component": "mhc_exact_mixing",
            "spectral_norm": torch.linalg.matrix_norm(averaging, 2).item(),
            "disagreement_before": disagreement.norm().item(),
            "disagreement_after": (averaging @ disagreement).norm().item(),
            "mean_preserved": True,
            "scope": "Fixed skip mixing only, not the input-dependent block Jacobian",
        }
    )
    logits0 = torch.tensor([[0.0, math.log(2)], [0.0, 0.0]])
    for scale in (1, 10, 1000):
        for iterations in (1, 20):
            for kind in ("literal_exp", "log_domain"):
                logits = logits0 * scale
                if kind == "literal_exp":
                    matrix = logits.exp()
                    for _ in range(iterations):
                        matrix = matrix / matrix.sum(0, keepdim=True)
                        matrix = matrix / matrix.sum(1, keepdim=True)
                else:
                    for _ in range(iterations):
                        logits = logits - logits.logsumexp(0, keepdim=True)
                        logits = logits - logits.logsumexp(1, keepdim=True)
                    matrix = logits.exp()
                records.append(
                    {
                        "component": "mhc_finite_sinkhorn",
                        "implementation": kind,
                        "logit_scale": scale,
                        "iterations": iterations,
                        "finite": finite(matrix),
                        "row_sum_max_error": (matrix.double().sum(1) - 1)
                        .abs()
                        .max()
                        .item(),
                        "column_sum_max_error": (matrix.double().sum(0) - 1)
                        .abs()
                        .max()
                        .item(),
                        "spectral_norm": torch.linalg.matrix_norm(
                            matrix.double(), 2
                        ).item()
                        if finite(matrix)
                        else None,
                    }
                )
    return records


def full_models(settings: dict) -> list[dict]:
    """Probe the complete frozen architecture without changing its source files."""
    records = []
    for layers, context in settings["full_stack_shapes_layers_context"]:
        for intervention, magnitude in settings["full_stack_interventions"]:
            torch.manual_seed(settings["seed"])
            model = full_stack(settings, layers, context)
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
            reference_model = copy.deepcopy(model).double()
            tokens = torch.randint(settings["full_stack_vocab"], (1, context))
            targets = (tokens + 1) % settings["full_stack_vocab"]
            states, handles = [], []

            def record_state(
                _module: object,
                _args: object,
                output: object,
                *,
                destination: list = states,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                destination.append(
                    value.detach().double().square().mean(-1).sqrt().max().item()
                )

            handles.extend(
                block.register_forward_hook(record_state) for block in model.blocks
            )
            logits, loss = model(tokens, targets)
            loss.backward()
            with torch.no_grad():
                reference = reference_model(tokens)[0]
            for handle in handles:
                handle.remove()
            grad_norm = gradient_norm(model)
            records.append(
                {
                    "component": "full_stack",
                    "layers": layers,
                    "context": context,
                    "intervention": intervention,
                    "magnitude": magnitude,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "tied_parameter_identity": model.token_embedding_table.weight
                    is model.lm_head.weight,
                    "logits_finite": finite(logits),
                    "loss_finite": finite(loss),
                    "gradients_finite": math.isfinite(grad_norm),
                    "loss": loss.item(),
                    "gradient_norm": grad_norm,
                    "maximum_logit": logits.detach().abs().max().item(),
                    "maximum_hidden_row_rms": max(states),
                    "per_layer_maximum_hidden_row_rms": states,
                    "real_arithmetic_state_rms_bound": 1 + math.sqrt(2 * layers),
                    "fp64_reference_finite": finite(reference),
                    "logits_relative_error": relative_error(logits, reference),
                }
            )
    return records


def run_components(settings: dict) -> list[dict]:
    """Execute every declared component grid, retaining all failures as data."""
    result = []
    for function in (
        normalization,
        attention_scaling,
        exclusive_attention,
        swiglu,
        rotary,
        tied_readout,
        full_models,
    ):
        torch.manual_seed(settings["seed"])
        result.extend(function(settings))
    result.extend(stream_mixing())
    return result
