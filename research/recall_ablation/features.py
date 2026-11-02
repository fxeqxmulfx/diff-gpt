"""Fresh architectural candidates under the passing ordinary-AdamW recipe."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

from diff_gpt.domain.model.gpt import GPTMini
from diff_gpt.domain.model.mini_block import (
    ResidualScaledProjection,
    RMSNormEmbedding,
    RMSNormProjection,
    RMSQKAttention,
    fix_qk_temperature,
    initialize_swiglu,
    repair_rms_backward,
)

if TYPE_CHECKING:
    from torch import nn

ARCHITECTURE_STAGES = (
    "swiglu",
    "scaled-init",
    "embedding-norm",
    "peri-norm",
    "peri-norm-repaired",
    "rms-repaired",
    "residual-scale",
    "qk-rms",
    "fixed-temperature",
)
PERI_STAGES = {"peri-norm", "peri-norm-repaired"}
PREDECESSOR = {
    "swiglu": "baseline",
    "scaled-init": "swiglu",
    "embedding-norm": "swiglu",
    "peri-norm": "swiglu",
    "peri-norm-repaired": "swiglu",
    "rms-repaired": "swiglu",
    "residual-scale": "rms-repaired",
    "qk-rms": "rms-repaired",
    "fixed-temperature": "qk-rms",
}
QK_RMS_STAGES = {"qk-rms", "fixed-temperature"}


def raw_parameter_digest(model: nn.Module) -> str:
    """Hash parameter bytes in the historical report's order."""
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_digest(model: nn.Module, *, exclude_mlp: bool = False) -> str:
    """Hash ordered names, shapes and values; optionally omit replaced MLPs."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if exclude_mlp and ".ffwd." in name:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_model(stage: str, seed: int) -> tuple[GPTMini, dict[str, object]]:
    """Keep historical shared weights and replace only the declared feature."""
    torch.manual_seed(seed)
    model = GPTMini(
        vocab_size=548,
        n_embd=128,
        n_layer=6,
        n_head=4,
        block_size=64,
        pad_vocab_size_to=1,
        use_checkpoint=False,
    )
    historical = raw_parameter_digest(model)
    shared = parameter_digest(model, exclude_mlp=True)
    if stage in ARCHITECTURE_STAGES:
        # Construct all replacements before drawing their Normal(0,.02)
        # weights. Historical shared tensors are already initialized and
        # remain byte-identical; no trained endpoint is used to initialize.
        initialize_swiglu(model.blocks, model.n_embd)
    elif stage not in {"baseline", "schedule-free"}:
        msg = f"unknown architectural feature {stage}"
        raise ValueError(msg)
    if parameter_digest(model, exclude_mlp=True) != shared:
        msg = "an MLP change modified shared historical weights"
        raise RuntimeError(msg)
    parent_initial = historical if stage == "swiglu" else raw_parameter_digest(model)
    features = ["SwiGLU"] if stage in ARCHITECTURE_STAGES else []
    scale = 1.0
    if stage == "scaled-init":
        scale = (2 * model.n_layer) ** -0.5
        with torch.no_grad():
            for block in model.blocks:
                block.sa.c_proj.weight.mul_(scale)
                block.ffwd.c_proj.weight.mul_(scale)
        features.append("depth-scaled projection initialization")
    if stage == "embedding-norm":
        model.token_embedding_table = RMSNormEmbedding(model.token_embedding_table)
        features.append("embedding RMS normalization")
        if raw_parameter_digest(model) != parent_initial:
            msg = "embedding normalization changed parameter values or order"
            raise RuntimeError(msg)
    if stage in PERI_STAGES:
        for block in model.blocks:
            block.sa.c_proj = RMSNormProjection(block.sa.c_proj)
            block.ffwd.c_proj = RMSNormProjection(block.ffwd.c_proj)
        features.append("Peri-RMSNorm branch outputs")
        if raw_parameter_digest(model) != parent_initial:
            msg = "Peri-RMSNorm changed parameter values or order"
            raise RuntimeError(msg)
    repaired_sites = 0
    if stage in {
        "peri-norm-repaired",
        "rms-repaired",
        "residual-scale",
        *QK_RMS_STAGES,
    }:
        repaired_sites = repair_rms_backward(model, model.n_embd)
        features.append("repaired FP64 radial/tangent RMS backward")
        if raw_parameter_digest(model) != parent_initial:
            msg = "RMS backward repair changed the initial parameters"
            raise RuntimeError(msg)
    if stage in QK_RMS_STAGES:
        for block in model.blocks:
            block.sa = RMSQKAttention(block.sa)
        repaired_sites += 2 * model.n_layer
        features.append("per-head QK RMS normalization with learned temperature")
        if raw_parameter_digest(model) != parent_initial:
            msg = "QK RMS normalization changed parameter values or order"
            raise RuntimeError(msg)
    removed_parameters = []
    if stage == "fixed-temperature":
        for index, block in enumerate(model.blocks):
            fix_qk_temperature(block.sa)
            removed_parameters.append(f"blocks.{index}.sa.log_alpha")
        features[-1] = "per-head QK RMS normalization with fixed temperature"
    residual_scale = 1.0
    if stage == "residual-scale":
        residual_scale = (2 * model.n_layer) ** -0.5
        for block in model.blocks:
            block.sa.c_proj = ResidualScaledProjection(block.sa.c_proj, residual_scale)
            block.ffwd.c_proj = ResidualScaledProjection(
                block.ffwd.c_proj, residual_scale
            )
        features.append("fixed forward residual scaling")
        if raw_parameter_digest(model) != parent_initial:
            msg = "forward residual scaling changed the initial parameters"
            raise RuntimeError(msg)
    if model.token_embedding_table.weight is not model.lm_head.weight:
        msg = "the architectural candidate broke embedding/readout weight tying"
        raise RuntimeError(msg)
    return model, {
        "features": features,
        "parent_stage": PREDECESSOR.get(stage),
        "parent_initial_parameters_sha256": parent_initial,
        "historical_initial_parameters_sha256": historical,
        "shared_non_mlp_initial_sha256": shared,
        "initial_projection_scale": scale,
        "embedding_normalization": (
            "native parameter-free RMSNorm, dtype epsilon, native backward"
            if stage == "embedding-norm"
            else None
        ),
        "branch_output_normalization": (
            "native parameter-free RMSNorm after attention and MLP output projections"
            if stage in PERI_STAGES
            else None
        ),
        "rms_backward": (
            "FP64 separated radial/tangent backward at all RMS sites, "
            "existing epsilons preserved"
            if repaired_sites
            else "native"
        ),
        "repaired_rms_sites": repaired_sites,
        "qk_normalization": (
            "repaired per-head RMS" if stage in QK_RMS_STAGES else "unit L2"
        ),
        "learned_attention_temperature": stage != "fixed-temperature",
        "removed_parameter_names": removed_parameters,
        "qk_gain_compensation": "exp(log_alpha) / head_dim"
        if stage in QK_RMS_STAGES
        else "exp(log_alpha)",
        "forward_residual_scale": residual_scale,
        "mlp_initialization": (
            "construct replacement MLPs after historical initialization; "
            "then Normal(0,.02) once per replacement matrix"
        ),
        "initialization_from_trained_weights": False,
    }
