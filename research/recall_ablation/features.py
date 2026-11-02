"""Fresh architectural candidates under the passing ordinary-AdamW recipe."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

from diff_gpt.domain.model.gpt import GPTMini
from diff_gpt.domain.model.mini_block import RMSNormEmbedding, initialize_swiglu

if TYPE_CHECKING:
    from torch import nn

ARCHITECTURE_STAGES = ("swiglu", "scaled-init", "embedding-norm")
PREDECESSOR = {
    "swiglu": "baseline",
    "scaled-init": "swiglu",
    "embedding-norm": "swiglu",
}


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
        if model.token_embedding_table.weight is not model.lm_head.weight:
            msg = "embedding normalization broke output-head weight tying"
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
        "mlp_initialization": (
            "construct replacement MLPs after historical initialization; "
            "then Normal(0,.02) once per replacement matrix"
        ),
        "initialization_from_trained_weights": False,
    }
