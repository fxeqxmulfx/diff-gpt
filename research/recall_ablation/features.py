"""Fresh architectural candidates under the passing ordinary-AdamW recipe."""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.gpt import GPT

ARCHITECTURE_STAGES = ("swiglu",)


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


def build_model(stage: str, seed: int) -> tuple[GPT, dict[str, object]]:
    """Keep historical shared weights and replace only the declared feature."""
    torch.manual_seed(seed)
    model = GPT(
        vocab_size=548,
        n_embd=128,
        n_layer=6,
        n_head=4,
        block_size=64,
        pad_vocab_size_to=1,
        use_checkpoint=False,
    )
    historical = hashlib.sha256()
    for parameter in model.parameters():
        historical.update(parameter.detach().numpy().tobytes())
    shared = parameter_digest(model, exclude_mlp=True)
    if stage == "swiglu":
        # Construct all replacements before drawing their Normal(0,.02)
        # weights. Historical shared tensors are already initialized and
        # remain byte-identical; no trained endpoint is used to initialize.
        replacements = [FeedForward(model.n_embd) for _ in model.blocks]
        for block, replacement in zip(model.blocks, replacements, strict=True):
            for parameter in replacement.parameters():
                nn.init.normal_(parameter, mean=0.0, std=0.02)
            block.ffwd = replacement
    elif stage not in {"baseline", "schedule-free"}:
        msg = f"unknown architectural feature {stage}"
        raise ValueError(msg)
    if parameter_digest(model, exclude_mlp=True) != shared:
        msg = "an MLP change modified shared historical weights"
        raise RuntimeError(msg)
    return model, {
        "features": ["SwiGLU"] if stage == "swiglu" else [],
        "historical_initial_parameters_sha256": historical.hexdigest(),
        "shared_non_mlp_initial_sha256": shared,
        "mlp_initialization": (
            "construct replacement MLPs after historical initialization; "
            "then Normal(0,.02) once per replacement matrix"
        ),
        "initialization_from_trained_weights": False,
    }
