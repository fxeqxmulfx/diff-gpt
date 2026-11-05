"""Strict pairing of original tensors and explicitly declared added parameters."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

BASE_CLASS = "diff_gpt.domain.model.gpt.GPT"
LEROPE_CLASS = "research.early99.experiments.learned_rope.LearnedRoPEGPT"
LEROPE_VARIANTS = {
    "peri-rms-embedding-scaled-lerope",
    "peri-rms-embedding-scaled-ssmax-lerope",
    "peri-rms-embedding-scaled-ssmax-lerope-log-angle",
}


def parameter_metadata(model: torch.nn.Module) -> dict:
    """Hash every tensor, and separately the unchanged parent when LeRoPE adds one."""
    full, common = hashlib.sha256(), hashlib.sha256()
    added = {}
    for name, parameter in model.named_parameters():
        raw = parameter.detach().cpu().contiguous()
        full.update(name.encode())
        full.update(raw.numpy().tobytes())
        if name == "lerope.log_scales":
            added[name] = {"shape": list(raw.shape), "initial_values": raw.tolist()}
        else:
            common.update(name.encode())
            common.update(raw.numpy().tobytes())
    result = {
        "initial_parameters_sha256": full.hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_class": type(model).__module__ + "." + type(model).__qualname__,
    }
    if added:
        result.update(
            shared_initial_parameters_sha256=common.hexdigest(), added_parameters=added
        )
    return result


def parent_initialization(metadata: dict) -> tuple[str, int]:
    """Accept exactly the declared LeRoPE vector, or require the original GPT."""
    variant = (
        metadata.get("experiment", {})
        .get("configuration", {})
        .get("model_variant", "baseline")
    )
    if variant not in LEROPE_VARIANTS:
        if metadata["model_class"] != BASE_CLASS or metadata.get("added_parameters"):
            message = "Undeclared model class or additional parameters"
            raise ValueError(message)
        return metadata["initial_parameters_sha256"], metadata["parameters"]
    bands = metadata["case"]["width"] // metadata["case"]["heads"] // 2
    expected = {
        "lerope.log_scales": {"shape": [bands], "initial_values": [0.0] * bands}
    }
    if (
        metadata["model_class"] != LEROPE_CLASS
        or metadata.get("added_parameters") != expected
    ):
        message = "LeRoPE initialization is not the declared zero frequency vector"
        raise ValueError(message)
    return metadata["shared_initial_parameters_sha256"], metadata["parameters"] - bands


def verify_paired_initialization(base: dict, new: dict) -> None:
    """Enforce identical original tensors and capacity, including the LeRoPE arm."""
    if parent_initialization(base) != parent_initialization(new):
        message = (
            "Parameter-preserving experiment changed "
            "original initialization or capacity"
        )
        raise ValueError(message)
