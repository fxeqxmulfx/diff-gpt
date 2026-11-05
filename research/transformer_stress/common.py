"""Shared measurement and exact-stack construction for independent stress tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.rms_norm import RMSNorm
from research.early99.experiments.learned_rope import LearnedRoPEGPT
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)
from research.early99.experiments.scalable_softmax import install_scalable_softmax

if TYPE_CHECKING:
    from torch import nn

ROOT = Path(__file__).resolve().parents[2]
EPS = torch.finfo(torch.float32).eps


def require(condition: bool, message: str) -> None:  # noqa: FBT001
    """Keep experiment integrity checks active under optimized Python."""
    if not condition:
        raise ValueError(message)


def finite(value: torch.Tensor) -> bool:
    """Check every tensor entry, rather than a potentially overflowing norm."""
    return bool(torch.isfinite(value).all())


def rms(value: torch.Tensor) -> float:
    """Measure RMS in FP64 even when the tested operation is FP32."""
    return value.detach().double().square().mean().sqrt().item()


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    """Measure relative L2 error, using an explicit floor for an exact zero."""
    return rms(value.double() - reference.double()) / max(rms(reference), 1e-30)


def clean(value: object) -> object:
    """Represent nonfinite measurements as null alongside explicit finite flags."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def write_json(path: Path, value: object) -> None:
    """Write strict JSON suitable for independent audits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def digest(values: dict[str, torch.Tensor]) -> str:
    """Fingerprint named tensor contents independent of archive metadata."""
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def source_manifest() -> dict[str, str]:
    """Pin every model and experiment dependency without editing frozen sources."""
    paths = [
        *ROOT.joinpath("src/diff_gpt/domain/model").glob("*.py"),
        *ROOT.joinpath("research/early99/experiments").glob("*.py"),
        *Path(__file__).parent.glob("*.py"),
        Path(__file__).with_name("plan.json"),
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def full_stack(settings: dict, layers: int, context: int) -> LearnedRoPEGPT:
    """Use the actual frozen factories' model operations in the actual order."""
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
    model = LearnedRoPEGPT(model, log_positions=True)
    # Freeze the tested FP32 epsilon before promoting a reference to FP64.
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.eps = EPS
    require(
        model.token_embedding_table.weight is model.lm_head.weight,
        "Embedding/readout parameter identity was lost",
    )
    require(
        all(
            not list(m.parameters()) for m in model.modules() if isinstance(m, RMSNorm)
        ),
        "RMS must remain parameter free",
    )
    return model


def gradient_norm(model: nn.Module) -> float:
    """Measure all available parameter gradients without FP32 norm overflow."""
    return math.sqrt(
        sum(
            parameter.grad.double().square().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
