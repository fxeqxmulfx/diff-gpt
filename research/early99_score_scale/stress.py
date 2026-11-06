"""Preserve every finite-precision score/gradient probe before benchmark training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.rms_norm import rms_norm

PROBABILITY_ERROR_LIMIT = 2e-6
GRADIENT_ERROR_LIMIT = 1e-3


def probe(multiplier: float, length: int, width: int, scale: float, seed: int) -> dict:
    """Compare a full attention row against independent FP64 differentiation."""
    generator = torch.Generator().manual_seed(seed + width * 10000 + length)
    q = (torch.randn(1, width, generator=generator) * scale).requires_grad_()
    k = (torch.randn(length, width, generator=generator) * scale).requires_grad_()
    qr, kr = q.detach().double().requires_grad_(), k.detach().double().requires_grad_()
    eps = torch.finfo(torch.float32).eps
    factor = multiplier * math.log(length) / math.sqrt(width)
    scores = (rms_norm(q, eps) @ rms_norm(k, eps).T) * factor
    reference = (
        F.rms_norm(qr, (width,), eps=eps) @ F.rms_norm(kr, (width,), eps=eps).T
    ) * factor
    weights, expected = scores.softmax(-1), reference.softmax(-1)
    values = torch.linspace(-1, 1, length).reshape(1, -1)
    gradients = torch.autograd.grad((weights * values).sum(), (q, k))
    refs = torch.autograd.grad((expected * values.double()).sum(), (qr, kr))
    errors = [
        float((a.double() - b).norm() / b.norm().clamp_min(1e-300))
        for a, b in zip(gradients, refs, strict=True)
    ]
    bound = multiplier * math.log(length) * math.sqrt(width)
    finite = all(bool(torch.isfinite(t).all()) for t in (scores, weights, *gradients))
    maximum = float(scores.detach().abs().max())
    probability_error = float((weights.double() - expected).detach().abs().max())
    return {
        "multiplier": multiplier,
        "visible_keys": length,
        "head_width": width,
        "qk_scale": scale,
        "finite": finite,
        "maximum_absolute_score": maximum,
        "real_score_bound": bound,
        "maximum_probability_absolute_error": probability_error,
        "query_relative_gradient_error": errors[0],
        "key_relative_gradient_error": errors[1],
        "within_declared_bound_slack": maximum <= bound + 1e-5 * max(1, bound),
        "accurate_probability": probability_error <= PROBABILITY_ERROR_LIMIT,
        "accurate_gradients": max(errors) <= GRADIENT_ERROR_LIMIT,
    }


def run() -> dict:
    """Execute the frozen grid and report failures without changing the grid."""
    plan_path = Path(__file__).with_name("plan.json")
    settings = json.loads(plan_path.read_text())["score_stress"]
    torch.set_num_threads(2)
    rows = [
        probe(m, n, d, scale, settings["base_seed"])
        for m in settings["multipliers"]
        for n in settings["visible_keys"]
        for d in settings["head_widths"]
        for scale in settings["qk_input_scales"]
    ]
    if len(rows) != settings["expected_cases"]:
        message = "Stress grid differs from its declaration"
        raise ValueError(message)
    failures = [
        row
        for row in rows
        if not all(
            row[key]
            for key in (
                "finite",
                "within_declared_bound_slack",
                "accurate_probability",
                "accurate_gradients",
            )
        )
    ]
    return {
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "expected_cases": len(rows),
        "passed": len(rows) - len(failures),
        "failures": failures,
        "maximum_probability_absolute_error": max(
            r["maximum_probability_absolute_error"] for r in rows
        ),
        "maximum_relative_gradient_error": max(
            max(r["query_relative_gradient_error"], r["key_relative_gradient_error"])
            for r in rows
        ),
        "scope": settings["scope"],
        "rows": rows,
    }


if __name__ == "__main__":
    result = run()
    destination = Path("runs/early99-score-scale/score-stress.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"rows", "failures"}})
    )
