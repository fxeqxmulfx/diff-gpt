"""Probe the declared inverse-width score equation against independent FP64."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.gpt import GPT
from research.early99.experiments.scalable_softmax import install_scalable_softmax
from research.early99_dimension_gain.factories import install_inverse_width

PLAN = Path(__file__).with_name("plan.json")


def probe(length: int, width: int, scale: float, settings: dict) -> dict:
    """Use production normalization/gain and an independent full-row derivative."""
    model = GPT(
        3,
        width,
        block_size=1,
        n_head=1,
        n_layer=1,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    )
    install_scalable_softmax(model)
    install_inverse_width(model)
    attention = model.blocks[0].sa
    generator = torch.Generator().manual_seed(
        settings["base_seed"] + width * 10000 + length
    )
    q = (torch.randn(1, width, generator=generator) * scale).requires_grad_()
    k = (torch.randn(length, width, generator=generator) * scale).requires_grad_()
    qr = q.detach().double().requires_grad_()
    kr = k.detach().double().requires_grad_()
    eps = settings["fp32_epsilon"]
    if eps != torch.finfo(torch.float32).eps:
        message = "Declared epsilon differs from production FP32 RMS"
        raise ValueError(message)
    qn, kn = attention.normalize_qk(q, k)
    gain = attention.attention_gain().reshape(())
    count_log = torch.tensor(length, dtype=q.dtype).log()
    scores = ((qn * count_log) @ kn.T) * gain
    reference = (
        F.rms_norm(qr, (width,), eps=eps) @ F.rms_norm(kr, (width,), eps=eps).T
    ) * (math.log(length) * float(gain))
    weights, expected = scores.softmax(-1), reference.softmax(-1)
    values = torch.linspace(-1, 1, length).reshape(1, -1)
    gradients = torch.autograd.grad((weights * values).sum(), (q, k))
    refs = torch.autograd.grad((expected * values.double()).sum(), (qr, kr))
    errors = [
        float((actual.double() - ref).norm() / ref.norm().clamp_min(1e-300))
        for actual, ref in zip(gradients, refs, strict=True)
    ]
    bound = math.log(length) * abs(float(gain)) * width
    maximum = float(scores.detach().abs().max())
    probability_error = float((weights.double() - expected).detach().abs().max())
    return {
        "visible_keys": length,
        "head_width": width,
        "qk_scale": scale,
        "stored_gain": float(gain),
        "ideal_gain": 1 / width,
        "qk_rms_retained": attention.use_qk_norm,
        "finite": all(
            bool(torch.isfinite(t).all()) for t in (scores, weights, *gradients)
        ),
        "maximum_absolute_score": maximum,
        "real_score_bound_with_stored_gain": bound,
        "ideal_real_score_bound": math.log(length),
        "maximum_probability_absolute_error": probability_error,
        "query_relative_gradient_error": errors[0],
        "key_relative_gradient_error": errors[1],
        "within_declared_bound_slack": maximum <= bound + 1e-5 * max(1, bound),
        "accurate_probability": probability_error
        <= settings["maximum_probability_absolute_error"],
        "accurate_gradients": max(errors)
        <= settings["maximum_relative_gradient_error"],
    }


def run() -> dict:
    """Retain the entire 120-case Cartesian grid, including zero inputs and N=1."""
    settings = json.loads(PLAN.read_text())["score_stress"]
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    rows = [
        probe(n, d, scale, settings)
        for d, n, scale in itertools.product(
            settings["head_widths"], settings["visible_keys"], settings["raw_qk_scales"]
        )
    ]
    if len(rows) != settings["expected_cases"]:
        message = "Stress grid differs from its declaration"
        raise ValueError(message)
    failures = [
        row
        for row in rows
        if not all(
            row[k]
            for k in (
                "qk_rms_retained",
                "finite",
                "within_declared_bound_slack",
                "accurate_probability",
                "accurate_gradients",
            )
        )
    ]
    sources = [Path(__file__), PLAN.with_name("factories.py")]
    return {
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
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
        "scope": (
            "Score arithmetic only, on the same quantized inputs; prior full-stack "
            "extreme failures remain. No FP32 or generalization theorem is inferred."
        ),
        "rows": rows,
    }


if __name__ == "__main__":
    result = run()
    destination = Path("runs/early99-dimension-gain/score-stress.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"rows", "failures"}})
    )
