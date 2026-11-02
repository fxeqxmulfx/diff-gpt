"""Preserve finite-epsilon gradients in the archived normalized GPT paths."""

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.domain.model.normalized_gpt import NormalizedGPT as GPT
from diff_gpt.domain.model.rms_norm import RMSNorm, rms_norm
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

EPS = 2**-23


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_forward_and_independent_vjp(dtype: torch.dtype, device: str) -> None:
    """Keep the source forward and check gradients against a real FP64 oracle."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator(device=device).manual_seed(155)
    eps = torch.finfo(torch.float64 if dtype == torch.float64 else torch.float32).eps
    for scale in (0.001, 1.0, 1000.0):
        value = (torch.randn((4, 32), generator=generator, device=device) * scale).to(
            dtype
        )
        value.requires_grad_()
        output = rms_norm(value)
        torch.testing.assert_close(output, F.rms_norm(value, (32,)), rtol=0, atol=0)
        upstream = torch.randn(value.shape, generator=generator, device=device).to(
            dtype
        )
        actual = torch.autograd.grad(output, value, upstream)[0]
        reference = value.detach().double().requires_grad_()
        expected = torch.autograd.grad(
            F.rms_norm(reference, (32,), eps=eps), reference, upstream.double()
        )[0]
        torch.testing.assert_close(actual, expected.to(dtype), rtol=2e-6, atol=1e-10)


def test_actual_gpt_norm_retains_radial_derivative() -> None:
    """Every Peri/embedding/final norm keeps the positive unit-scale derivative."""
    model = GPT(64, 32, n_layer=2, use_checkpoint=False)
    norms = [module for module in model.modules() if isinstance(module, nn.RMSNorm)]
    assert len(norms) == 2 + 4 * model.n_layer
    for norm in norms:
        assert not list(norm.parameters())
        logit = torch.tensor(0.0, requires_grad=True)
        output = norm(2 * logit.sigmoid() * torch.ones(1, 32))
        gradient = torch.autograd.grad((output - 0.25).square().mean(), logit)[0]
        amplitude = 1 / math.sqrt(1 + EPS)
        expected = (amplitude - 0.25) * EPS / (1 + EPS) ** 1.5
        assert gradient.item() == pytest.approx(expected, rel=2e-6)


def test_compiled_radial_gradient_and_zero_input() -> None:
    """The production module retains the oracle gradient under max-autotune."""
    norm = torch.compile(
        RMSNorm(32), mode="max-autotune", fullgraph=True, dynamic=False
    )
    value = torch.ones(4, 32, requires_grad=True)
    gradient = torch.autograd.grad(norm(value).sum(), value)[0]
    torch.testing.assert_close(gradient, torch.full_like(value, EPS / (1 + EPS) ** 1.5))
    zero = torch.zeros_like(value, requires_grad=True)
    gradient = torch.autograd.grad(norm(zero).sum(), zero)[0]
    torch.testing.assert_close(gradient, torch.full_like(zero, EPS**-0.5))


def test_higher_derivatives() -> None:
    """The rearranged VJP remains differentiable away from zero input."""
    sample = torch.tensor(
        [[0.2, -0.4, 0.7, 0.9]], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(lambda value: rms_norm(value, EPS), sample)
    assert torch.autograd.gradgradcheck(lambda value: rms_norm(value, EPS), sample)


@pytest.mark.parametrize("seed", [11, 17])
def test_actual_norm_recovers_heldout_amplitude_with_schedule_free(seed: int) -> None:
    """A known finite solution recovers; the native FP32 control stays stuck."""
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(2, (192, 32), generator=generator).float() * 2 - 1
    train, heldout = signs[:64], signs[64:]
    model = GPT(64, 32, n_layer=1, use_checkpoint=False)
    outcomes = []
    for norm in (
        model.blocks[0].ffn_out_norm,
        nn.RMSNorm(32, elementwise_affine=False),
    ):
        logit = nn.Parameter(torch.tensor(0.0))
        optimizer = ScheduleFreeAdamW(
            [logit], lr=0.01, betas=(0.9, 0.98), weight_decay=0.0
        )
        for _ in range(6000):
            optimizer.zero_grad(set_to_none=True)
            output = norm(2 * logit.sigmoid() * train)
            (output - train / 4).square().mean().backward()
            optimizer.step()
        optimizer.eval()
        with torch.no_grad():
            output = norm(2 * logit.sigmoid() * heldout)
            outcomes.append(float((output - heldout / 4).square().mean() / (1 / 16)))
    recovery_threshold, broken_floor = 1e-4, 8.9
    assert outcomes[0] < recovery_threshold
    assert outcomes[1] > broken_floor
