"""
Task151: post-RMS amplitude recovery, including a deliberately broken control.

The supplied-feature regression isolates gain learning; it is not evidence
that learned gates improve modular-division grokking in a full transformer.
"""

import copy

import pytest
import torch
from torch.nn import functional as F

from diff_gpt.domain.model.residual_gpt import ResidualGPT
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW


def make(mode: str) -> ResidualGPT:
    """Paired common parameters with exactly unit gates at initialization."""
    torch.manual_seed(151)
    torch.set_num_threads(1)
    return ResidualGPT(
        11, 32, 8, 4, 2, mixing=mode, use_checkpoint=False, pad_vocab_size_to=1
    )


def test_gated_transformer_initially_matches_ordinary_values_and_gradients() -> None:
    """New gates do not change the common source model at initialization."""
    ordinary, gated = make("add"), make("gated")
    tokens = torch.tensor([[1, 2, 3], [2, 2, 4], [4, 4, 4]])
    a, b = ordinary(tokens)[0], gated(tokens)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, parameter in ordinary.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            dict(gated.named_parameters())[name].grad,
            rtol=2e-5,
            atol=2e-8,
        )
    assert gated.branch_logits.grad is not None
    assert torch.isfinite(gated.branch_logits.grad).all()
    minimum_gradient = 1e-5
    assert gated.branch_logits.grad.norm() > minimum_gradient


@pytest.mark.parametrize("seed", [11, 17])
def test_post_norm_gate_recovers_amplitude_while_pre_norm_control_stalls(
    seed: int,
) -> None:
    """Actual GPT gain learns x/4; moving the same gain inside RMS must fail."""
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(2, (192, 32), generator=generator).float() * 2 - 1
    train, heldout = signs[:64], signs[64:]
    model = make("gated")
    model.requires_grad_(requires_grad=False)
    model.branch_logits.requires_grad_(requires_grad=True)
    broken = copy.deepcopy(model)
    outcomes = []
    for current, before in ((model, False), (broken, True)):
        optimizer = ScheduleFreeAdamW(
            [current.branch_logits], lr=0.03, betas=(0.9, 0.98), weight_decay=0.0
        )

        def branch(
            value: torch.Tensor,
            current: ResidualGPT = current,
            *,
            before: bool = before,
        ) -> torch.Tensor:
            if before:
                return F.rms_norm(current.scale_branch(0, value), (32,), eps=2**-23)
            return current.scale_branch(0, F.rms_norm(value, (32,), eps=2**-23))

        initial = branch(train)
        torch.testing.assert_close(initial, train, rtol=0, atol=0)
        first_gradient = torch.autograd.grad(
            (initial - train / 4).square().mean(), current.branch_logits
        )[0]
        if before:
            assert torch.count_nonzero(first_gradient) == 0
        else:
            useful_gradient = 0.7
            assert first_gradient[0] > useful_gradient
        for _ in range(1024):
            optimizer.zero_grad(set_to_none=True)
            (branch(train) - train / 4).square().mean().backward()
            optimizer.step()
        optimizer.eval()
        with torch.no_grad():
            outcomes.append(
                float((branch(heldout) - heldout / 4).square().mean() / (1 / 16))
            )
    recovery_threshold, broken_floor = 1e-4, 8.9
    assert outcomes[0] < recovery_threshold
    assert outcomes[1] > broken_floor
