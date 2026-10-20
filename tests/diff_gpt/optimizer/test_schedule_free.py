"""Tests for Schedule-Free AdamW."""

import copy
import math

import pytest
import torch
from torch.testing import assert_close

from diff_gpt.optimizer.schedule_free import ScheduleFreeAdamW

# test_step_follows_the_update: hyperparameters far from the defaults, the
# lr of each step (it falls twice, to 0 once) and the gradients of the four
# entries.
BETA1, BETA2, EPS, DECAY = 0.6, 0.75, 0.125, 0.3
RATES = (0.0, 0.2, 0.1, 0.0, 0.3)
GRADIENTS = (
    (1.0, -2.0, 0.5, 0.0),
    (0.0, 3.0, -1.0, 2.0),
    (-1.0, 0.0, 0.25, -0.5),
    (2.0, -0.5, 1.0, 1.0),
    (0.1, 0.3, -2.0, 0.0),
)
# Rounding error of 5 steps in float64.
EXACT = 1e-14


def _flat(*tensors: torch.Tensor) -> torch.Tensor:
    return torch.cat([t.detach().flatten() for t in tensors])


@pytest.mark.parametrize("warmup", [0, 3])
def test_step_follows_the_update(warmup: int) -> None:
    """
    Check y, z, x and the averaging total against the update written out per entry.

    The lr changes between steps and is 0 on two of them, so the warmup
    ramp, the bias correction of v, eps outside its square root, the decay
    at y and the weights of x, which follow the largest rate so far, all
    show. x is read in eval mode after every step.
    """
    p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([[0.5], [3.0]], dtype=torch.float64))
    opt = ScheduleFreeAdamW(
        [p, q],
        betas=(BETA1, BETA2),
        weight_decay=DECAY,
        eps=EPS,
        warmup=warmup,
    )
    z = [2.0, -1.0, 0.5, 3.0]
    x = list(z)
    v = [0.0] * len(z)
    total = maximum = 0.0
    for t, (lr, g) in enumerate(zip(RATES, GRADIENTS, strict=True), start=1):
        y = [BETA1 * xi + (1 - BETA1) * zi for xi, zi in zip(x, z, strict=True)]
        assert_close(
            _flat(p, q), torch.tensor(y, dtype=torch.float64), atol=EXACT, rtol=EXACT
        )
        rate = lr * (min(1, t / warmup) if warmup else 1)
        maximum = max(maximum, rate)
        total += maximum**2
        c = maximum**2 / total if total else 0.0
        for i, gi in enumerate(g):
            v[i] = BETA2 * v[i] + (1 - BETA2) * gi * gi
            corrected = math.sqrt(v[i] / (1 - BETA2**t))
            z[i] -= rate * gi / (corrected + EPS) + rate * DECAY * y[i]
            x[i] = (1 - c) * x[i] + c * z[i]
        p.grad = torch.tensor(g[:2], dtype=torch.float64)
        q.grad = torch.tensor(g[2:], dtype=torch.float64).reshape(2, 1)
        opt.param_groups[0]["lr"] = lr
        opt.step()
        assert_close(
            _flat(opt.state[p]["base"], opt.state[q]["base"]),
            torch.tensor(z, dtype=torch.float64),
            atol=EXACT,
            rtol=EXACT,
        )
        opt.eval()
        assert_close(
            _flat(p, q), torch.tensor(x, dtype=torch.float64), atol=EXACT, rtol=EXACT
        )
        opt.train()
        assert opt.param_groups[0]["total"] == pytest.approx(total, rel=EXACT)
        assert opt.param_groups[0]["step"] == t


def test_eval_writes_x_and_train_writes_y_back() -> None:
    """
    Check that eval() writes x, train() writes y, and repeats change nothing.

    eval() recovers x from y = (1 - β1)·z + β1·x, so the round trip gives
    back y up to rounding.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(3, 4))
    opt = ScheduleFreeAdamW([p], lr=0.1)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    y = p.detach().clone()
    z = opt.state[p]["base"]
    beta1 = opt.param_groups[0]["betas"][0]
    opt.eval()
    x = p.detach().clone()
    assert_close(x, (y - (1 - beta1) * z) / beta1)
    assert not torch.equal(x, y)
    opt.eval()
    assert torch.equal(p, x)
    opt.train()
    assert_close(p, y)
    back = p.detach().clone()
    opt.train()
    assert torch.equal(p, back)


def test_switching_modes_before_a_step_keeps_the_weights() -> None:
    """Check that z = x = the initial weights, so y and x are those bit for bit."""
    initial = torch.tensor([1.5, -1.5, 0.7, -0.3])
    p = torch.nn.Parameter(initial.clone())
    opt = ScheduleFreeAdamW([p])
    for _ in range(4):
        opt.eval()
        assert torch.equal(p, initial)
        opt.train()
        assert torch.equal(p, initial)


def test_zero_gradients_without_decay_keep_every_iterate() -> None:
    """Check that z, x and y stay at the initial weights bit for bit."""
    initial = torch.tensor([1.5, -1.5, 0.7, -0.3])
    p = torch.nn.Parameter(initial.clone())
    opt = ScheduleFreeAdamW([p], lr=0.1)
    p.grad = torch.zeros_like(p)
    for _ in range(20):
        opt.step()
        assert torch.equal(opt.state[p]["base"], initial)
        assert torch.equal(p, initial)
    opt.eval()
    assert torch.equal(p, initial)


def test_state_dict_resumes_the_same_steps() -> None:
    """
    Check that an optimizer loaded from a state dict takes the same next step.

    The state dict is saved in eval mode and carries the hyperparameters,
    the step count, the largest rate and the averaging total along with z
    and v; the weights hold x.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4))
    opt = ScheduleFreeAdamW([p], lr=0.1, warmup=3)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    opt.eval()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = ScheduleFreeAdamW([q], lr=0.9)
    # Loading keeps tensors that already match the parameter; copy them.
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert resumed.param_groups[0]["training"] is False
    for optimizer, param in ((opt, p), (resumed, q)):
        optimizer.train()
        param.grad = torch.ones_like(param)
        optimizer.step()
    assert torch.equal(p, q)


def test_converges_on_a_quadratic() -> None:
    """Check that 100 steps on |θ|² bring both y and x within 1 of the minimum."""
    theta = torch.nn.Parameter(torch.tensor([10.0, -10.0]))
    opt = ScheduleFreeAdamW([theta], lr=1.0, warmup=10)
    for _ in range(100):
        opt.zero_grad()
        (theta**2).sum().backward()
        opt.step()
    assert theta.norm() < 1.0
    opt.eval()
    assert theta.norm() < 1.0


def test_step_in_eval_mode_raises() -> None:
    """Check that step() refuses to run while the parameters hold x."""
    p = torch.nn.Parameter(torch.zeros(2))
    opt = ScheduleFreeAdamW([p])
    opt.eval()
    p.grad = torch.ones_like(p)
    with pytest.raises(RuntimeError, match=r"call train\(\) before step\(\)"):
        opt.step()


def test_step_needs_a_gradient_for_every_parameter() -> None:
    """Check that a parameter without a gradient stops the step."""
    p, q = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    opt = ScheduleFreeAdamW([p, q])
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="dense gradient for every parameter"):
        opt.step()


def test_leaves_out_tensors_that_do_not_require_grad() -> None:
    """Check that frozen tensors are not optimized, so they need no gradient."""
    p = torch.nn.Parameter(torch.zeros(2))
    frozen = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    opt = ScheduleFreeAdamW([p, frozen])
    params = opt.param_groups[0]["params"]
    assert len(params) == 1
    assert params[0] is p


def test_takes_one_parameter_group() -> None:
    """Check that parameter groups are refused, at construction and later."""
    p = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="one parameter group"):
        ScheduleFreeAdamW([{"params": [p]}])
    opt = ScheduleFreeAdamW([p])
    with pytest.raises(ValueError, match="one parameter group"):
        opt.add_param_group({"params": [torch.nn.Parameter(torch.zeros(2))]})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr": -1e-3}, "lr"),
        ({"lr": math.inf}, "lr"),
        ({"lr": math.nan}, "lr"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"eps": 0.0}, "eps"),
        ({"betas": (1.0, 0.999)}, "betas"),
        ({"betas": (0.0, 0.999)}, "betas"),
        ({"betas": (0.9, -0.1)}, "betas"),
        ({"warmup": -1}, "warmup"),
        ({"warmup": 1.5}, "warmup"),
    ],
)
def test_rejects_hyperparameters_out_of_range(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Check that each hyperparameter out of range raises ValueError naming it."""
    p = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match=match):
        ScheduleFreeAdamW([p], **kwargs)
