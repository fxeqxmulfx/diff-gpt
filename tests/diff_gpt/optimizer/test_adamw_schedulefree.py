import torch
import pytest
from torch.testing import assert_close

from diff_gpt.optimizer.adamw_schedulefree import AdamWScheduleFree


def test_optimizer_convergence_quadratic():
    r"""
    Minimize convex objective: $\mathcal{L}(\theta) = \|\theta\|^2$.
    $\theta^* = \mathbf{0}$.
    Verify: $\lim_{t \to \infty} \|\theta_t\| < \epsilon$.
    """
    theta = torch.tensor([10.0, -10.0], requires_grad=True)
    # Using small LR to ensure stability without schedule
    opt = AdamWScheduleFree([theta], lr=1.0, warmup_steps=10)

    opt.train()

    # Optimization Loop
    for _ in range(100):
        loss = (theta**2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Check convergence in Train mode ($y_t$)
    assert torch.norm(theta) < 1.0, f"Failed to converge: {theta}"

    # Check convergence in Eval mode ($x_t$)
    opt.eval()
    assert torch.norm(theta) < 1.0


def test_state_transitions_train_eval():
    r"""
    Verify interpolation between auxiliary states.
    Optimizer maintains $y$ (train) and $z$ (anchor).

    $\text{Eval}: \theta \leftarrow x = \text{lerp}(y, z, 1 - \frac{1}{\beta_1})$
    $\text{Train}: \theta \leftarrow y = \text{lerp}(x, z, 1 - \beta_1)^{-1} \text{ (effectively restoring } y \text{)}$

    Verify: $\theta_{\text{train}} \neq \theta_{\text{eval}}$ after update.
    """
    param = torch.tensor([1.0], requires_grad=True)
    opt = AdamWScheduleFree([param], lr=0.1)

    opt.train()

    # 1. Perform multiple steps to allow divergence between y and z
    # At k=0, ckp1=1.0 => y collapses to z. We need k > 0.
    for _ in range(5):
        loss = param * 2
        loss.backward()
        opt.step()
        opt.zero_grad()

    val_train = param.clone()

    # 2. Switch to Eval
    opt.eval()
    val_eval = param.clone()

    # 3. Assert Difference ($y \neq x$)
    assert not torch.allclose(val_train, val_eval), (
        "Train and Eval weights should differ after update"
    )

    # 4. Switch back to Train
    opt.train()
    val_train_restored = param.clone()

    # 5. Assert Restoration ($y_{restored} \equiv y_{orig}$)
    assert_close(val_train, val_train_restored)


def test_warmup_linearity():
    r"""
    Verify Learning Rate schedule $\eta_t$.
    $\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}} \quad \forall t < T_{warmup}$
    """
    T_warmup = 10
    LR_max = 0.1
    param = torch.tensor([0.0], requires_grad=True)
    opt = AdamWScheduleFree([param], lr=LR_max, warmup_steps=T_warmup)

    opt.train()

    for t in range(1, T_warmup + 1):
        loss = param.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Check internal state for current LR
        current_lr = opt.param_groups[0]["scheduled_lr"]
        expected_lr = LR_max * (t / T_warmup)

        assert abs(current_lr - expected_lr) < 1e-6


def test_execution_guard_raises():
    r"""
    Verify Mode Guard.
    $\text{step}() \implies \text{Error if } \neg \text{TrainMode}$
    """
    param = torch.tensor([0.0], requires_grad=True)
    opt = AdamWScheduleFree([param], lr=1e-3)

    # Default init sets train_mode=False via defaults, but let's be explicit
    opt.eval()

    with pytest.raises(Exception) as excinfo:
        opt.step()

    assert "Optimizer was not in train mode" in str(excinfo.value)


def test_weight_decay_application():
    r"""
    Verify $L_2$ Regularization.
    $\mathcal{L}(\theta) = 0 \implies \nabla \mathcal{L} = 0$.
    Update: $\theta_{t+1} \leftarrow \theta_t - \eta \lambda \theta_t$.
    $\implies \|\theta_{t+1}\| < \|\theta_t\|$.
    """
    param = torch.tensor([10.0], requires_grad=True)
    opt = AdamWScheduleFree([param], lr=0.1, weight_decay=1.0)  # High decay

    opt.train()

    # Zero gradient step
    param.grad = torch.zeros_like(param)

    prev_norm = torch.norm(param)
    opt.step()
    new_norm = torch.norm(param)

    assert new_norm < prev_norm


def test_amsgrad_state_and_monotonic_max():
    """
    With amsgrad=True the optimizer must allocate `max_exp_avg_sq` and it
    must be a non-decreasing element-wise max of `exp_avg_sq` across steps.
    """
    torch.manual_seed(0)
    param = torch.randn(4, requires_grad=True)
    opt = AdamWScheduleFree([param], lr=0.01, amsgrad=True, foreach=False)
    opt.train()

    # Feed deliberately varying-magnitude gradients; exp_avg_sq will not
    # be monotonic, but max_exp_avg_sq must be.
    grad_magnitudes = [5.0, 0.1, 3.0, 0.05, 7.0, 0.2]
    prev_max = None
    for g in grad_magnitudes:
        param.grad = torch.full_like(param, g)
        opt.step()
        state = opt.state[param]
        assert "max_exp_avg_sq" in state
        current_max = state["max_exp_avg_sq"].clone()
        if prev_max is not None:
            # Element-wise: max must be >= previous max.
            assert (current_max >= prev_max - 1e-7).all()
        # And must dominate the current exp_avg_sq.
        assert (current_max >= state["exp_avg_sq"] - 1e-7).all()
        prev_max = current_max


def test_amsgrad_no_state_when_disabled():
    """amsgrad=False must not allocate max_exp_avg_sq state."""
    param = torch.randn(4, requires_grad=True)
    opt = AdamWScheduleFree([param], lr=0.01, amsgrad=False, foreach=False)
    opt.train()
    param.grad = torch.ones_like(param)
    opt.step()
    assert "max_exp_avg_sq" not in opt.state[param]


def test_amsgrad_default_is_true():
    """Default should now allocate max_exp_avg_sq state (amsgrad on by default)."""
    param = torch.randn(4, requires_grad=True)
    opt = AdamWScheduleFree([param], lr=0.01, foreach=False)
    opt.train()
    param.grad = torch.ones_like(param)
    opt.step()
    assert "max_exp_avg_sq" in opt.state[param]


def test_amsgrad_foreach_parity():
    """foreach and loop paths must agree when amsgrad=True."""
    torch.manual_seed(0)
    params_a = [torch.randn(10, 10, requires_grad=True) for _ in range(3)]
    params_b = [p.clone().detach().requires_grad_(True) for p in params_a]
    grads = [torch.randn_like(p) for p in params_a]

    opt_a = AdamWScheduleFree(params_a, lr=0.01, amsgrad=True, foreach=True)
    opt_b = AdamWScheduleFree(params_b, lr=0.01, amsgrad=True, foreach=False)
    opt_a.train()
    opt_b.train()

    # Two steps so amsgrad's running-max actually diverges from exp_avg_sq.
    for _ in range(2):
        for i in range(len(params_a)):
            params_a[i].grad = grads[i].clone()
            params_b[i].grad = grads[i].clone()
        opt_a.step()
        opt_b.step()

    for pa, pb in zip(params_a, params_b):
        assert_close(pa, pb, atol=1e-6, rtol=1e-6)
    for pa, pb in zip(params_a, params_b):
        assert_close(
            opt_a.state[pa]["max_exp_avg_sq"],
            opt_b.state[pb]["max_exp_avg_sq"],
            atol=1e-6,
            rtol=1e-6,
        )


def test_amsgrad_converges_on_quadratic():
    """Sanity: amsgrad variant still minimizes a simple convex target."""
    theta = torch.tensor([10.0, -10.0], requires_grad=True)
    opt = AdamWScheduleFree([theta], lr=1.0, warmup_steps=10, amsgrad=True)
    opt.train()
    for _ in range(100):
        loss = (theta**2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert torch.norm(theta) < 1.0


def test_foreach_parity():
    r"""
    Verify implementation consistency.
    $f_{\text{foreach}}(\theta) \equiv f_{\text{loop}}(\theta)$
    """
    params_a = [torch.randn(10, 10, requires_grad=True) for _ in range(3)]
    params_b = [p.clone().detach().requires_grad_(True) for p in params_a]

    grad = [torch.randn_like(p) for p in params_a]

    # Optimizer A: Foreach
    opt_a = AdamWScheduleFree(params_a, lr=0.01, foreach=True)
    opt_a.train()

    # Optimizer B: Loop
    opt_b = AdamWScheduleFree(params_b, lr=0.01, foreach=False)
    opt_b.train()

    # Step
    for i in range(len(params_a)):
        params_a[i].grad = grad[i].clone()
        params_b[i].grad = grad[i].clone()

    opt_a.step()
    opt_b.step()

    for pa, pb in zip(params_a, params_b):
        assert_close(pa, pb, atol=1e-6, rtol=1e-6)
