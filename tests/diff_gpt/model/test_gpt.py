import torch
import torch.nn.functional as F
from torch.testing import assert_close

from diff_gpt.model.gpt import GPT, _gaussian_soft_ce
from diff_gpt.model.kv_cache import KVCache


def test_gpt_initialization_and_shapes():
    r"""
    Verify structural constraints.
    1. Weight Tying: $W_{emb} \equiv W_{head}$
    2. Output Shape: $x \in \mathbb{Z}^{B \times T} \implies \text{logits} \in \mathbb{R}^{B \times T \times V}$
    """
    B, T, V, C = 2, 10, 128, 32

    model = GPT(vocab_size=V, n_embd=C, block_size=20, n_layer=2, n_head=4)

    # Check Weight Tying
    # $\theta_{emb} \leftarrow \theta_{head}$
    assert model.token_embedding_table.weight is model.lm_head.weight

    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))

    # Forward with Loss
    logits, loss = model(idx, targets)

    # Check Shapes
    assert logits.shape == (B, T, V)
    assert loss.dim() == 0  # Scalar $\mathcal{L}$


def test_gpt_kv_cache_consistency():
    r"""
    Verify equivalence between Parallel (Batch) and Serial (Step-by-step) inference.

    $\text{Let } \Phi(x_{0:T}) \to y_{0:T}$ (Full Context)
    $\text{Let } \phi(x_t, S_{t-1}) \to (y_t, S_t)$ (Incremental)

    Verify: $\Phi(x)_{t} \approx \phi(x_t, S_{t-1})$

    This validates correct RoPE slicing:
    $\text{freqs} = \mathbf{F}[S_{pos} : S_{pos}+1]$
    """
    B, T, V, C = 1, 5, 64, 16
    H = 2

    model = GPT(
        vocab_size=V, n_embd=C, block_size=20, n_layer=1, n_head=H, use_checkpoint=False
    )
    model.eval()

    idx = torch.randint(0, V, (B, T))

    # 1. Full Forward (Ground Truth)
    with torch.no_grad():
        logits_full, _ = model(idx)  # (B, T, V)

    # 2. Step-by-Step with KVCache
    kv = KVCache(batch_size=B, num_heads=H, seq_len=20, head_dim=C // H, num_layers=1)

    outputs = []
    for t in range(T):
        idx_step = idx[:, t : t + 1]  # (B, 1)

        with torch.no_grad():
            # forward handles kv_cache.get_pos() internally to slice freqs_cis
            logits_step, _ = model(idx_step, kv_cache=kv)
            outputs.append(logits_step)

    logits_serial = torch.cat(outputs, dim=1)  # (B, T, V)

    # Check Logits Equivalence
    # $|y_{batch} - y_{serial}| < \epsilon$
    assert_close(logits_full, logits_serial, atol=1e-5, rtol=1e-5)


def test_gradient_checkpointing_flow():
    r"""
    Verify backward pass with Activation Checkpointing.
    $\frac{\partial \mathcal{L}}{\partial \theta} \neq \mathbf{0}$
    """
    B, T, V = 2, 8, 128

    # Enable checkkpointing
    model = GPT(vocab_size=V, n_embd=32, n_layer=2, use_checkpoint=True)
    model.train()

    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))

    logits, loss = model(idx, targets)

    loss.backward()

    # Verify Gradients on Input Embeddings (Proxy for full backward flow)
    assert model.token_embedding_table.weight.grad is not None
    assert torch.norm(model.token_embedding_table.weight.grad) > 0


def test_generation_output_len():
    r"""
    Verify autoregressive loop termination.
    $T_{out} = T_{in} + T_{new}$
    """
    B, T, V, N_new = 1, 5, 64, 3
    model = GPT(vocab_size=V, n_embd=32, block_size=10, n_layer=1)
    model.eval()

    idx = torch.randint(0, V, (B, T))

    # Naive generation (no KV cache passed to generate method in provided snippet)
    out_idx = model.generate(idx, max_new_tokens=N_new, sampler=None)

    assert out_idx.shape == (B, T + N_new)


def test_generate_seed_determinism():
    """
    Passing the same `seed` must yield identical outputs; different seeds differ.
    """
    from diff_gpt.sampler.temperature import TemperatureSampler

    B, T, V = 1, 5, 64
    model = GPT(vocab_size=V, n_embd=32, block_size=16, n_layer=1)
    model.eval()
    idx = torch.randint(0, V, (B, T))
    # High temperature flattens the distribution so different seeds actually diverge.
    sampler = TemperatureSampler(temperature=20.0)

    a = model.generate(idx, max_new_tokens=8, sampler=sampler, seed=7)
    b = model.generate(idx, max_new_tokens=8, sampler=sampler, seed=7)
    c = model.generate(idx, max_new_tokens=8, sampler=sampler, seed=99)

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_gpt_attn_res_projections_receive_gradient():
    """
    End-to-end: the per-layer AttnRes pseudo-queries must receive gradient
    through the full forward/backward.

    Caveat (paper eq. 4): at layer 0, the cross-block attention has only
    v_0=embedding as its unique key (the pseudocode's partial_block=embed
    just duplicates it), so the softmax over a single effective key is
    degenerate and attn_res_proj at layer 0 can't receive signal. All other
    sub-layers have distinct keys and must get gradient.
    """
    B, T, V = 2, 6, 64
    model = GPT(
        vocab_size=V,
        n_embd=16,
        block_size=12,
        n_layer=4,
        n_head=2,
        attn_res_layers_per_block=2,
    )
    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    _, loss = model(idx, targets)
    loss.backward()
    # mlp_res_proj is exercised with distinct keys at every layer (since by
    # then the attn output has been added to the partial sum).
    for block in model.blocks:
        assert block.mlp_res_proj.weight.grad is not None
        assert torch.norm(block.mlp_res_proj.weight.grad) > 0
    # attn_res_proj gets gradient for every layer except the very first.
    for block in list(model.blocks)[1:]:
        assert block.attn_res_proj.weight.grad is not None
        assert torch.norm(block.attn_res_proj.weight.grad) > 0


def test_vocab_size_must_be_aligned_to_pad():
    """GPT asserts the caller supplies a pre-padded vocab_size."""
    import pytest

    with pytest.raises(AssertionError, match="multiple of"):
        GPT(vocab_size=70, n_embd=16, block_size=8, n_layer=1, n_head=2)

    # pad=1 disables the check (any vocab_size accepted)
    GPT(
        vocab_size=70, n_embd=16, block_size=8, n_layer=1, n_head=2, pad_vocab_size_to=1
    )


def test_logit_softcap_bounds_output():
    """Default softcap = 15 bounds logits to |x| <= softcap (saturating at tanh=±1)."""
    B, T, V = 2, 6, 64
    model = GPT(vocab_size=V, n_embd=16, block_size=12, n_layer=2, n_head=2)
    model.eval()
    # Blow up lm_head so raw logits would otherwise be enormous.
    with torch.no_grad():
        model.lm_head.weight.mul_(100.0)
    idx = torch.randint(0, V, (B, T))
    with torch.no_grad():
        logits, _ = model(idx)
    cap = model.logit_softcap
    assert logits.abs().max().item() <= cap + 1e-5
    # And the clamp actually did something — max reached the ceiling.
    assert logits.abs().max().item() >= cap - 1e-3


def test_logit_softcap_disabled_when_zero():
    """softcap=0 must be a no-op — large logits survive unclamped."""
    B, T, V = 1, 4, 64
    model = GPT(
        vocab_size=V, n_embd=16, block_size=8, n_layer=1, n_head=2, logit_softcap=0.0
    )
    model.eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(100.0)
    idx = torch.randint(0, V, (B, T))
    with torch.no_grad():
        logits, _ = model(idx)
    # Without softcap, scaling weights by 100 must push max logit well past 15.
    assert logits.abs().max().item() > 15.0


def test_soft_ce_matches_plain_ce_when_sigma_is_tiny():
    """σ→0 makes the soft-target Gaussian collapse to a one-hot at `target`,
    so the soft-CE loss must match F.cross_entropy up to float precision."""
    torch.manual_seed(0)
    N, C = 32, 64
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    plain = F.cross_entropy(logits, targets)
    soft = _gaussian_soft_ce(logits, targets, sigma=1e-3)
    assert_close(soft, plain, atol=1e-5, rtol=1e-5)


def test_soft_ce_penalizes_near_miss_less_than_far_miss():
    """Core ordinal property: given two predictions equally confident but
    one is close to the true bin and the other far, soft-CE must prefer the
    near miss. Plain CE is blind to this."""
    C = 64
    target_bin = 32
    targets = torch.tensor([target_bin])
    # Two one-hot-like logit configurations; identical peak height, different locations.
    near = torch.full((1, C), -10.0)
    far = torch.full((1, C), -10.0)
    near[0, target_bin + 2] = 10.0  # off by 2 bins
    far[0, target_bin + 20] = 10.0  # off by 20 bins
    loss_near = _gaussian_soft_ce(near, targets, sigma=2.0)
    loss_far = _gaussian_soft_ce(far, targets, sigma=2.0)
    assert loss_near < loss_far, f"near={loss_near} should be < far={loss_far}"
    # Plain CE gives identical loss for both — it's distance-blind.
    plain_near = F.cross_entropy(near, targets)
    plain_far = F.cross_entropy(far, targets)
    assert_close(plain_near, plain_far)


def test_soft_ce_respects_ignore_index():
    """Positions with target=-100 must be skipped (required for column-masked
    TimeXer-style training where non-target channels are marked -100)."""
    torch.manual_seed(0)
    N, C = 16, 64
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    # Mask half; their logits become irrelevant.
    mask = torch.arange(N) < N // 2
    targets_masked = targets.clone()
    targets_masked[~mask] = -100
    loss_with_junk = _gaussian_soft_ce(
        logits.clone(), targets_masked, sigma=1.0
    )
    # Ground truth: only the unmasked half.
    loss_half = _gaussian_soft_ce(logits[mask], targets[mask], sigma=1.0)
    assert_close(loss_with_junk, loss_half)


def test_soft_ce_all_masked_returns_zero_with_gradient():
    """If every position is ignored, loss must be 0 but still backprop-safe
    (so training loops that occasionally see a fully-masked micro-batch
    don't blow up)."""
    torch.manual_seed(0)
    logits = torch.randn(8, 64, requires_grad=True)
    targets = torch.full((8,), -100)
    loss = _gaussian_soft_ce(logits, targets, sigma=1.0)
    assert loss.item() == 0.0
    loss.backward()  # must not raise
    assert logits.grad is not None
    assert torch.all(logits.grad == 0)


def test_gpt_with_soft_ce_trains():
    """End-to-end: GPT configured with label_smoothing_sigma>0 still
    produces a finite scalar loss and gradient flow."""
    torch.manual_seed(0)
    B, T, V = 2, 6, 64
    model = GPT(
        vocab_size=V,
        n_embd=16,
        block_size=12,
        n_layer=2,
        n_head=2,
        label_smoothing_sigma=1.5,
    )
    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    _, loss = model(idx, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding_table.weight.grad is not None
    assert torch.norm(model.token_embedding_table.weight.grad) > 0


def test_gpt_rejects_negative_sigma():
    import pytest

    with pytest.raises(AssertionError, match="label_smoothing_sigma"):
        GPT(vocab_size=64, n_embd=16, n_layer=1, n_head=2, label_smoothing_sigma=-0.1)


def test_overfitting_capability():
    r"""
    Sanity check: Model capacity.
    $\lim_{i \to \infty} \mathcal{L}(\theta_i) \to 0$ on a single batch.
    """
    B, T, V = 1, 4, 64
    model = GPT(vocab_size=V, n_embd=32, n_layer=2, n_head=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))

    for _ in range(50):
        optimizer.zero_grad()
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1
