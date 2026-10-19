import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from diff_gpt.model.gpt import GPT
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


def test_gpt2_init():
    r"""
    GPT-2 init: weights $\sim \mathcal{N}(0, 0.02)$, zero biases, LayerNorm
    $\gamma = 1, \beta = 0$; the two projections that write into the residual
    stream use $\sigma = 0.02 / \sqrt{2 n_{layer}}$.
    """
    torch.manual_seed(0)
    n_layer = 4
    model = GPT(vocab_size=64, n_embd=64, block_size=32, n_head=4, n_layer=n_layer)
    residual_std = 0.02 / math.sqrt(2 * n_layer)

    for name, p in model.named_parameters():
        if ".ln_" in name or name.startswith("ln_f."):
            fill = 1.0 if name.endswith("weight") else 0.0
            assert torch.equal(p, torch.full_like(p, fill)), name
        elif name.endswith("bias"):
            assert torch.count_nonzero(p) == 0, name
        else:
            is_residual = name.endswith(("sa.c_proj.weight", "ffwd.net.2.weight"))
            std = residual_std if is_residual else 0.02
            assert p.std().item() == pytest.approx(std, rel=0.1), name


def test_gpt_kv_cache_consistency():
    r"""
    Verify equivalence between Parallel (Batch) and Serial (Step-by-step) inference.

    $\text{Let } \Phi(x_{0:T}) \to y_{0:T}$ (Full Context)
    $\text{Let } \phi(x_t, S_{t-1}) \to (y_t, S_t)$ (Incremental)

    Verify: $\Phi(x)_{t} \approx \phi(x_t, S_{t-1})$

    This validates the position offset: step $t$ reads the position
    embedding at $S_{pos} = t$.
    """
    B, T, V, C = 1, 5, 64, 16
    H, L = 2, 2

    model = GPT(
        vocab_size=V, n_embd=C, block_size=20, n_layer=L, n_head=H, use_checkpoint=False
    )
    model.eval()

    idx = torch.randint(0, V, (B, T))

    # 1. Full Forward (Ground Truth)
    with torch.no_grad():
        logits_full, _ = model(idx)  # (B, T, V)

    # 2. Step-by-Step with KVCache
    kv = KVCache(batch_size=B, num_heads=H, seq_len=20, head_dim=C // H, num_layers=L)

    outputs = []
    for t in range(T):
        idx_step = idx[:, t : t + 1]  # (B, 1)

        with torch.no_grad():
            # forward offsets positions by kv_cache.get_pos() internally
            logits_step, _ = model(idx_step, kv_cache=kv)
            outputs.append(logits_step)

    logits_serial = torch.cat(outputs, dim=1)  # (B, T, V)

    # Check Logits Equivalence
    # $|y_{batch} - y_{serial}| < \epsilon$
    assert_close(logits_full, logits_serial, atol=1e-5, rtol=1e-5)


def test_forward_rejects_positions_beyond_block_size():
    """
    Learned position embeddings exist for block_size positions only — for a
    plain forward and for a KV-cached continuation alike.
    """
    V, C, H, block_size = 64, 16, 2, 8
    model = GPT(vocab_size=V, n_embd=C, block_size=block_size, n_layer=1, n_head=H)
    model.eval()

    with torch.no_grad():
        model(torch.randint(0, V, (1, block_size)))  # exactly at the limit
        with pytest.raises(AssertionError, match="exceed block_size"):
            model(torch.randint(0, V, (1, block_size + 1)))

        kv = KVCache(
            batch_size=1, num_heads=H, seq_len=block_size, head_dim=C // H, num_layers=1
        )
        model(torch.randint(0, V, (1, block_size - 2)), kv_cache=kv)
        with pytest.raises(AssertionError, match="exceed block_size"):
            model(torch.randint(0, V, (1, 3)), kv_cache=kv)


def test_loss_ignores_masked_targets():
    """
    Targets of -100 (the loader's mask for non-target columns) are skipped:
    the loss equals the cross-entropy over the unmasked positions alone.
    """
    torch.manual_seed(0)
    B, T, V = 2, 6, 64
    model = GPT(vocab_size=V, n_embd=16, block_size=12, n_layer=1, n_head=2)
    model.eval()

    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    masked = targets.clone()
    masked[:, ::2] = -100

    with torch.no_grad():
        logits, loss = model(idx, masked)

    keep = masked != -100
    assert_close(loss, F.cross_entropy(logits[keep], targets[keep]))


def test_every_parameter_receives_gradient():
    """
    No dead parameters: train_ddp wraps the model in a plain DDP, which
    requires every parameter to get a gradient on every step.
    """
    torch.manual_seed(0)
    B, T, V = 2, 6, 64
    model = GPT(vocab_size=V, n_embd=16, block_size=12, n_layer=2, n_head=2)

    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    _, loss = model(idx, targets)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.norm(p.grad) > 0, name


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


def test_generate_slides_window_past_block_size():
    """
    generate() conditions on the last block_size tokens only, so it can run
    past block_size without leaving the learned position range.
    """
    B, T, V, N_new = 1, 5, 64, 10
    model = GPT(vocab_size=V, n_embd=32, block_size=8, n_layer=1)
    model.eval()

    idx = torch.randint(0, V, (B, T))
    out_idx = model.generate(idx, max_new_tokens=N_new, sampler=None)

    assert out_idx.shape == (B, T + N_new)
    assert torch.equal(out_idx[:, :T], idx)


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


def test_vocab_size_must_be_aligned_to_pad():
    """GPT asserts the caller supplies a pre-padded vocab_size."""
    with pytest.raises(AssertionError, match="multiple of"):
        GPT(vocab_size=70, n_embd=16, block_size=8, n_layer=1, n_head=2)

    # pad=1 disables the check (any vocab_size accepted)
    GPT(
        vocab_size=70, n_embd=16, block_size=8, n_layer=1, n_head=2, pad_vocab_size_to=1
    )


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
