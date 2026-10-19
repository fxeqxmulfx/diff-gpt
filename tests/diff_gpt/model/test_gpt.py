# Copyright 2025 Mikhail Borisov
"""Tests for the GPT model: init, forward, KV cache, loss and generation."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.testing import assert_close
from torch.utils.flop_counter import FlopCounterMode

from diff_gpt.model.gpt import GPT
from diff_gpt.model.kv_cache import KVCache
from diff_gpt.sampler.temperature import TemperatureSampler

# F.cross_entropy's ignore_index; the loader masks non-target columns with it.
IGNORE_INDEX = -100
# Loss the model must reach by overfitting a single batch.
MAX_OVERFIT_LOSS = 0.1


def test_gpt_initialization_and_shapes() -> None:
    r"""
    Verify structural constraints.

    1. Weight Tying: $W_{emb} \equiv W_{head}$
    2. Output Shape: $x \in \mathbb{Z}^{B \times T} \implies
       \text{logits} \in \mathbb{R}^{B \times T \times V}$
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


def test_gpt_init() -> None:
    r"""
    Check the initialization of every parameter.

    Init: weights $\sim \mathcal{N}(0, 0.02)$, LayerNorm $\gamma = 1$;
    the two projections that write into the residual stream use
    $\sigma = 0.02 / \sqrt{2 n_{layer}}$.
    """
    torch.manual_seed(0)
    n_layer = 4
    model = GPT(vocab_size=64, n_embd=64, block_size=32, n_head=4, n_layer=n_layer)
    residual_std = 0.02 / math.sqrt(2 * n_layer)

    for name, p in model.named_parameters():
        if ".ln_" in name or name.startswith("ln_f."):
            assert torch.equal(p, torch.ones_like(p)), name
        else:
            is_residual = name.endswith(("sa.c_proj.weight", "ffwd.net.2.weight"))
            std = residual_std if is_residual else 0.02
            assert p.std().item() == pytest.approx(std, rel=0.1), name


def test_model_has_no_biases() -> None:
    """Neither Linear nor LayerNorm layers carry a bias: every parameter is a weight."""
    model = GPT(vocab_size=64, n_embd=16, block_size=8, n_head=2, n_layer=2)
    names = [name for name, _ in model.named_parameters()]
    assert names
    assert all(name.endswith("weight") for name in names), names


def test_estimate_flops_matches_flop_counter() -> None:
    """
    estimate_flops() agrees with PyTorch's FLOP counter.

    The count covers one forward and backward pass. The math SDPA backend runs
    attention as plain matmuls the counter can see; the estimate's only surplus
    is 6 per LayerNorm gain.
    """
    torch.manual_seed(0)
    B, V, T = 2, 64, 16
    model = GPT(
        vocab_size=V,
        n_embd=32,
        block_size=T,
        n_head=2,
        n_layer=2,
        use_checkpoint=False,
    )
    idx = torch.randint(0, V, (B, T))
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        _, loss = model(idx, idx)
        loss.backward()

    expected = model.estimate_flops() * B * T
    assert counter.get_total_flops() == pytest.approx(expected, rel=0.01)


def test_gpt_kv_cache_consistency() -> None:
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
        vocab_size=V,
        n_embd=C,
        block_size=20,
        n_layer=L,
        n_head=H,
        use_checkpoint=False,
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


def test_forward_rejects_positions_beyond_block_size() -> None:
    """
    Check that forward rejects positions beyond block_size.

    Learned position embeddings exist for block_size positions only — for a
    plain forward and for a KV-cached continuation alike.
    """
    V, C, H, block_size = 64, 16, 2, 8
    model = GPT(vocab_size=V, n_embd=C, block_size=block_size, n_layer=1, n_head=H)
    model.eval()

    with torch.no_grad():
        model(torch.randint(0, V, (1, block_size)))  # exactly at the limit
        with pytest.raises(ValueError, match="exceed block_size"):
            model(torch.randint(0, V, (1, block_size + 1)))

        kv = KVCache(
            batch_size=1,
            num_heads=H,
            seq_len=block_size,
            head_dim=C // H,
            num_layers=1,
        )
        model(torch.randint(0, V, (1, block_size - 2)), kv_cache=kv)
        with pytest.raises(ValueError, match="exceed block_size"):
            model(torch.randint(0, V, (1, 3)), kv_cache=kv)


def test_loss_ignores_masked_targets() -> None:
    """
    Check that the loss skips targets of -100.

    The loader masks non-target columns with -100; the loss equals the
    cross-entropy over the unmasked positions alone.
    """
    torch.manual_seed(0)
    B, T, V = 2, 6, 64
    model = GPT(vocab_size=V, n_embd=16, block_size=12, n_layer=1, n_head=2)
    model.eval()

    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    masked = targets.clone()
    masked[:, ::2] = IGNORE_INDEX

    with torch.no_grad():
        logits, loss = model(idx, masked)

    keep = masked != IGNORE_INDEX
    assert_close(loss, F.cross_entropy(logits[keep], targets[keep]))


def test_every_parameter_receives_gradient() -> None:
    """
    Check that every parameter receives a nonzero gradient.

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


def test_gradient_checkpointing_flow() -> None:
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

    _, loss = model(idx, targets)

    loss.backward()

    # Verify Gradients on Input Embeddings (Proxy for full backward flow)
    assert model.token_embedding_table.weight.grad is not None
    assert torch.norm(model.token_embedding_table.weight.grad) > 0


def test_generation_output_len() -> None:
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


def test_generate_slides_window_past_block_size() -> None:
    """
    Check that generate() slides its context window to run past block_size.

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


def test_generate_seed_determinism() -> None:
    """Passing the same `seed` must yield identical outputs; different seeds differ."""
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


def test_vocab_size_must_be_aligned_to_pad() -> None:
    """GPT requires the caller to supply a pre-padded vocab_size."""
    with pytest.raises(ValueError, match="multiple of"):
        GPT(vocab_size=70, n_embd=16, block_size=8, n_layer=1, n_head=2)

    # pad=1 disables the check (any vocab_size accepted)
    GPT(
        vocab_size=70,
        n_embd=16,
        block_size=8,
        n_layer=1,
        n_head=2,
        pad_vocab_size_to=1,
    )


def test_overfitting_capability() -> None:
    r"""
    Sanity-check the model capacity.

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

    assert loss.item() < MAX_OVERFIT_LOSS


def test_forward_takes_non_contiguous_targets() -> None:
    """
    Check that forward accepts targets sliced out of one tensor with the inputs.

    `targets.view(B * T)` raised on the non-contiguous slice x[:, 1:].
    """
    torch.manual_seed(0)
    model = GPT(vocab_size=64, n_embd=16, block_size=16, n_head=2, n_layer=1)
    x = torch.randint(0, 64, (2, 9))
    _, loss = model(x[:, :-1], x[:, 1:])
    _, loss_contiguous = model(x[:, :-1], x[:, 1:].contiguous())
    assert loss is not None
    assert torch.equal(loss, loss_contiguous)


def test_gpt_rejects_n_layer_below_one() -> None:
    """
    Check that a GPT without blocks is rejected.

    Without blocks no insert_kv advances the KV cache, so cached logits
    went wrong and Engine.generate failed.
    """
    with pytest.raises(ValueError, match="n_layer must be >= 1"):
        GPT(vocab_size=64, n_embd=16, block_size=16, n_head=2, n_layer=0)


def test_device_keeps_the_index() -> None:
    """Check that device names the weights' device, index included."""
    model = GPT(vocab_size=64, n_embd=16, block_size=16, n_head=2, n_layer=1)
    assert model.device == torch.device("cpu")
    if torch.cuda.is_available():
        assert model.to("cuda").device == torch.device("cuda", 0)
