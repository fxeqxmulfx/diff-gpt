import torch
from torch.testing import assert_close

from diff_gpt.model.gpt import GPT
from diff_gpt.model.kv_cache import KVCache


def test_gpt_initialization_and_shapes():
    r"""
    Verify structural constraints.
    1. Weight Tying: $W_{emb} \equiv W_{head}$
    2. Output Shape: $x \in \mathbb{Z}^{B \times T} \implies \text{logits} \in \mathbb{R}^{B \times T \times V}$
    """
    B, T, V, C = 2, 10, 100, 32

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
    B, T, V, C = 1, 5, 50, 16
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
    B, T, V = 2, 8, 100

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
    B, T, V, N_new = 1, 5, 50, 3
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


def test_overfitting_capability():
    r"""
    Sanity check: Model capacity.
    $\lim_{i \to \infty} \mathcal{L}(\theta_i) \to 0$ on a single batch.
    """
    B, T, V = 1, 4, 20
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
