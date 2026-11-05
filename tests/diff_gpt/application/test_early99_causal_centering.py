"""Independent equations, causal support and cache coverage for causal CSA."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.kv_cache import KVCache
from diff_gpt.domain.model.rope import apply_rope
from research.early99.experiments.causal_centering import install_causal_centering
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)


def prepared(*, dtype: torch.dtype = torch.float32) -> GPT:
    """Build the exact parent candidate for independent mechanism checks."""
    torch.manual_seed(23)
    model = GPT(
        vocab_size=13,
        n_embd=16,
        n_head=2,
        n_layer=2,
        block_size=16,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    ).to(dtype=dtype)
    install_peri_rms(model)
    install_embedding_rms(model)
    install_residual_scaling(model)
    return model


@pytest.mark.parametrize("clipping_epsilon", [1e-12, 0.25])
def test_dense_matches_independent_zero_sum_matrix(clipping_epsilon: float) -> None:
    """Match an explicit masked matrix including the clipped own-value region."""
    model = prepared(dtype=torch.float64)
    parent = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    install_causal_centering(model)
    assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert_close(model.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert model.token_embedding_table.weight is model.lm_head.weight
    att = model.blocks[0].sa
    # Raising the test threshold exercises clipping with a measurable signal,
    # while keeping Q/K scores nonuniform. Scientific runs keep epsilon 1e-12.
    att.xsa_epsilon = clipping_epsilon
    values = torch.randn(2, 7, 16, dtype=torch.float64)
    cos, sin = model.rope_cos[:7], model.rope_sin[:7]
    # Reference uses the explicit full attention matrix and no centering anchor.
    q, k, v = (
        part.reshape(2, 7, 2, 8).transpose(1, 2)
        for part in F.linear(values, att.c_attn.weight).chunk(3, dim=-1)
    )
    eps = torch.finfo(torch.float64).eps
    q, k = F.rms_norm(q, (8,), eps=eps), F.rms_norm(k, (8,), eps=eps)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    weights = (q @ k.transpose(-2, -1)) * (att.log_alpha.exp().view(1, 2, 1, 1) / 8)
    allowed = torch.ones(7, 7, dtype=torch.bool).tril()
    weights = weights.masked_fill(~allowed, -torch.inf).softmax(-1)
    uniform = allowed.to(torch.float64) / allowed.sum(-1, keepdim=True)
    centered_weights = weights - uniform
    assert_close(
        centered_weights.sum(-1),
        torch.zeros(2, 2, 7, dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )
    attended = centered_weights @ v
    unit = v / v.norm(dim=-1, keepdim=True).clamp_min(att.xsa_epsilon)
    attended = attended - (attended * unit).sum(-1, keepdim=True) * unit
    raw = F.linear(attended.transpose(1, 2).reshape(2, 7, 16), att.c_proj.weight)
    reference = F.rms_norm(raw, (16,), eps=eps) * 0.5
    reference[:, 0] = 0
    assert_close(att(values, cos, sin), reference, rtol=1e-9, atol=1e-10)


def test_attention_has_no_future_gradient_and_annihilates_constants() -> None:
    """Future values have zero influence and constant values cancel exactly."""
    model = prepared(dtype=torch.float64)
    install_causal_centering(model)
    att = model.blocks[0].sa
    values = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    output = att(values, model.rope_cos[:6], model.rope_sin[:6])
    (grad,) = torch.autograd.grad(output[:, 2, 0].sum(), values)
    assert torch.count_nonzero(grad[:, :3]).item() > 0
    assert torch.count_nonzero(grad[:, 3:]).item() == 0
    constant = torch.randn(2, 1, 16, dtype=torch.float64).expand(-1, 6, -1)
    result = att(constant, model.rope_cos[:6], model.rope_sin[:6])
    assert torch.count_nonzero(result).item() == 0


def cache(batch: int) -> KVCache:
    """Allocate cache metadata for the fixed small test model."""
    return KVCache(batch_size=batch, num_heads=2, seq_len=16, head_dim=8, num_layers=2)


def test_dense_token_and_chunk_caches_and_prefix_causality() -> None:
    """Every execution path uses the same visible-prefix mean."""
    model = prepared().eval()
    install_causal_centering(model)
    tokens = torch.tensor([[2, 2, 2, 4, 8, 1, 3], [3, 3, 7, 2, 6, 0, 4]])
    with torch.no_grad():
        dense = model(tokens)[0]
        state = cache(2)
        incremental = torch.cat(
            [model(tokens[:, i : i + 1], kv_cache=state)[0] for i in range(7)], dim=1
        )
        assert_close(incremental, dense, rtol=2e-5, atol=3e-6)
        state = cache(2)
        chunked = torch.cat(
            [
                model(tokens[:, :2], kv_cache=state)[0],
                model(tokens[:, 2:5], kv_cache=state)[0],
                model(tokens[:, 5:], kv_cache=state)[0],
            ],
            dim=1,
        )
        assert_close(chunked, dense, rtol=2e-5, atol=3e-6)
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 13
        assert_close(model(changed)[0][:, :4], dense[:, :4], rtol=0, atol=0)


def test_shared_prefix_mean_includes_prompt_and_private_suffix() -> None:
    """Shared prompt groups contribute their own prefix sums and counts."""
    model = prepared().eval()
    install_causal_centering(model)
    prompts = torch.tensor([[2, 2, 4], [3, 7, 1]])
    suffix = torch.tensor([[5, 6], [8, 9], [2, 4], [6, 8]])
    with torch.no_grad():
        prefix = cache(2)
        model(prompts, kv_cache=prefix)
        shared = cache(4)
        shared.share_prefix(prefix)
        actual = torch.cat(
            [model(suffix[:, i : i + 1], kv_cache=shared)[0] for i in range(2)], dim=1
        )
        full = torch.cat([prompts.repeat_interleave(2, dim=0), suffix], dim=1)
        expected = model(full)[0][:, 3:]
        assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        assert shared.get_pos() == full.size(1)
        assert prefix.get_pos() == prompts.size(1)
