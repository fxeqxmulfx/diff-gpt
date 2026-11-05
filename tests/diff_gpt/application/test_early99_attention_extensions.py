"""Independent equations and cache checks for SSMax and shared LeRoPE."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.kv_cache import KVCache
from diff_gpt.domain.model.rope import apply_rope
from research.early99.experiments.learned_rope import LearnedRoPEGPT
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)
from research.early99.experiments.scalable_softmax import install_scalable_softmax


def prepared(dtype: torch.dtype = torch.float32) -> GPT:
    """Build the fixed parent without drawing alternative initial tensors."""
    torch.manual_seed(23)
    model = GPT(
        13,
        16,
        block_size=16,
        n_head=2,
        n_layer=2,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    ).to(dtype=dtype)
    install_peri_rms(model)
    install_embedding_rms(model)
    install_residual_scaling(model)
    return model


def cache(batch: int) -> KVCache:
    """Allocate the fixed small test model cache."""
    return KVCache(batch_size=batch, num_heads=2, seq_len=16, head_dim=8, num_layers=2)


@pytest.mark.parametrize(
    ("qk_norm", "learned"), [(True, False), (False, False), (True, True)]
)
@pytest.mark.parametrize("clipping_epsilon", [1e-12, 0.25])
def test_ssmax_matches_independent_masked_logits_and_gradients(
    clipping_epsilon: float, *, qk_norm: bool, learned: bool
) -> None:
    """Check explicit masked SSMax/XSA equations and their input gradients."""
    model = prepared(torch.float64)
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    install_scalable_softmax(model, normalize_qk=qk_norm)
    assert_close(model.state_dict(), state, rtol=0, atol=0)
    assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert model.token_embedding_table.weight is model.lm_head.weight
    att = model.blocks[0].sa
    att.xsa_epsilon = clipping_epsilon
    value = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    if learned:
        wrapped = LearnedRoPEGPT(model)
        with torch.no_grad():
            wrapped.lerope.log_scales.copy_(torch.tensor([0.2, -0.1, 0.6, 0.3]))
        cos, sin = wrapped.lerope(0, 7)
    else:
        cos, sin = model.rope_cos[:7], model.rope_sin[:7]
    q, k, v = [
        part.reshape(2, 7, 2, 8).transpose(1, 2)
        for part in F.linear(value, att.c_attn.weight).chunk(3, -1)
    ]
    eps = torch.finfo(torch.float64).eps
    if qk_norm:
        q, k = F.rms_norm(q, (8,), eps=eps), F.rms_norm(k, (8,), eps=eps)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = (q @ k.transpose(-2, -1)) * (att.log_alpha.exp().view(1, 2, 1, 1) / 8)
    scores = scores * torch.arange(1, 8, dtype=torch.float64).log().view(1, 1, 7, 1)
    allowed = torch.ones(7, 7, dtype=torch.bool).tril()
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    assert_close(weights.sum(-1), torch.ones(2, 2, 7, dtype=torch.float64))
    attended = weights @ v
    unit = v / v.norm(dim=-1, keepdim=True).clamp_min(clipping_epsilon)
    exclusive = attended - (attended * unit).sum(-1, keepdim=True) * unit
    raw = F.linear(exclusive.transpose(1, 2).reshape(2, 7, 16), att.c_proj.weight)
    expected = F.rms_norm(raw, (16,), eps=eps) * 0.5
    expected[:, 0] = 0
    actual = att(value, cos, sin)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[:, 0]) == 0
    assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    probe = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * probe).sum(), value, retain_graph=True)[0]
    ge = torch.autograd.grad((expected * probe).sum(), value)[0]
    assert_close(ga, ge, rtol=1e-7, atol=1e-8)
    single = att(value[:, :1], cos[:1], sin[:1])
    assert torch.isfinite(single).all()
    assert torch.count_nonzero(single) == 0


def adapted(kind: str) -> GPT | LearnedRoPEGPT:
    """Set nonzero learned frequencies before testing cached execution."""
    model = prepared().eval()
    if "lerope" in kind:
        if kind == "ssmax-lerope":
            install_scalable_softmax(model)
        model = LearnedRoPEGPT(model)
        with torch.no_grad():
            model.lerope.log_scales.copy_(torch.tensor([0.2, -0.1, 0.6, 0.3]))
    else:
        install_scalable_softmax(model, normalize_qk=kind == "ssmax")
    return model


@pytest.mark.parametrize("kind", ["ssmax", "ssmax-noqk", "lerope", "ssmax-lerope"])
def test_caches_positions_and_future_causality(kind: str) -> None:
    """Check global positions, causal support and selected readout rows."""
    model = adapted(kind)
    tokens = torch.tensor([[2, 2, 2, 4, 8, 1, 3], [3, 3, 7, 2, 6, 0, 4]])
    with torch.no_grad():
        dense = model(tokens)[0]
        state = cache(2)
        tokenwise = torch.cat(
            [model(tokens[:, i : i + 1], kv_cache=state)[0] for i in range(7)], dim=1
        )
        state = cache(2)
        chunked = torch.cat(
            [
                model(tokens[:, a:b], kv_cache=state)[0]
                for a, b in [(0, 2), (2, 5), (5, 7)]
            ],
            dim=1,
        )
        assert_close(tokenwise, dense, rtol=3e-5, atol=4e-6)
        assert_close(chunked, dense, rtol=3e-5, atol=4e-6)
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 13
        assert_close(model(changed)[0][:, :4], dense[:, :4], rtol=0, atol=0)
        positions = torch.tensor([[1, 5], [2, 4]])
        assert_close(
            model(tokens, positions=positions)[0],
            dense.gather(1, positions[..., None].expand(-1, -1, 13)),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("kind", ["ssmax", "ssmax-noqk", "lerope", "ssmax-lerope"])
def test_shared_prefix_includes_global_position(kind: str) -> None:
    """Shared prompt groups use their full global positions and key counts."""
    model = adapted(kind)
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
        assert_close(actual, model(full)[0][:, 3:], rtol=3e-5, atol=4e-6)
        assert shared.get_pos() == full.size(1)
        assert prefix.get_pos() == prompts.size(1)


@pytest.mark.parametrize("ssmax", [False, True])
def test_lerope_initial_weights_outputs_and_frequency_gradient(*, ssmax: bool) -> None:
    """Preserve parent initialization and check learned-frequency derivatives."""
    parent = prepared()
    if ssmax:
        install_scalable_softmax(parent)
    rng = torch.get_rng_state().clone()
    model = LearnedRoPEGPT(parent)
    assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert model.token_embedding_table.weight is model.lm_head.weight
    state = model.state_dict()
    assert set(state) - set(parent.state_dict()) == {"lerope.log_scales"}
    for name, value in parent.state_dict().items():
        assert_close(state[name], value, rtol=0, atol=0)
    tokens = torch.tensor([[2, 4, 8, 1, 3], [3, 7, 2, 6, 4]])
    assert_close(model(tokens)[0], parent(tokens)[0], rtol=0, atol=0)
    model.double()
    output = model(tokens)[0][:, -1, 3].sum()
    gradient = torch.autograd.grad(output, model.lerope.log_scales)[0]
    minimum_signal = 1e-6
    assert gradient.norm() > minimum_signal
    reference = []
    for i in range(4):
        with torch.no_grad():
            model.lerope.log_scales[i] = 1e-5
            plus = model(tokens)[0][:, -1, 3].sum()
            model.lerope.log_scales[i] = -1e-5
            minus = model(tokens)[0][:, -1, 3].sum()
            model.lerope.log_scales[i] = 0
        reference.append((plus - minus) / 2e-5)
    assert_close(gradient, torch.stack(reference), rtol=1e-5, atol=1e-8)


def test_learned_rotations_preserve_norm_and_relative_scores() -> None:
    """Arbitrary learned angles preserve the rotary identities numerically."""
    model = adapted("lerope").double()
    q, k = torch.randn(8, dtype=torch.float64), torch.randn(8, dtype=torch.float64)
    cos, sin = model.lerope(0, 9)
    rotated_q = apply_rope(q, cos[3], sin[3])
    rotated_k = apply_rope(k, cos[8], sin[8])
    assert_close(rotated_q.norm(), q.norm(), rtol=1e-14, atol=1e-14)
    # Base angles were initially rounded in float32; derive the offset from the
    # same frequencies, rather than requiring those rounded integer multiples.
    angles = (
        model.lerope.base_angles[8] - model.lerope.base_angles[3]
    ) * model.lerope.log_scales.exp()
    shifted_k = apply_rope(k, angles.cos(), angles.sin())
    assert_close(rotated_q @ rotated_k, q @ shifted_k, rtol=1e-14, atol=1e-14)
