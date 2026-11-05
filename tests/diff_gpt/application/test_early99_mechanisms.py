"""Independent checks for the isolated cosine and Peri-RMS experiments."""

import copy

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.kv_cache import KVCache
from research.current_grokking.protocol import Case
from research.early99.experiments.factories import Configuration
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)


def test_cosine_uses_actual_exposure_after_warmup_and_holds_floor() -> None:
    """A short final batch must not shift the halfway or final cosine rate."""
    case = Case("schedule", "arithmetic", {}, batch=8, warmup=1, learning_rate=0.01)
    configuration = Configuration(
        "cosine", 3, schedule="cosine", decay_epochs=2, minimum_lr_ratio=0.5
    )
    # Train size 21: issued batch sizes are 8, 8, 5. One warmup update
    # consumes 8/21 epochs; updates 4 and 7 are exactly 1 and 2 epochs later.
    for updates, expected in [(0, 0), (1, 0.03), (4, 0.0225), (7, 0.015), (10, 0.015)]:
        assert configuration.learning_rate_at(case, updates, 21) == pytest.approx(
            expected
        )


def test_peri_rms_preserves_weights_rng_and_matches_branch_equations() -> None:
    """Check new norm placement against two independent functional updates."""
    torch.manual_seed(11)
    base = GPT(
        vocab_size=11,
        n_embd=16,
        n_head=2,
        n_layer=2,
        block_size=8,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    )
    candidate = copy.deepcopy(base)
    rng = torch.get_rng_state().clone()
    install_peri_rms(candidate)
    assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert_close(candidate.state_dict(), base.state_dict(), rtol=0, atol=0)
    assert [n for n, _ in candidate.named_parameters()] == [
        n for n, _ in base.named_parameters()
    ]
    value = torch.randn(2, 4, 16)
    cos, sin = base.rope_cos[:4], base.rope_sin[:4]
    block = base.blocks[0]
    expected = value + F.rms_norm(
        block.sa(block.ln_1(value), cos, sin), (16,), eps=torch.finfo(value.dtype).eps
    )
    expected = expected + F.rms_norm(
        block.ffwd(block.ln_2(expected)), (16,), eps=torch.finfo(value.dtype).eps
    )
    actual = candidate.blocks[0](value, cos, sin)
    assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters()
    )


@pytest.mark.parametrize(
    "model_variant", ["peri-rms", "peri-rms-embedding", "peri-rms-embedding-scaled"]
)
def test_peri_rms_keeps_prefix_causality_and_cached_generation(
    model_variant: str,
) -> None:
    """Dense and cached generation must agree through repeated-prefix tokens."""
    torch.manual_seed(7)
    model = GPT(
        vocab_size=11,
        n_embd=16,
        n_head=2,
        n_layer=2,
        block_size=8,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    )
    install_peri_rms(model)
    if model_variant != "peri-rms":
        install_embedding_rms(model)
    if model_variant == "peri-rms-embedding-scaled":
        install_residual_scaling(model)
    model.eval()
    tokens = torch.tensor([[2, 2, 2, 4, 8, 1], [3, 3, 7, 2, 6, 0]])
    cache = KVCache(batch_size=2, num_heads=2, seq_len=8, head_dim=8, num_layers=2)
    with torch.no_grad():
        dense, _ = model(tokens)
        cached = torch.cat(
            [model(tokens[:, i : i + 1], kv_cache=cache)[0] for i in range(6)], dim=1
        )
        prefix, _ = model(tokens[:, :3])
    assert_close(dense[:, :3], prefix, rtol=1e-5, atol=1e-5)
    assert_close(cached, dense, rtol=1e-4, atol=1e-5)


def test_embedding_rms_preserves_tying_rng_and_matches_functional_reference() -> None:
    """An added embedding norm must not replace or redraw the shared weight."""
    torch.manual_seed(5)
    model = GPT(
        vocab_size=11,
        n_embd=16,
        n_head=2,
        n_layer=2,
        block_size=8,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    )
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    install_peri_rms(model)
    install_embedding_rms(model)
    assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert_close(model.state_dict(), original, rtol=0, atol=0)
    assert model.token_embedding_table.weight is model.lm_head.weight
    tokens = torch.tensor([[2, 2, 4, 8]])
    expected = F.rms_norm(
        F.embedding(tokens, model.lm_head.weight),
        (16,),
        eps=torch.finfo(torch.float32).eps,
    )
    assert_close(model.token_embedding_table(tokens), expected, rtol=0, atol=0)


def test_depth_coefficient_follows_both_branch_norms() -> None:
    """Scaling before RMS must not silently replace the declared residual update."""
    torch.manual_seed(19)
    base = GPT(
        vocab_size=11,
        n_embd=16,
        n_head=2,
        n_layer=2,
        block_size=8,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    )
    install_peri_rms(base)
    install_embedding_rms(base)
    candidate = copy.deepcopy(base)
    rng = torch.get_rng_state().clone()
    install_residual_scaling(candidate)
    assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert_close(candidate.state_dict(), base.state_dict(), rtol=0, atol=0)
    value = torch.randn(2, 4, 16)
    cos, sin = base.rope_cos[:4], base.rope_sin[:4]
    block = base.blocks[0]
    scale = 0.5  # 1/sqrt(2L), L=2.
    expected = value + scale * block.sa(block.ln_1(value), cos, sin)
    expected = expected + scale * block.ffwd(block.ln_2(expected))
    assert_close(candidate.blocks[0](value, cos, sin), expected, rtol=0, atol=0)
