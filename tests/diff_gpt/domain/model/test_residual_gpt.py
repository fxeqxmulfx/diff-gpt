"""Audit full depth mixing against explicit equations and complete saved states."""

import copy
import math

import pytest
import torch
from torch.nn import functional as F

from diff_gpt.application.transformer_conditioning import TransformerQueryConditioner
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.kv_cache import KVCache
from diff_gpt.domain.model.residual_gpt import MODES, ResidualGPT
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

XSA_CLIP = 1e-12


def make(mode: str, *, checkpointing: bool = False) -> ResidualGPT:
    """Construct identical common parameters for every tested variant."""
    torch.manual_seed(145)
    torch.set_num_threads(1)
    return ResidualGPT(
        11, 8, 8, 2, 2, mixing=mode, use_checkpoint=checkpointing, pad_vocab_size_to=1
    ).double()


def manual(model: ResidualGPT, tokens: torch.Tensor) -> torch.Tensor:
    """Evaluate paper depth mixing and dense causal XSA from parameter matrices."""
    width = model.n_embd
    eps = torch.finfo(model.token_embedding_table.weight.dtype).eps

    def norm(value: torch.Tensor, epsilon: float = eps) -> torch.Tensor:
        return value / (value.square().mean(-1, keepdim=True) + epsilon).sqrt()

    def mix(sources: list[torch.Tensor], index: int) -> torch.Tensor:
        if index == 0:
            return sources[0]
        mixer = model.residual_mixers[index - 1]
        query = mixer.query
        values = torch.stack(sources, -2)
        keys = norm(values, mixer.eps)
        if model.mixing == "attnres":
            scores = (keys * mixer.key_gain * query).sum(-1)
        else:
            scale = getattr(mixer, "query_scale", 1.0)
            physical = query * scale
            scores = (keys * norm(physical, mixer.eps)).sum(-1) / math.sqrt(width)
        return (scores.softmax(-1)[..., None] * values).sum(-2)

    value = norm(F.embedding(tokens, model.token_embedding_table.weight))
    sources = [value]
    still_constant = (tokens != tokens[:, :1]).cumsum(1) == 0
    length = tokens.shape[1]
    for index in range(2 * model.n_layer):
        block = model.blocks[index // 2]
        if model.mixing in ("attnres", "qk", "conditioned_qk"):
            value = mix(sources, index)
        if index % 2 == 0:
            q, k, v = F.linear(norm(value), block.sa.c_attn.weight).chunk(3, -1)
            q, k, v = (
                t.reshape(len(tokens), length, 2, 4).transpose(1, 2) for t in (q, k, v)
            )
            cos, sin = model.rope_cos[:length], model.rope_sin[:length]

            def rotate(
                z: torch.Tensor, cos: torch.Tensor = cos, sin: torch.Tensor = sin
            ) -> torch.Tensor:
                a, b = norm(z).chunk(2, -1)
                return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)

            scores = rotate(q) @ rotate(k).transpose(-1, -2) / 2
            future = ~torch.ones(length, length, dtype=torch.bool).tril()
            weights = scores.masked_fill(future, float("-inf")).softmax(-1)
            # Dense source XSA, independent of the centered implementation.
            average = weights @ v
            own = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            projected = average - (average * own).sum(-1, keepdim=True) * own
            projected = torch.cat(
                (torch.zeros_like(projected[:, :, :1]), projected[:, :, 1:]), 2
            )
            branch = F.linear(
                projected.transpose(1, 2).reshape_as(value), block.sa.c_proj.weight
            )
            valid = (v[:, :, 0].norm(dim=-1) > XSA_CLIP).all(1)
            still_constant = still_constant & valid[:, None]
        else:
            gate, activation = F.linear(norm(value), block.ffwd.c_fc.weight).chunk(
                2, -1
            )
            branch = F.linear(F.silu(gate) * activation, block.ffwd.c_proj.weight)
        branch = norm(branch)
        branch = torch.where(still_constant[..., None], branch[:, :1], branch)
        if model.mixing in ("add", "scaled"):
            value = value + model.branch_scale * branch
        else:
            sources.append(branch)
    if model.mixing in ("attnres", "qk", "conditioned_qk"):
        value = mix(sources, 2 * model.n_layer)
    return F.linear(norm(value), model.lm_head.weight)


@pytest.mark.parametrize("mode", MODES)
def test_common_initialization_and_parameter_counts(mode: str) -> None:
    """Query draws never change the paired source branches or tied readout."""
    model = make(mode)
    torch.manual_seed(145)
    source = GPT(11, 8, 8, 2, 2, use_checkpoint=False, pad_vocab_size_to=1).double()
    for name, parameter in source.named_parameters():
        torch.testing.assert_close(
            dict(model.named_parameters())[name], parameter, rtol=0, atol=0
        )
    extra = sum(p.numel() for p in model.parameters()) - sum(
        p.numel() for p in source.parameters()
    )
    assert (
        extra
        == {"add": 0, "scaled": 0, "attnres": 64, "qk": 32, "conditioned_qk": 32}[mode]
    )
    assert model.lm_head.weight is model.token_embedding_table.weight
    if mode == "attnres":
        for mixer in model.residual_mixers:
            assert torch.equal(mixer.query, torch.zeros_like(mixer.query))
            assert torch.equal(mixer.key_gain, torch.ones_like(mixer.key_gain))
    if mode == "add":
        tokens = torch.tensor([[1, 2, 3], [4, 4, 4]])
        torch.testing.assert_close(model(tokens)[0], source(tokens)[0], rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_full_equations_and_every_parameter_gradient(mode: str) -> None:
    """Include constant prefixes, arbitrary suffixes and all depth-query parameters."""
    actual = make(mode)
    reference = copy.deepcopy(actual)
    tokens = torch.tensor([[1, 2, 3, 4], [2, 2, 2, 5], [3, 3, 3, 3]])
    a = actual(tokens)[0]
    b = manual(reference, tokens)
    torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-10)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, p), (_, q) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert p.grad is not None, name
        assert q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-6, atol=2e-8, msg=name)


@pytest.mark.parametrize("mode", MODES)
def test_causality_chunked_cache_and_checkpointing(mode: str) -> None:
    """Depth sources stay local to a token; source K/V cache positions still advance."""
    model = make(mode)
    tokens = torch.tensor([[1, 1, 2, 3], [2, 2, 2, 4]])
    changed = tokens.clone()
    changed[:, -1] = 7
    expected = model(tokens)[0]
    torch.testing.assert_close(
        model(changed)[0][:, :-1], expected[:, :-1], rtol=0, atol=0
    )
    for chunk in (1, 2):
        cache = KVCache(2, 2, 8, 4, 2)
        with torch.no_grad():
            output = torch.cat(
                [model(part, kv_cache=cache)[0] for part in tokens.split(chunk, dim=1)],
                1,
            )
        assert cache.get_pos() == tokens.shape[1]
        # Source one_query_attention explicitly evaluates softmax in FP32,
        # including for FP64 values. Compare at that arithmetic precision.
        torch.testing.assert_close(
            output,
            expected,
            rtol=8 * torch.finfo(torch.float32).eps,
            atol=2 * torch.finfo(torch.float32).eps,
        )
    recomputed = copy.deepcopy(model)
    recomputed.use_checkpoint = True
    a, b = model(tokens)[0], recomputed(tokens)[0]
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for p, q in zip(model.parameters(), recomputed.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_conditioner_preserves_complete_model_and_saved_continuation() -> None:
    """Rechart several queries and restore state after the adapter already exists."""
    model = make("conditioned_qk")
    with torch.no_grad():
        for index, mixer in enumerate(model.residual_mixers):
            mixer.query.mul_(1e6 if index % 2 else 1.0)
    optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.003, betas=(0.9, 0.98))
    conditioner = TransformerQueryConditioner(model, optimizer)
    tokens = torch.tensor([[1, 2], [2, 3], [4, 4]])
    labels = torch.tensor([5, 7, 9])
    before = model(tokens)[0].detach()
    branches = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not n.startswith("residual_mixers")
    }
    conditioner()
    torch.testing.assert_close(model(tokens)[0], before, rtol=2e-8, atol=2e-10)
    for name, value in branches.items():
        torch.testing.assert_close(
            dict(model.named_parameters())[name], value, rtol=0, atol=0
        )

    def steps(
        net: ResidualGPT,
        opt: ScheduleFreeAdamW,
        change: TransformerQueryConditioner,
        count: int,
    ) -> None:
        for _ in range(count):
            opt.zero_grad(set_to_none=True)
            change()
            F.cross_entropy(net(tokens)[0][:, -1], labels).backward()
            opt.step()

    steps(model, optimizer, conditioner, 5)
    saved = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    restored = make("conditioned_qk")
    resumed = ScheduleFreeAdamW(restored.parameters())
    adapter = TransformerQueryConditioner(restored, resumed)
    restored.load_state_dict(saved[0])
    resumed.load_state_dict(saved[1])
    steps(model, optimizer, conditioner, 7)
    steps(restored, resumed, adapter, 7)
    torch.testing.assert_close(
        model.state_dict(), restored.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        optimizer.state_dict(), resumed.state_dict(), rtol=0, atol=0
    )


def test_invalid_mode_and_conditioner_owner_are_rejected() -> None:
    """Do not silently train a requested architecture with another configuration."""
    with pytest.raises(ValueError, match="Unknown residual mixing"):
        make("invalid")
    a, b = make("conditioned_qk"), make("conditioned_qk")
    with pytest.raises(ValueError, match="own every"):
        TransformerQueryConditioner(a, ScheduleFreeAdamW(b.parameters()))
