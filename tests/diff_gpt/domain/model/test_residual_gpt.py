"""Audit the canonical scaled residual model and complete saved states."""

import copy

import pytest
import torch
from torch.nn import functional as F

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.kv_cache import KVCache
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

XSA_CLIP = 1e-12


def make(*, checkpointing: bool = False) -> GPT:
    """Construct identical common parameters for every tested variant."""
    torch.manual_seed(145)
    torch.set_num_threads(1)
    return GPT(
        11, 8, 8, 2, 2, use_checkpoint=checkpointing, pad_vocab_size_to=1
    ).double()


def manual(model: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """Evaluate scaled residuals and dense causal XSA from parameter matrices."""
    eps = torch.finfo(model.token_embedding_table.weight.dtype).eps

    def norm(value: torch.Tensor, epsilon: float = eps) -> torch.Tensor:
        return value / (value.square().mean(-1, keepdim=True) + epsilon).sqrt()

    value = norm(F.embedding(tokens, model.token_embedding_table.weight))
    still_constant = (tokens != tokens[:, :1]).cumsum(1) == 0
    length = tokens.shape[1]
    for index in range(2 * model.n_layer):
        block = model.blocks[index // 2]
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
        value = value + model.branch_scale * branch
    return F.linear(norm(value), model.lm_head.weight)


def test_full_equations_and_every_parameter_gradient() -> None:
    """Include constant prefixes, arbitrary suffixes and all depth-query parameters."""
    actual = make()
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


def test_causality_chunked_cache_and_checkpointing() -> None:
    """Depth sources stay local to a token; source K/V cache positions still advance."""
    model = make()
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


def test_complete_schedule_free_state_resume() -> None:
    """Restoring all optimizer state gives the same subsequent model and moments."""
    model = make()
    optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.003, betas=(0.9, 0.98))
    tokens = torch.tensor([[1, 2], [2, 3], [4, 4]])
    labels = torch.tensor([5, 7, 9])

    def steps(net: GPT, opt: ScheduleFreeAdamW, count: int) -> None:
        for _ in range(count):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(tokens)[0][:, -1], labels).backward()
            opt.step()

    steps(model, optimizer, 5)
    saved = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    restored = make()
    resumed = ScheduleFreeAdamW(restored.parameters())
    restored.load_state_dict(saved[0])
    resumed.load_state_dict(saved[1])
    steps(model, optimizer, 7)
    steps(restored, resumed, 7)
    torch.testing.assert_close(
        model.state_dict(), restored.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        optimizer.state_dict(), resumed.state_dict(), rtol=0, atol=0
    )


@pytest.mark.parametrize("option", ["mixing", "centered_xsa", "rope_layers"])
def test_removed_architecture_options_are_rejected(option: str) -> None:
    """Historical options must fail explicitly, never silently select another model."""
    with pytest.raises(TypeError, match="unexpected keyword"):
        GPT(64, **{option: None})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compiled_canonical_model_matches_values_and_gradients(device: str) -> None:
    """The current model compiles as one graph, including its guarded backward."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = make(checkpointing=True).float().to(device)
    reference = copy.deepcopy(model)
    compiled = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
    tokens = torch.tensor([[1, 1, 2], [3, 3, 3]], device=device)
    actual, expected = compiled(tokens)[0], reference(tokens)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for parameter, other in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-4, atol=2e-6)
