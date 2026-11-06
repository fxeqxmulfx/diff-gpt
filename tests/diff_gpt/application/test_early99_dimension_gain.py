"""Inverse-width scores, cached equations, provenance and native update equivalence."""

import copy
from typing import TYPE_CHECKING

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.kv_cache import KVCache
from diff_gpt.domain.model.rope import apply_rope
from research.current_grokking.data import arithmetic
from research.current_grokking.protocol import Case
from research.current_grokking.train import Stepper
from research.early99.experiments.factories import FactoryStepper
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)
from research.early99.experiments.scalable_softmax import install_scalable_softmax
from research.early99.experiments.trainer import run as original_train
from research.early99_dimension_gain.factories import (
    Configuration,
    InverseWidthAttention,
    configured_factories,
    install_inverse_width,
)
from research.early99_dimension_gain.run import prepare_output, train

if TYPE_CHECKING:
    from pathlib import Path

UNCHANGED_WIDTH = 16


def prepared(head_width: int, dtype: torch.dtype = torch.float32) -> GPT:
    """Install only the score wrapper on matched parent tensors."""
    torch.manual_seed(23)
    model = GPT(
        13,
        2 * head_width,
        block_size=16,
        n_head=2,
        n_layer=2,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    ).to(dtype=dtype)
    install_peri_rms(model)
    install_embedding_rms(model)
    install_residual_scaling(model)
    install_scalable_softmax(model)
    state, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    parameters = dict(model.named_parameters())
    install_inverse_width(model)
    assert_close(model.state_dict(), state, rtol=0, atol=0)
    assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert model.token_embedding_table.weight is model.lm_head.weight
    assert all(p is dict(model.named_parameters())[n] for n, p in parameters.items())
    for name, value in buffers.items():
        assert_close(dict(model.named_buffers())[name], value, rtol=0, atol=0)
    assert all(block.sa.use_qk_norm for block in model.blocks)
    assert all(
        torch.equal(
            block.sa.attention_gain(),
            torch.full((1, 2, 1, 1), 1 / head_width, dtype=dtype),
        )
        for block in model.blocks
    )
    return model


def cache(batch: int, head_width: int) -> KVCache:
    """Use a cache matching the small test model."""
    return KVCache(
        batch_size=batch, num_heads=2, seq_len=16, head_dim=head_width, num_layers=2
    )


def test_width_rule_and_rejected_parents() -> None:
    """Keep width-16 controls exact and reject the no-Q/K ablation as a parent."""
    for width in (2, 16, 32, 128):
        attention = prepared(width).blocks[0].sa
        quarter = attention.log_alpha.exp().view(1, 2, 1, 1) / width / 4
        actual = attention.attention_gain()
        if width == UNCHANGED_WIDTH:
            assert torch.equal(actual, quarter)
        elif width < UNCHANGED_WIDTH:
            assert torch.all(actual > quarter)
        else:
            assert torch.all(actual < quarter)
    with pytest.raises(ValueError, match="Only Q/K-normalized"):
        Configuration(model_variant="peri-rms-embedding-scaled-ssmax-noqk")
    with pytest.raises(ValueError, match="Only Q/K-normalized"):
        Configuration(gain_rule="inverse_sqrt_width")
    attention.use_qk_norm = False
    with pytest.raises(ValueError, match="requires Q/K RMS"):
        InverseWidthAttention(attention)


@pytest.mark.parametrize("head_width", [8, 16, 32])
@pytest.mark.parametrize("clipping_epsilon", [1e-12, 0.25])
def test_independent_scores_outputs_and_all_gradients(
    head_width: int, clipping_epsilon: float
) -> None:
    """Compare explicit SSMax/XSA equations, not another call to the wrapper."""
    model = prepared(head_width, torch.float64)
    att = model.blocks[0].sa
    att.xsa_epsilon = clipping_epsilon
    width = 2 * head_width
    value = torch.randn(2, 7, width, dtype=torch.float64, requires_grad=True)
    cos, sin = model.rope_cos[:7], model.rope_sin[:7]
    q, k, v = [
        part.reshape(2, 7, 2, head_width).transpose(1, 2)
        for part in F.linear(value, att.c_attn.weight).chunk(3, -1)
    ]
    eps = torch.finfo(torch.float64).eps
    q, k = F.rms_norm(q, (head_width,), eps=eps), F.rms_norm(k, (head_width,), eps=eps)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = (q @ k.transpose(-2, -1)) / head_width
    scores = scores * torch.arange(1, 8, dtype=torch.float64).log().view(1, 1, 7, 1)
    allowed = torch.ones(7, 7, dtype=torch.bool).tril()
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    attended = weights @ v
    unit = v / v.norm(dim=-1, keepdim=True).clamp_min(clipping_epsilon)
    exclusive = attended - (attended * unit).sum(-1, keepdim=True) * unit
    raw = F.linear(exclusive.transpose(1, 2).reshape(2, 7, width), att.c_proj.weight)
    expected = F.rms_norm(raw, (width,), eps=eps) * 0.5
    expected[:, 0] = 0
    actual = att(value, cos, sin)
    assert_close(actual, expected, rtol=1e-8, atol=1e-9)
    probe = torch.randn_like(actual)
    inputs = [value, att.c_attn.weight, att.c_proj.weight]
    ga = torch.autograd.grad((actual * probe).sum(), inputs, retain_graph=True)
    ge = torch.autograd.grad((expected * probe).sum(), inputs)
    for first, second in zip(ga, ge, strict=True):
        assert torch.isfinite(first).all()
        assert_close(first, second, rtol=2e-7, atol=2e-8)
    assert torch.count_nonzero(att(value[:, :1], cos[:1], sin[:1])) == 0


@pytest.mark.parametrize("head_width", [8, 16, 32])
def test_causal_chunked_and_shared_cache(head_width: int) -> None:
    """Global counts, guarded prefixes and selected readout remain equivalent."""
    model = prepared(head_width).eval()
    tokens = torch.tensor([[2, 2, 2, 4, 8, 1, 3], [3, 3, 7, 2, 6, 0, 4]])
    with torch.no_grad():
        dense = model(tokens)[0]
        state = cache(2, head_width)
        tokenwise = torch.cat(
            [model(tokens[:, i : i + 1], kv_cache=state)[0] for i in range(7)], dim=1
        )
        state = cache(2, head_width)
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
        prompts = tokens[:, :3]
        suffix = torch.tensor([[5, 6], [8, 9], [2, 4], [6, 8]])
        prefix = cache(2, head_width)
        model(prompts, kv_cache=prefix)
        shared = cache(4, head_width)
        shared.share_prefix(prefix)
        actual = torch.cat(
            [model(suffix[:, i : i + 1], kv_cache=shared)[0] for i in range(2)], dim=1
        )
        full = torch.cat([prompts.repeat_interleave(2, dim=0), suffix], dim=1)
        assert_close(actual, model(full)[0][:, 3:], rtol=3e-5, atol=4e-6)
        positions = torch.tensor([[1, 5], [2, 4]])
        assert_close(
            model(tokens, positions=positions)[0],
            dense.gather(1, positions[..., None].expand(-1, -1, 13)),
            rtol=0,
            atol=0,
        )


def test_source_archive_and_metadata_are_explicit(tmp_path: Path) -> None:
    """Keep the original loop and reject changed settings or archived sources."""
    assert train is original_train
    cfg = Configuration()
    plan = prepare_output(tmp_path, cfg, "cpu", compiled=False)
    assert "research/early99_dimension_gain/factories.py" in plan["source_sha256"]
    assert "research/early99/experiments/trainer.py" in plan["source_sha256"]
    prepare_output(tmp_path, cfg, "cpu", compiled=False)
    with pytest.raises(ValueError, match="settings/sources changed"):
        prepare_output(
            tmp_path, Configuration(learning_rate_multiplier=3), "cpu", compiled=False
        )
    path = tmp_path / "experiment_source/research/early99_dimension_gain/factories.py"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="Archived experiment source changed"):
        prepare_output(tmp_path, cfg, "cpu", compiled=False)
    case = Case(
        "dimension-metadata",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=16,
        heads=2,
        layers=2,
        batch=8,
    )
    metadata = cfg.metadata(case)
    assert metadata["scalable_softmax"]["s"] == 1
    assert metadata["dimension_gain"]["fixed_gain"] == 1 / 8
    assert metadata["dimension_gain"]["qk_rms_retained"]
    assert metadata["dimension_gain"]["added_parameters"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_width", [8, 32])
def test_cuda_capture_rng_moments_and_issued_updates(head_width: int) -> None:
    """Compare graphs with uncaptured native AdamW, including a short batch."""
    torch.compiler.reset()
    case = Case(
        "dimension-capture",
        "arithmetic",
        {"operation": "div", "prime": 7},
        width=2 * head_width,
        heads=2,
        layers=2,
        batch=8,
        learning_rate=0.003,
        weight_decay=0.1,
        warmup=3,
    )
    task = arithmetic(case, "cuda")
    cfg = Configuration()
    factories = configured_factories(cfg)
    torch.manual_seed(5)
    model = factories.model(task, case, "cuda")
    initial = copy.deepcopy(model.state_dict())
    stepper = FactoryStepper(
        model, task, case, "cuda", configuration=cfg, factories=factories, compiled=True
    )
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    stepper.prepare([8, 5])
    assert_close(model.state_dict(), initial, rtol=0, atol=0)
    assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    reference = factories.model(task, case, "cuda")
    reference.load_state_dict(initial)
    eager = Stepper(reference, task, cfg.training_case(case), "cuda", compiled=False)
    issued_sizes = [8, 5, 8, 5, 8]
    for completed, size in enumerate(issued_sizes):
        stepper.step(torch.arange(size), completed)
        eager.rate.fill_(
            cfg.learning_rate_at(case, completed, len(task.splits["train"]))
        )
        eager.update(torch.arange(size, device="cuda"))
    torch.cuda.synchronize()
    assert_close(model.state_dict(), reference.state_dict(), rtol=4e-4, atol=3e-6)
    for first, second in zip(
        stepper.optimizer.state.values(), eager.optimizer.state.values(), strict=True
    ):
        assert first["step"].item() == second["step"].item() == len(issued_sizes)
        assert_close(first["exp_avg"], second["exp_avg"], rtol=5e-4, atol=3e-6)
        assert_close(first["exp_avg_sq"], second["exp_avg_sq"], rtol=5e-4, atol=3e-7)
