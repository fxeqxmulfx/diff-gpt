"""Keep architectural recall screens isolated from the passing AdamW control."""

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.model.kv_cache import KVCache
from diff_gpt.domain.model.mini_block import MiniRMSNorm
from diff_gpt.domain.model.rms_norm import RMSNorm as RepairedRMSNorm
from research.recall_ablation.features import build_model, raw_parameter_digest

HISTORICAL_INITIAL = "64c8ceab8de24ad993b068c1cf7f13c632f71b09c6dc623eecba00502a74959a"
BASELINE_PARAMETERS = 1_249_816
SWIGLU_PARAMETERS = 1_249_048
HISTORICAL_RMS_EPS = 1e-5


def test_swiglu_changes_only_the_mlp_of_the_recorded_control() -> None:
    """Preserve every shared matrix, head temperature and historical seed."""
    baseline, control = build_model("baseline", 1)
    candidate, treatment = build_model("swiglu", 1)
    assert control["historical_initial_parameters_sha256"] == HISTORICAL_INITIAL
    assert treatment["historical_initial_parameters_sha256"] == HISTORICAL_INITIAL
    assert (
        control["shared_non_mlp_initial_sha256"]
        == treatment["shared_non_mlp_initial_sha256"]
    )
    assert sum(p.numel() for p in baseline.parameters()) == BASELINE_PARAMETERS
    assert sum(p.numel() for p in candidate.parameters()) == SWIGLU_PARAMETERS
    shared = {
        name: value
        for name, value in baseline.state_dict().items()
        if ".ffwd." not in name
    }
    for name, expected in shared.items():
        assert_close(candidate.state_dict()[name], expected, rtol=0, atol=0)
    assert all(isinstance(block.ffwd, FeedForward) for block in candidate.blocks)
    assert treatment["features"] == ["SwiGLU"]
    assert treatment["initialization_from_trained_weights"] is False


@pytest.mark.parametrize(
    "stage",
    [
        "swiglu",
        "embedding-norm",
        "peri-norm",
        "peri-norm-repaired",
        "rms-repaired",
        "residual-scale",
        "qk-rms",
        "fixed-temperature",
    ],
)
def test_candidate_keeps_finite_gradients_to_all_parameters(stage: str) -> None:
    """Exercise the full residual stack and the optimized query readout."""
    model, _ = build_model(stage, 1)
    tokens = torch.randint(548, (2, 8))
    positions = torch.tensor([[4, 7], [5, 7]])
    targets = torch.randint(548, (2, 2))
    _, loss = model(tokens, targets, positions=positions)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


def test_public_gpt_is_the_passing_fixed_temperature_candidate() -> None:
    """Promotion preserves the exact architecture and tested initialization."""
    candidate, _ = build_model("fixed-temperature", 1)
    torch.manual_seed(1)
    model = GPT(
        548,
        128,
        block_size=64,
        n_head=4,
        n_layer=6,
        pad_vocab_size_to=1,
        use_checkpoint=False,
    )
    assert_close(model.state_dict(), candidate.state_dict(), rtol=0, atol=0)
    tokens = torch.randint(548, (2, 8))
    assert_close(model(tokens)[0], candidate(tokens)[0], rtol=0, atol=0)
    norms = [
        module for module in model.modules() if isinstance(module, RepairedRMSNorm)
    ]
    assert len(norms) == 4 * model.n_layer + 1


def test_scaled_initialization_changes_only_residual_projection_magnitudes() -> None:
    """Reuse every Gaussian draw and multiply only the twelve output matrices."""
    parent, _ = build_model("swiglu", 1)
    candidate, metadata = build_model("scaled-init", 1)
    assert metadata["parent_initial_parameters_sha256"] == raw_parameter_digest(parent)
    scale = (2 * parent.n_layer) ** -0.5
    assert metadata["initial_projection_scale"] == scale
    for name, parameter in parent.named_parameters():
        expected = parameter * scale if name.endswith("c_proj.weight") else parameter
        assert_close(dict(candidate.named_parameters())[name], expected, rtol=0, atol=0)


def test_embedding_normalization_reuses_every_parameter_and_keeps_head_tied() -> None:
    """Add exactly native RMS after lookup without reinitialization or scaling."""
    parent, _ = build_model("swiglu", 1)
    candidate, metadata = build_model("embedding-norm", 1)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert candidate.token_embedding_table.weight is candidate.lm_head.weight
    assert raw_parameter_digest(candidate) == raw_parameter_digest(parent)
    assert metadata["parent_stage"] == "swiglu"
    assert metadata["initial_projection_scale"] == 1.0
    tokens = torch.tensor([[1, 36, 292, 37, 293]])
    expected = F.rms_norm(parent.token_embedding_table(tokens), (parent.n_embd,))
    assert_close(candidate.token_embedding_table(tokens), expected, rtol=0, atol=0)


def test_peri_rms_changes_only_branch_outputs_and_preserves_every_parameter() -> None:
    """Keep raw embeddings and input norms; check each added post-projection RMS."""
    parent, _ = build_model("swiglu", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("peri-norm", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert raw_parameter_digest(candidate) == raw_parameter_digest(parent)
    assert candidate.token_embedding_table.weight is candidate.lm_head.weight
    assert metadata["parent_stage"] == "swiglu"
    assert metadata["embedding_normalization"] is None
    assert metadata["initial_projection_scale"] == 1.0
    assert metadata["forward_residual_scale"] == 1.0
    torch.manual_seed(3)
    for original, changed in zip(parent.blocks, candidate.blocks, strict=True):
        value = torch.randn(2, 4, parent.n_embd)
        assert_close(changed.ln_1(value), original.ln_1(value), rtol=0, atol=0)
        assert_close(changed.ln_2(value), original.ln_2(value), rtol=0, atol=0)
        hidden = torch.randn(2, 4, original.ffwd.c_proj.in_features)
        for old_projection, new_projection, inputs in (
            (original.sa.c_proj, changed.sa.c_proj, value),
            (original.ffwd.c_proj, changed.ffwd.c_proj, hidden),
        ):
            expected = F.rms_norm(old_projection(inputs), (parent.n_embd,))
            assert_close(new_projection(inputs), expected, rtol=0, atol=0)


def test_repaired_peri_preserves_forward_epsilons_and_all_initial_parameters() -> None:
    """Change RMS backward only, including pre-branch, post-branch and final sites."""
    native, _ = build_model("peri-norm", 1)
    native_rng = torch.get_rng_state()
    repaired, metadata = build_model("peri-norm-repaired", 1)
    assert torch.equal(torch.get_rng_state(), native_rng)
    assert_close(repaired.state_dict(), native.state_dict(), rtol=0, atol=0)
    assert metadata["repaired_rms_sites"] == 4 * repaired.n_layer + 1
    native_modules = dict(native.named_modules())
    corrected = {
        name: module
        for name, module in repaired.named_modules()
        if isinstance(module, RepairedRMSNorm)
    }
    assert len(corrected) == 4 * repaired.n_layer + 1
    for name, norm in corrected.items():
        old = native_modules[name]
        expected_eps = 1e-5 if isinstance(old, MiniRMSNorm) else old.eps
        assert norm.eps == expected_eps
        assert not list(norm.parameters())
        epsilon = torch.finfo(torch.float32).eps if norm.eps is None else norm.eps
        value = torch.ones(2, repaired.n_embd, requires_grad=True)
        gradient = torch.autograd.grad(norm(value).sum(), value)[0]
        expected = torch.full_like(value, epsilon / (1 + epsilon) ** 1.5)
        assert_close(gradient, expected, rtol=2e-6, atol=1e-12)
    tokens = torch.randint(548, (2, 8))
    assert_close(repaired(tokens)[0], native(tokens)[0], rtol=0, atol=0)


def test_repaired_swiglu_control_excludes_peri_and_preserves_forward() -> None:
    """Isolate the requested RMS backward before testing further architecture."""
    native, _ = build_model("swiglu", 1)
    native_rng = torch.get_rng_state()
    repaired, metadata = build_model("rms-repaired", 1)
    assert torch.equal(torch.get_rng_state(), native_rng)
    assert_close(repaired.state_dict(), native.state_dict(), rtol=0, atol=0)
    assert metadata["repaired_rms_sites"] == 2 * repaired.n_layer + 1
    assert metadata["branch_output_normalization"] is None
    assert metadata["embedding_normalization"] is None
    assert metadata["initial_projection_scale"] == 1.0
    for module in repaired.modules():
        if isinstance(module, RepairedRMSNorm):
            assert module.eps == HISTORICAL_RMS_EPS
    tokens = torch.randint(548, (2, 8))
    assert_close(repaired(tokens)[0], native(tokens)[0], rtol=0, atol=0)


def test_forward_residual_scaling_keeps_repaired_parent_and_initial_weights() -> None:
    """Scale only projected branch outputs, with the repaired RMS held constant."""
    parent, _ = build_model("rms-repaired", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("residual-scale", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    scale = (2 * candidate.n_layer) ** -0.5
    assert metadata["parent_stage"] == "rms-repaired"
    assert metadata["forward_residual_scale"] == scale
    assert metadata["initial_projection_scale"] == 1.0
    assert metadata["embedding_normalization"] is None
    assert metadata["branch_output_normalization"] is None
    assert metadata["repaired_rms_sites"] == 2 * candidate.n_layer + 1
    for old, new in zip(parent.blocks, candidate.blocks, strict=True):
        for original, changed in (
            (old.sa.c_proj, new.sa.c_proj),
            (old.ffwd.c_proj, new.ffwd.c_proj),
        ):
            inputs = torch.randn(2, 4, original.in_features)
            assert_close(changed(inputs), original(inputs) * scale, rtol=0, atol=0)


def test_qk_rms_preserves_parameters_rng_and_learned_cosine_temperature() -> None:
    """Isolate Q/K normalization without an accidental head-dimension gain."""
    parent, _ = build_model("rms-repaired", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("qk-rms", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert metadata["parent_stage"] == "rms-repaired"
    assert metadata["learned_attention_temperature"] is True
    assert metadata["repaired_rms_sites"] == 4 * candidate.n_layer + 1
    assert metadata["embedding_normalization"] is None
    assert metadata["branch_output_normalization"] is None
    assert metadata["initial_projection_scale"] == 1.0
    assert metadata["forward_residual_scale"] == 1.0
    for original, changed in zip(parent.blocks, candidate.blocks, strict=True):
        attention = changed.sa
        assert attention.log_alpha.requires_grad
        assert isinstance(attention.q_norm, RepairedRMSNorm)
        assert isinstance(attention.k_norm, RepairedRMSNorm)
        q = torch.randn(2, attention.n_head, 4, attention.head_dim)
        k = torch.randn_like(q)
        q_rms, k_rms = attention.normalize_qk(q, k)
        assert_close(q_rms, F.rms_norm(q, (attention.head_dim,)), rtol=0, atol=0)
        assert_close(k_rms, F.rms_norm(k, (attention.head_dim,)), rtol=0, atol=0)
        assert_close(
            attention.attention_gain(),
            original.sa.attention_gain() / attention.head_dim,
            rtol=0,
            atol=0,
        )
        # FP64 makes the numerical epsilon negligible, so the same learned
        # cosine temperature must produce the same score before/after RMS.
        q64, k64 = q.double(), k.double()
        original.sa.double()
        attention.double()
        old_q, old_k = original.sa.normalize_qk(q64, k64)
        new_q, new_k = attention.normalize_qk(q64, k64)
        old_scores = (old_q @ old_k.transpose(-1, -2)) * original.sa.attention_gain()
        new_scores = (new_q @ new_k.transpose(-1, -2)) * attention.attention_gain()
        assert_close(new_scores, old_scores, rtol=1e-12, atol=1e-12)


def test_qk_rms_cache_reproduces_full_context_predictions() -> None:
    """Keep the head-gain compensation valid for incremental decode too."""
    model, _ = build_model("qk-rms", 1)
    model.eval()
    tokens = torch.randint(548, (2, 8))
    cache = KVCache(batch_size=2, num_heads=4, seq_len=64, head_dim=32, num_layers=6)
    with torch.no_grad():
        expected = model(tokens)[0]
        actual = torch.cat(
            [model(tokens[:, i : i + 1], kv_cache=cache)[0] for i in range(8)], 1
        )
    assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_fixed_temperature_keeps_initial_forward_and_all_remaining_parameters() -> None:
    """Remove only trainable gains; keep the initial scores and every matrix."""
    parent, _ = build_model("qk-rms", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("fixed-temperature", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert metadata["parent_stage"] == "qk-rms"
    assert metadata["parent_initial_parameters_sha256"] == raw_parameter_digest(parent)
    assert metadata["learned_attention_temperature"] is False
    assert metadata["repaired_rms_sites"] == 4 * candidate.n_layer + 1
    old_parameters = dict(parent.named_parameters())
    new_parameters = dict(candidate.named_parameters())
    removed = {f"blocks.{i}.sa.log_alpha" for i in range(candidate.n_layer)}
    assert set(metadata["removed_parameter_names"]) == removed
    assert old_parameters.keys() - new_parameters.keys() == removed
    assert sum(p.numel() for p in candidate.parameters()) == (
        SWIGLU_PARAMETERS - candidate.n_layer * candidate.n_head
    )
    for name, parameter in new_parameters.items():
        assert_close(parameter, old_parameters[name], rtol=0, atol=0)
    buffers = dict(candidate.named_buffers())
    for name in removed:
        assert not buffers[name].requires_grad
        assert name not in candidate.state_dict()
        assert_close(buffers[name], old_parameters[name], rtol=0, atol=0)
    for original, changed in zip(parent.blocks, candidate.blocks, strict=True):
        assert_close(
            changed.sa.attention_gain(), original.sa.attention_gain(), rtol=0, atol=0
        )
        expected = torch.full_like(
            changed.sa.attention_gain(), changed.sa.head_dim**-0.5
        )
        assert_close(changed.sa.attention_gain(), expected)
    tokens = torch.randint(548, (2, 8))
    assert_close(candidate(tokens)[0], parent(tokens)[0], rtol=0, atol=0)
