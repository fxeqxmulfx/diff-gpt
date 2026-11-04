"""Keep architectural recall screens isolated from the passing AdamW control."""

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.gpt import GPT, PrefixStableGPT
from diff_gpt.domain.model.kv_cache import KVCache
from diff_gpt.domain.model.mini_block import MiniRMSNorm
from diff_gpt.domain.model.rms_norm import RMSNorm as RepairedRMSNorm
from research.recall_ablation.features import build_model, raw_parameter_digest

HISTORICAL_INITIAL = "64c8ceab8de24ad993b068c1cf7f13c632f71b09c6dc623eecba00502a74959a"
BASELINE_PARAMETERS = 1_249_816
SWIGLU_PARAMETERS = 1_249_048
HISTORICAL_RMS_EPS = 1e-5
MODERN_XSA_EPS = 1e-12


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
        "sdpa",
        "rms-epsilon",
        "xsa-epsilon",
        "centered-xsa",
        "first-zero",
        "prefix-stable",
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


def test_public_gpt_is_the_passing_prefix_stable_candidate() -> None:
    """Promotion preserves the exact architecture and tested initialization."""
    candidate, _ = build_model("prefix-stable", 1)
    candidate_rng = torch.get_rng_state().clone()
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
    assert isinstance(model, PrefixStableGPT)
    assert torch.equal(torch.get_rng_state(), candidate_rng)
    assert_close(model.state_dict(), candidate.state_dict(), rtol=0, atol=0)
    tokens = torch.randint(548, (2, 8))
    tokens[:, :4] = tokens[:, :1].clone()
    assert_close(model(tokens)[0], candidate(tokens)[0], rtol=0, atol=0)
    norms = [
        module for module in model.modules() if isinstance(module, RepairedRMSNorm)
    ]
    assert len(norms) == 4 * model.n_layer + 1
    assert all(norm.eps is None for norm in norms)
    assert all(block.sa.xsa_epsilon == MODERN_XSA_EPS for block in model.blocks)


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


def test_sdpa_preserves_fixed_parent_parameters_rng_and_forward() -> None:
    """Change only the dense attention evaluation, preserving its real gain."""
    parent, _ = build_model("fixed-temperature", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("sdpa", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert metadata["parent_stage"] == "fixed-temperature"
    assert metadata["parent_initial_parameters_sha256"] == raw_parameter_digest(parent)
    assert raw_parameter_digest(candidate) == raw_parameter_digest(parent)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert metadata["learned_attention_temperature"] is False
    assert metadata["repaired_rms_sites"] == 4 * candidate.n_layer + 1
    assert metadata["embedding_normalization"] is None
    assert metadata["branch_output_normalization"] is None
    assert metadata["forward_residual_scale"] == 1.0
    assert metadata["initial_projection_scale"] == 1.0
    tokens = torch.randint(548, (2, 8))
    assert_close(candidate(tokens)[0], parent(tokens)[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("queries", "keys"), [(3, 3), (3, 7)])
def test_sdpa_scores_and_gradients_match_independent_dense_oracle(
    queries: int, keys: int
) -> None:
    """Check score scaling and the causal offset for a cached query chunk."""
    model, _ = build_model("sdpa", 1)
    attention = model.blocks[0].sa.double()
    tensors = [
        torch.randn(2, 4, length, 32, dtype=torch.float64, requires_grad=True)
        for length in (queries, keys, keys)
    ]
    q, k, v = tensors
    alpha = attention.attention_gain()
    actual = attention.attend_dense(q, k, v, alpha)
    causal = torch.ones(queries, keys, dtype=torch.bool).tril(diagonal=keys - queries)
    scores = (q @ k.transpose(-1, -2)) * alpha
    expected = scores.masked_fill(~causal, -torch.inf).softmax(-1) @ v
    assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, tensors, upstream)
    expected_grads = torch.autograd.grad(expected, tensors, upstream)
    assert_close(actual_grads, expected_grads, rtol=1e-12, atol=1e-12)


def test_sdpa_cache_reproduces_full_context_predictions() -> None:
    """Verify token-by-token decode and cached chunks after a fused prefill."""
    model, _ = build_model("sdpa", 1)
    model.eval()
    tokens = torch.randint(548, (2, 8))
    with torch.no_grad():
        expected = model(tokens)[0]
        for chunks in ([1] * 8, [3, 2, 3]):
            cache = KVCache(
                batch_size=2, num_heads=4, seq_len=64, head_dim=32, num_layers=6
            )
            position = 0
            results = []
            for length in chunks:
                results.append(
                    model(tokens[:, position : position + length], kv_cache=cache)[0]
                )
                position += length
            assert_close(torch.cat(results, 1), expected, rtol=1e-5, atol=1e-5)


def test_dtype_epsilon_changes_only_input_and_final_norms() -> None:
    """Keep explicit attention, all initial weights, fixed gains and the repair."""
    parent, _ = build_model("fixed-temperature", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("rms-epsilon", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert raw_parameter_digest(candidate) == raw_parameter_digest(parent)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert_close(
        dict(candidate.named_buffers()), dict(parent.named_buffers()), rtol=0, atol=0
    )
    assert metadata["parent_stage"] == "fixed-temperature"
    assert metadata["parent_initial_parameters_sha256"] == raw_parameter_digest(parent)
    assert metadata["learned_attention_temperature"] is False
    assert metadata["dense_attention"] == "explicit causal softmax"
    assert metadata["repaired_rms_sites"] == 4 * candidate.n_layer + 1
    assert metadata["embedding_normalization"] is None
    assert metadata["branch_output_normalization"] is None
    assert metadata["initial_projection_scale"] == 1.0
    assert metadata["forward_residual_scale"] == 1.0
    expected_sites = {"ln_f"} | {
        f"blocks.{i}.ln_{branch}" for i in range(candidate.n_layer) for branch in (1, 2)
    }
    assert set(metadata["epsilon_changed_sites"]) == expected_sites
    old_modules = dict(parent.named_modules())
    for name, module in candidate.named_modules():
        assert type(module) is type(old_modules[name])
        if isinstance(module, RepairedRMSNorm):
            assert module.eps is None
            if name in expected_sites:
                assert old_modules[name].eps == HISTORICAL_RMS_EPS
            else:
                assert old_modules[name].eps is None


def test_dtype_epsilon_forward_and_radial_gradient_match_independent_equations() -> (
    None
):
    """Exercise all thirteen changed sites at embedding and unit amplitudes."""
    candidate, metadata = build_model("rms-epsilon", 1)
    modules = dict(candidate.named_modules())
    epsilon = torch.finfo(torch.float32).eps
    for name in metadata["epsilon_changed_sites"]:
        norm = modules[name]
        for amplitude in (0.02, 1.0):
            value = torch.full((2, candidate.n_embd), amplitude, requires_grad=True)
            actual = norm(value)
            assert_close(actual, F.rms_norm(value, (candidate.n_embd,)), rtol=0, atol=0)
            gradient = torch.autograd.grad(actual.sum(), value)[0]
            expected = epsilon / (value.detach().double().square() + epsilon) ** 1.5
            assert_close(gradient, expected.float(), rtol=2e-6, atol=1e-12)


def test_xsa_epsilon_keeps_all_passing_features_and_initial_weights() -> None:
    """Change only the own-value clipping threshold, keeping ordinary projection."""
    parent, _ = build_model("rms-epsilon", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("xsa-epsilon", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert_close(
        dict(candidate.named_buffers()), dict(parent.named_buffers()), rtol=0, atol=0
    )
    assert raw_parameter_digest(candidate) == raw_parameter_digest(parent)
    assert metadata["parent_initial_parameters_sha256"] == raw_parameter_digest(parent)
    assert metadata["parent_stage"] == "rms-epsilon"
    assert metadata["xsa_epsilon"] == MODERN_XSA_EPS
    assert metadata["dense_attention"] == "explicit causal softmax"
    assert metadata["learned_attention_temperature"] is False
    assert metadata["embedding_normalization"] is None
    assert metadata["branch_output_normalization"] is None
    assert metadata["initial_projection_scale"] == 1.0
    assert metadata["forward_residual_scale"] == 1.0
    for block in candidate.blocks:
        assert block.sa.xsa_epsilon == MODERN_XSA_EPS
    for norm in candidate.modules():
        if isinstance(norm, RepairedRMSNorm):
            assert norm.eps is None
    tokens = torch.randint(548, (2, 8))
    assert_close(candidate(tokens)[0], parent(tokens)[0], rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [0.0, 1e-13, 1e-9, 2e-6])
def test_xsa_clipping_preserves_perpendicular_and_clips_parallel(
    magnitude: float,
) -> None:
    """A clipped parallel component scales by 1-(norm/epsilon)^2; others survive."""
    parent, _ = build_model("rms-epsilon", 1)
    candidate, _ = build_model("xsa-epsilon", 1)
    for attention in (parent.blocks[0].sa, candidate.blocks[0].sa):
        own = torch.zeros(1, 1, 1, 32, dtype=torch.float64)
        own[..., 0] = magnitude
        own.requires_grad_()
        attended = torch.zeros_like(own)
        attended[..., 0], attended[..., 1] = 3.0, 5.0
        attended.requires_grad_()
        expected = torch.zeros_like(own)
        expected[..., 0] = 3 * (1 - min(1, magnitude / attention.xsa_epsilon) ** 2)
        expected[..., 1] = 5.0
        actual = attention.project_exclusive(attended, own)
        assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        gradients = torch.autograd.grad(actual.sum(), (attended, own))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_centered_xsa_preserves_weights_rng_and_all_passing_settings() -> None:
    """Keep the threshold, epsilons, fixed gain and explicit attention unchanged."""
    parent, _ = build_model("xsa-epsilon", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("centered-xsa", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert_close(
        dict(candidate.named_buffers()), dict(parent.named_buffers()), rtol=0, atol=0
    )
    assert raw_parameter_digest(candidate) == raw_parameter_digest(parent)
    assert metadata["parent_initial_parameters_sha256"] == raw_parameter_digest(parent)
    assert metadata["parent_stage"] == "xsa-epsilon"
    assert metadata["xsa_epsilon"] == MODERN_XSA_EPS
    assert metadata["xsa_projection"] == "centered"
    assert metadata["dense_attention"] == "explicit causal softmax"
    assert metadata["learned_attention_temperature"] is False
    assert metadata["embedding_normalization"] is None
    assert metadata["branch_output_normalization"] is None
    assert metadata["initial_projection_scale"] == 1.0
    assert metadata["forward_residual_scale"] == 1.0
    assert all(
        module.eps is None
        for module in candidate.modules()
        if isinstance(module, RepairedRMSNorm)
    )


@pytest.mark.parametrize("value_scale", [1.0, 1e-14])
def test_centered_xsa_matches_fp64_parent_forward_and_gradients(
    value_scale: float,
) -> None:
    """Preserve real attention, including the below-clipping own-value remainder."""
    parent, _ = build_model("xsa-epsilon", 1)
    candidate, _ = build_model("centered-xsa", 1)
    parent.double()
    candidate.double()
    with torch.no_grad():
        for model in (parent, candidate):
            for block in model.blocks:
                block.sa.c_attn.weight[2 * model.n_embd :].mul_(value_scale)
    tokens = torch.randint(548, (2, 8))
    expected, actual = parent(tokens)[0], candidate(tokens)[0]
    assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    for (name, original), (_, changed) in zip(
        parent.named_parameters(), candidate.named_parameters(), strict=True
    ):
        assert_close(changed.grad, original.grad, rtol=1e-9, atol=1e-9, msg=name)
    if value_scale < 1:
        attention = candidate.blocks[0].sa
        value = torch.randn(2, 4, candidate.n_embd, dtype=torch.float64)
        branch = attention(value, candidate.rope_cos[:4], candidate.rope_sin[:4])
        assert torch.count_nonzero(branch[:, 0]), (
            "below-clipping first output must survive"
        )


def test_centered_xsa_cache_and_guard_match_full_predictions() -> None:
    """Verify raw cache storage, offset chunks and the future prefix guard output."""
    model, _ = build_model("centered-xsa", 1)
    model.eval()
    tokens = torch.randint(548, (2, 8))
    with torch.no_grad():
        expected = model(tokens)[0]
        for chunks in ([1] * 8, [3, 2, 3]):
            cache = KVCache(
                batch_size=2, num_heads=4, seq_len=64, head_dim=32, num_layers=6
            )
            position, parts = 0, []
            for length in chunks:
                parts.append(
                    model(tokens[:, position : position + length], kv_cache=cache)[0]
                )
                position += length
            assert_close(torch.cat(parts, 1), expected, rtol=1e-5, atol=1e-5)
        attention = model.blocks[0].sa
        value = torch.randn(2, 8, model.n_embd)
        output, guard = attention(
            value, model.rope_cos[:8], model.rope_sin[:8], return_guard=True
        )
        assert_close(
            output,
            attention(value, model.rope_cos[:8], model.rope_sin[:8]),
            rtol=0,
            atol=0,
        )
        assert guard.shape == (2,)
        assert guard.all()


@pytest.mark.parametrize(
    ("stage", "checkpointing"),
    [
        ("centered-xsa", False),
        ("first-zero", False),
        ("prefix-stable", False),
        ("prefix-stable", True),
    ],
)
def test_xsa_candidates_compile_with_matching_parameter_gradients(
    stage: str, *, checkpointing: bool
) -> None:
    """Check the fullgraph training path before launching a numerical ablation."""
    eager, _ = build_model(stage, 1)
    compiled_model, _ = build_model(stage, 1)
    eager.use_checkpoint = compiled_model.use_checkpoint = checkpointing
    tokens = torch.randint(548, (2, 8))
    if stage == "prefix-stable":
        tokens[:, :4] = tokens[:, :1].clone()
    positions = torch.tensor([[4, 7], [5, 7]])
    targets = torch.randint(548, (2, 2))

    def objective(
        inputs: torch.Tensor, query: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return compiled_model(inputs, labels, positions=query)[1]

    expected = eager(tokens, targets, positions=positions)[1]
    actual = torch.compile(objective, fullgraph=True, dynamic=False)(
        tokens, positions, targets
    )
    assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.backward()
    actual.backward()
    for (name, original), (_, changed) in zip(
        eager.named_parameters(), compiled_model.named_parameters(), strict=True
    ):
        assert_close(changed.grad, original.grad, rtol=2e-4, atol=2e-6, msg=name)


def test_first_zero_keeps_all_passing_parameters_and_settings() -> None:
    """The extra zeroing adds no parameters, random draws or other norm changes."""
    parent, _ = build_model("centered-xsa", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("first-zero", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert_close(
        dict(candidate.named_buffers()), dict(parent.named_buffers()), rtol=0, atol=0
    )
    assert raw_parameter_digest(candidate) == raw_parameter_digest(parent)
    assert metadata["parent_initial_parameters_sha256"] == raw_parameter_digest(parent)
    assert metadata["parent_stage"] == "centered-xsa"
    assert metadata["first_position_forced_zero"] is True
    assert metadata["xsa_epsilon"] == MODERN_XSA_EPS
    assert metadata["xsa_projection"] == "centered"
    assert metadata["dense_attention"] == "explicit causal softmax"
    assert metadata["learned_attention_temperature"] is False
    assert metadata["embedding_normalization"] is None
    assert metadata["branch_output_normalization"] is None
    assert metadata["initial_projection_scale"] == 1.0
    assert metadata["forward_residual_scale"] == 1.0
    tokens = torch.randint(548, (2, 8))
    assert_close(candidate(tokens)[0], parent(tokens)[0], rtol=0, atol=0)


def test_first_zero_removes_clipped_remainder_and_its_direct_gradient() -> None:
    """Use tiny values to distinguish explicit zeroing from the centered parent."""
    parent, _ = build_model("centered-xsa", 1)
    candidate, _ = build_model("first-zero", 1)
    parent.double()
    candidate.double()
    with torch.no_grad():
        for model in (parent, candidate):
            model.blocks[0].sa.c_attn.weight[2 * model.n_embd :].mul_(1e-14)
    value = torch.randn(2, 4, candidate.n_embd, dtype=torch.float64, requires_grad=True)
    old = parent.blocks[0].sa(value, parent.rope_cos[:4], parent.rope_sin[:4])
    actual, guard = candidate.blocks[0].sa(
        value, candidate.rope_cos[:4], candidate.rope_sin[:4], return_guard=True
    )
    assert torch.count_nonzero(old[:, 0])
    assert not torch.count_nonzero(actual[:, 0])
    assert not guard.any()
    assert_close(actual[:, 1:], old[:, 1:], rtol=0, atol=0)
    gradient = torch.autograd.grad(actual[:, 0].sum(), value)[0]
    assert not torch.count_nonzero(gradient)


def test_first_zero_cache_preserves_later_positions_and_chunk_continuation() -> None:
    """A cached call after position zero must keep its attention output."""
    model, _ = build_model("first-zero", 1)
    model.eval()
    tokens = torch.randint(548, (2, 8))
    with torch.no_grad():
        expected = model(tokens)[0]
        for chunks in ([1] * 8, [3, 2, 3]):
            cache = KVCache(
                batch_size=2, num_heads=4, seq_len=64, head_dim=32, num_layers=6
            )
            position, parts = 0, []
            for length in chunks:
                parts.append(
                    model(tokens[:, position : position + length], kv_cache=cache)[0]
                )
                position += length
            assert_close(torch.cat(parts, 1), expected, rtol=1e-5, atol=1e-5)


def _repeated_prefix_tokens() -> torch.Tensor:
    """Use different constant-prefix lengths, followed by ordinary token changes."""
    return torch.tensor(
        [[17, 17, 17, 17, 21, 23, 25, 27], [19, 19, 19, 31, 33, 35, 37, 39]]
    )


def test_prefix_adapter_preserves_weights_rng_buffers_and_mode() -> None:
    """Share exact parameter objects and retain the accepted model settings."""
    parent, _ = build_model("first-zero", 1)
    parent_rng = torch.get_rng_state()
    candidate, metadata = build_model("prefix-stable", 1)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert_close(candidate.state_dict(), parent.state_dict(), rtol=0, atol=0)
    assert_close(
        dict(candidate.named_buffers()), dict(parent.named_buffers()), rtol=0, atol=0
    )
    assert raw_parameter_digest(candidate) == raw_parameter_digest(parent)
    assert candidate.token_embedding_table.weight is candidate.lm_head.weight
    assert metadata["parent_initial_parameters_sha256"] == raw_parameter_digest(parent)
    assert metadata["parent_stage"] == "first-zero"
    assert metadata["repeated_prefix_rewrite"] is True
    assert metadata["first_position_forced_zero"] is True
    assert metadata["xsa_projection"] == "centered"
    assert metadata["xsa_epsilon"] == MODERN_XSA_EPS
    assert metadata["learned_attention_temperature"] is False
    assert metadata["dense_attention"] == "explicit causal softmax"
    assert metadata["embedding_normalization"] is None
    assert metadata["branch_output_normalization"] is None
    assert metadata["initial_projection_scale"] == 1.0
    assert metadata["forward_residual_scale"] == 1.0
    parent.eval()
    before_wrap = torch.get_rng_state()
    wrapped = PrefixStableGPT(parent)
    assert torch.equal(torch.get_rng_state(), before_wrap)
    assert not wrapped.training
    assert wrapped.use_checkpoint == parent.use_checkpoint
    assert all(
        p is q for p, q in zip(wrapped.parameters(), parent.parameters(), strict=True)
    )


@pytest.mark.parametrize("value_scale", [1.0, 1e-14])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_prefix_forward_gradients_and_clipping_guard_match_parent(
    value_scale: float, *, checkpointing: bool
) -> None:
    """Preserve real equations and permanently disable rewriting after clipping."""
    parent, _ = build_model("first-zero", 1)
    candidate, _ = build_model("prefix-stable", 1)
    parent.double()
    candidate.double()
    candidate.use_checkpoint = checkpointing
    with torch.no_grad():
        for model in (parent, candidate):
            model.blocks[0].sa.c_attn.weight[2 * model.n_embd :].mul_(value_scale)
    tokens = _repeated_prefix_tokens()
    positions = torch.tensor([[5, 7], [5, 7]])
    targets = torch.tensor([[300, 301], [302, 303]])
    expected, original_loss = parent(tokens, targets, positions=positions)
    actual, loss = candidate(tokens, targets, positions=positions)
    assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    original_loss.backward()
    loss.backward()
    for (name, original), (_, changed) in zip(
        parent.named_parameters(), candidate.named_parameters(), strict=True
    ):
        assert_close(changed.grad, original.grad, rtol=1e-9, atol=1e-9, msg=name)
    with torch.no_grad():
        initial = (tokens != tokens[:, :1]).cumsum(1) == 0
        constant = initial
        value = candidate.token_embedding_table(tokens)
        for block in candidate.blocks:
            value, constant = block(
                value, candidate.rope_cos[:8], candidate.rope_sin[:8], constant=constant
            )
            if value_scale < 1:
                assert not constant.any()
            else:
                assert torch.equal(constant, initial)
                assert torch.equal(
                    value[constant], value[:, :1].expand_as(value)[constant]
                )


def test_prefix_cache_matches_full_context_with_repeated_tokens() -> None:
    """Keep token and chunk decoding valid when full-context states are shared."""
    model, _ = build_model("prefix-stable", 1)
    model.eval()
    tokens = _repeated_prefix_tokens()
    with torch.no_grad():
        expected = model(tokens)[0]
        for chunks in ([1] * 8, [3, 2, 3]):
            cache = KVCache(
                batch_size=2, num_heads=4, seq_len=64, head_dim=32, num_layers=6
            )
            offset, parts = 0, []
            for length in chunks:
                parts.append(
                    model(tokens[:, offset : offset + length], kv_cache=cache)[0]
                )
                offset += length
            assert_close(torch.cat(parts, 1), expected, rtol=1e-5, atol=1e-5)
