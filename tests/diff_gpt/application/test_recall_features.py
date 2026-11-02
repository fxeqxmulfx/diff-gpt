"""Keep architectural recall screens isolated from the passing AdamW control."""

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from diff_gpt.domain.model.feed_forward import FeedForward
from diff_gpt.domain.model.gpt import GPT
from research.recall_ablation.features import build_model, raw_parameter_digest

HISTORICAL_INITIAL = "64c8ceab8de24ad993b068c1cf7f13c632f71b09c6dc623eecba00502a74959a"
BASELINE_PARAMETERS = 1_249_816
SWIGLU_PARAMETERS = 1_249_048


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


@pytest.mark.parametrize("stage", ["swiglu", "embedding-norm", "peri-norm"])
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


def test_public_gpt_is_the_passing_swiglu_candidate() -> None:
    """Promotion preserves the exact architecture and tested initialization."""
    candidate, _ = build_model("swiglu", 1)
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
