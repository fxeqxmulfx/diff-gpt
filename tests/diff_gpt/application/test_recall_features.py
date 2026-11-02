"""Keep architectural recall screens isolated from the passing AdamW control."""

import torch
from torch.testing import assert_close

from diff_gpt.domain.model.feed_forward import FeedForward
from research.recall_ablation.features import build_model

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


def test_swiglu_candidate_keeps_finite_gradients_to_all_parameters() -> None:
    """Exercise the full residual stack and the optimized query readout."""
    model, _ = build_model("swiglu", 1)
    tokens = torch.randint(548, (2, 8))
    positions = torch.tensor([[4, 7], [5, 7]])
    targets = torch.randint(548, (2, 2))
    _, loss = model(tokens, targets, positions=positions)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name
