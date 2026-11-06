"""Independent gradient identities and noninvasive instrumentation checks."""

import pytest
import torch
from torch.nn import functional as F

from diff_gpt.domain.model.gpt import GPT
from research.current_grokking.train import Logits
from research.early99.experiments.models import (
    install_embedding_rms,
    install_peri_rms,
    install_residual_scaling,
)
from research.early99.experiments.scalable_softmax import install_scalable_softmax
from research.early99_dimension_gain.factories import install_inverse_width
from research.transformer_path_diagnostic.diagnose import observe
from research.transformer_path_diagnostic.metrics import decomposition, probe_like

EXACT_TOLERANCE = 1e-14
TINY_GRADIENT = 1e-20
PATH_TOLERANCE = 1e-5
TWO_LAYER_BRANCHES = 4


class ToyTask:
    """Exercise multiple causal positions, repeated prefixes and every output."""

    @staticmethod
    def forward(
        model: Logits, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use a fixed artificial next-symbol target on a small vocabulary."""
        return model(batch), (batch + 1) % 13

    @staticmethod
    def loss(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Use mean cross-entropy, independently of the scientific task."""
        return F.cross_entropy(output.reshape(-1, 13), targets.reshape(-1))


def test_known_additive_cancellation() -> None:
    """Recover an analytic near-cancellation and reject a missing branch."""
    identity = torch.tensor([2.0, -1.0], dtype=torch.float64)
    branch = -0.75 * identity
    result = decomposition(identity, branch, 0.25 * identity)
    assert result["cosine"] == pytest.approx(-1)
    assert result["sum_to_separate_norms"] == pytest.approx(1 / 7)
    with pytest.raises(ValueError, match="does not reconstruct"):
        decomposition(identity, torch.zeros_like(identity), 0.25 * identity)


def test_tied_weight_two_uses() -> None:
    """Check two shared matrix uses against an independent product derivative."""
    weight = torch.tensor(
        [[0.2, -0.3], [0.5, 0.7]], dtype=torch.float64, requires_grad=True
    )
    hidden = weight[0:1]
    output = hidden @ weight.T
    cotangent = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    total = torch.autograd.grad(output, weight, grad_outputs=cotangent)[0]
    readout = cotangent.T @ hidden.detach()
    lookup = torch.zeros_like(weight)
    lookup[0] = (cotangent @ weight.detach())[0]
    result = decomposition(lookup, readout, total)
    assert result["reconstruction_relative_l2"] < EXACT_TOLERANCE


def test_probe_distinguishes_confidence_from_zero_jacobian() -> None:
    """A confident loss can have tiny gradients while the logit map remains live."""
    parameter = torch.tensor([[30.0, -30.0, -30.0]], requires_grad=True)
    loss = F.cross_entropy(parameter, torch.tensor([0]))
    loss_gradient = torch.autograd.grad(loss, parameter)[0]
    before = torch.get_rng_state().clone()
    probe = probe_like(parameter, 17)
    gradient = torch.autograd.grad((parameter * probe).sum(), parameter)[0]
    assert torch.equal(before, torch.get_rng_state())
    assert probe.sum().item() == pytest.approx(0, abs=1e-7)
    assert gradient.norm().item() == pytest.approx(1, abs=1e-7)
    assert loss_gradient.norm().item() < TINY_GRADIENT


@pytest.mark.parametrize("normalized", [False, True])
def test_actual_graph_forward_and_gradient_parity(*, normalized: bool) -> None:
    """Check every parameter, residual and tied path through two real GPT layers."""
    torch.manual_seed(12)
    gpt = GPT(
        13,
        16,
        block_size=4,
        n_head=2,
        n_layer=2,
        use_checkpoint=False,
        pad_vocab_size_to=1,
    )
    if normalized:
        install_peri_rms(gpt)
        install_embedding_rms(gpt)
        install_residual_scaling(gpt)
        install_scalable_softmax(gpt)
        install_inverse_width(gpt)
    batch = torch.tensor([[1, 1, 2, 3], [4, 5, 4, 6], [7, 7, 7, 8]])
    original_softmax = torch.Tensor.softmax
    result = observe(Logits(gpt), ToyTask(), batch, 173)
    assert result["exact_forward_and_all_parameter_gradient_parity"]
    assert result["unchanged_model_and_rng"]
    assert torch.Tensor.softmax is original_softmax
    for objective in result["objectives"].values():
        assert len(objective["residual_paths"]) == TWO_LAYER_BRANCHES
        assert (
            objective["tied_weight_paths"]["reconstruction_relative_l2"]
            <= PATH_TOLERANCE
        )
        assert all(
            path["gradient_paths"]["reconstruction_relative_l2"] <= PATH_TOLERANCE
            for path in objective["residual_paths"]
        )
        assert "blocks.1.sa.softmax.output" in objective["sites"]
        assert "blocks.1.ffwd.silu.output" in objective["sites"]
