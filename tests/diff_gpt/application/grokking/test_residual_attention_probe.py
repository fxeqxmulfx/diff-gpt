"""The synthetic residual probe has a solution and exposes conditioning."""

import math
import unittest

import torch

from diff_gpt.application.grokking.residual_attention_probe import (
    WIDTH,
    Probe,
    ProbeCase,
    dataset,
)


class ResidualProbeTests(unittest.TestCase):
    """Check the task independently before running any learning comparison."""

    def setUp(self) -> None:
        """Keep CPU reductions deterministic."""
        torch.set_num_threads(1)

    def test_teacher_has_known_student_weights(self) -> None:
        """An explicit query reproduces independently generated targets."""
        values, targets, chosen = dataset(123)
        model = Probe(ProbeCase("qk", 0.0, 11))
        with torch.no_grad():
            model.query.zero_()
            model.query[0] = math.sqrt(WIDTH)
            error = (model(values) - targets).square().mean()
            scores = model.scores(values)
        tolerance = 1e-10
        assert float(error) < tolerance
        assert torch.equal(scores.argmax(-1), chosen)
        assert len(chosen.unique()) > 1

    def test_saturation_and_large_query_conditioning(self) -> None:
        """Q/K scores stay bounded, but huge raw queries remain ill-conditioned."""
        values, targets, _ = dataset(123)
        gradients = {}
        for mode, scale in (("key_only", 1e6), ("qk", 1e6), ("qk", 1.0)):
            model = Probe(ProbeCase(mode, scale, 11))
            loss = (model(values) - targets).square().mean()
            (gradient,) = torch.autograd.grad(loss, model.query)
            assert torch.isfinite(gradient).all()
            gradients[mode, scale] = float(gradient.norm())
            if mode == "qk":
                assert float(model.scores(values).detach().abs().max()) <= math.sqrt(
                    WIDTH
                )
        assert gradients["key_only", 1e6] == 0.0
        assert gradients["qk", 1e6] > 0.0
        conditioning_ratio = 1000
        assert gradients["qk", 1.0] > conditioning_ratio * gradients["qk", 1e6]

    def test_permuting_sources_preserves_the_prediction(self) -> None:
        """The student cannot use the selected source's slot as a shortcut."""
        values, _, _ = dataset(123)
        model = Probe(ProbeCase("qk", 1.0, 11))
        permutation = torch.tensor([3, 6, 2, 0, 7, 5, 1, 4])
        torch.testing.assert_close(
            model(values), model(values[:, permutation]), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
