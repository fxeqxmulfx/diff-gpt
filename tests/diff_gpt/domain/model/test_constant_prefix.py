"""Guarded repeated-prefix states preserve the source parameterized model."""

import copy
import unittest

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.gpt import GPT
from tests.diff_gpt.domain.model.source_equations import source_gpt


class ConstantPrefixTests(unittest.TestCase):
    """Check real-function correspondence and the gradient cancellation repair."""

    def setUp(self) -> None:
        """Fix initialization and CPU reductions."""
        torch.set_num_threads(1)
        torch.manual_seed(147)

    @staticmethod
    def model() -> GPT:
        """Use positive FP32 RMS epsilon in every precision comparison."""
        model = GPT(16, 16, 5, 2, 4, pad_vocab_size_to=1, use_checkpoint=False)
        for module in model.modules():
            if isinstance(module, torch.nn.RMSNorm):
                module.eps = torch.finfo(torch.float32).eps
        return model

    def test_repeated_tokens_match_independent_ffn_path_and_gradients(self) -> None:
        """Shared-token gradients need no ambient-position subtraction."""
        model = self.model()
        tokens = torch.tensor([[3, 3, 3, 3], [7, 7, 7, 7]])
        logits, _ = model(tokens)
        state = model.embedding_norm(model.token_embedding_table(tokens[:, :1]))
        for block in model.blocks:
            state = state + model.branch_scale * block.ffn_out_norm(
                block.ffwd(block.ln_2(state))
            )
        expected = model.lm_head(model.ln_f(state)).expand(-1, tokens.shape[1], -1)
        torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-7)
        direction = torch.randn_like(logits)
        parameters = tuple(model.parameters())
        actual_gradients = torch.autograd.grad(
            (logits * direction).sum(), parameters, allow_unused=True
        )
        expected_gradients = torch.autograd.grad(
            (expected * direction).sum(), parameters, allow_unused=True
        )
        for parameter, actual, reference in zip(
            parameters, actual_gradients, expected_gradients, strict=True
        ):
            actual_tensor = torch.zeros_like(parameter) if actual is None else actual
            reference_tensor = (
                torch.zeros_like(parameter) if reference is None else reference
            )
            torch.testing.assert_close(
                actual_tensor, reference_tensor, rtol=3e-5, atol=2e-6
            )

    def test_nonconstant_suffix_values_and_gradients(
        self,
    ) -> None:
        """A repeated causal prefix may be followed by arbitrary other tokens."""
        stable = self.model().double()
        source = copy.deepcopy(stable)
        tokens = torch.tensor([[2, 2, 5, 6], [3, 3, 3, 1], [1, 1, 1, 1], [2, 4, 2, 4]])
        got, _ = stable(tokens)
        want = source_gpt(source, tokens)
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-9)
        labels = torch.tensor([2, 5, 7, 11])
        actual = torch.autograd.grad(
            F.cross_entropy(got[:, -1], labels), stable.parameters()
        )
        expected = torch.autograd.grad(
            F.cross_entropy(want[:, -1], labels), source.parameters()
        )
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)

    def test_clipped_value_head_disables_the_prefix_rewrite(self) -> None:
        """A below-epsilon head can make repeated-token positions differ."""
        stable = GPT(16, 8, 3, 2, 1, pad_vocab_size_to=1, use_checkpoint=False).double()
        for module in stable.modules():
            if isinstance(module, torch.nn.RMSNorm):
                module.eps = torch.finfo(torch.float32).eps
        block = stable.blocks[0]
        with torch.no_grad():
            block.sa.c_attn.weight[16:].copy_(torch.eye(8, dtype=torch.float64) * 1e-13)
            block.sa.c_proj.weight.copy_(torch.eye(8, dtype=torch.float64))
            block.ffwd.c_proj.weight.zero_()
        source = copy.deepcopy(stable)
        tokens = torch.tensor([[3, 3, 3]])
        got, _ = stable(tokens)
        want = source_gpt(source, tokens)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
        distinct_threshold = 1e-14
        assert (
            float((got[:, 1:] - got[:, :1]).detach().abs().max()) > distinct_threshold
        )


if __name__ == "__main__":
    unittest.main()
