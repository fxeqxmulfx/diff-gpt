"""Shared learned log-frequency multipliers on the matched Peri-RMS parent."""

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diff_gpt.domain.model.gpt import PrefixStableGPT

if TYPE_CHECKING:
    from diff_gpt.domain.model.gpt import GPT
    from diff_gpt.domain.model.kv_cache import KVCache


class LearnedFrequencies(nn.Module):
    """
    LeRoPE theta_hat = theta * exp(alpha), shared by all layers and heads.

    Source: arXiv:2607.10134v1 Section 3.1. Start alpha at zero, preserve
    the benchmark base 10000 and half-split channel layout. This adds d_h/2
    parameters; no claim of identical parameter count to the parent is made.
    """

    def __init__(self, head_dim: int, positions: int, reference: torch.Tensor) -> None:
        """Create zero log-scales and fixed angle factors without a random draw."""
        super().__init__()
        self.log_scales = nn.Parameter(reference.new_zeros(head_dim // 2))
        # Match the original float32 table construction, then follow model dtype.
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        angles = torch.outer(
            torch.arange(positions, dtype=torch.float32), 10000**-exponents
        )
        self.register_buffer("base_angles", angles.to(reference), persistent=False)

    def forward(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute differentiable angles on every call, including captured updates."""
        angles = self.base_angles[start:end] * self.log_scales.exp()
        return angles.cos(), angles.sin()


class LearnedRoPEGPT(PrefixStableGPT):
    """Retain guarded GPT computation, replacing only the source of rotary tables."""

    def __init__(self, model: GPT) -> None:
        """Reuse parent parameters and append one shared frequency vector."""
        super().__init__(model)
        self.lerope = LearnedFrequencies(
            model.n_embd // model.n_head, model.block_size, model.rope_cos
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        *,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Use learned angles with the original causal cache and readout behavior."""
        start = 0 if kv_cache is None else kv_cache.get_pos()
        end = start + idx.shape[1]
        if end > self.block_size:
            message = f"positions up to {end} exceed block_size {self.block_size}"
            raise ValueError(message)
        cos, sin = self.lerope(start, end)
        value = self.token_embedding_table(idx)
        constant = (idx != idx[:, :1]).cumsum(dim=1) == 0 if kv_cache is None else None
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                result = checkpoint(
                    block,
                    value,
                    cos,
                    sin,
                    kv_cache,
                    constant=constant,
                    use_reentrant=False,
                )
            else:
                result = block(value, cos, sin, kv_cache, constant=constant)
            if constant is None:
                value = result
            else:
                value, constant = result
        if constant is not None:
            value = torch.where(constant[..., None], value[:, :1], value)
        value = self.ln_f(value)
        if positions is not None:
            value = value.gather(1, positions[..., None].expand(-1, -1, self.n_embd))
        logits = self.lm_head(value)
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        )
        return logits, loss
