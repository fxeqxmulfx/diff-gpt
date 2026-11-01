"""
Full depth attention inside the normalized source RoPE transformer.

Source: Attention Residuals, arXiv:2603.15031v1, Section3.1 Eqs2--4 and
Section5 (zero pseudo-query initialization); GPT at0fab228. The embedding
and completed attention/MLP branch outputs are the sources, including a
final depth aggregation. There is no artificial zero source or extra
running residual sum. The first, singleton aggregation needs no parameter.

Host-specific deviations from the paper: retain the GPT Peri-RMS output
normalizations, Q/K-normalized RoPE/XSA, SwiGLU and tied readout. The qk and
conditioned_qk modes also normalize depth queries, remove the key gain and
divide scores by sqrt(width); their queries start at unit RMS. Key-only
attnres retains its learned key gain and zero queries. scaled uses ordinary
residual addition with fixed branch coefficient1/sqrt(2*depth).
The optional gated mode applies 2*sigmoid(s), s initially zero, AFTER each
Peri output norm. Source: transformer task151 known-solution benchmark and
Transformer.Residual.gain_derivative_ratio. This preserves an amplitude
gradient; full-transformer grokking improvement is not established.
"""

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diff_gpt.domain.model.attention_residual import QKAttentionResidual
from diff_gpt.domain.model.conditioned_attention_residual import (
    ConditionedQKAttentionResidual,
)
from diff_gpt.domain.model.gpt import GPT

if TYPE_CHECKING:
    from diff_gpt.domain.model.block import Block
    from diff_gpt.domain.model.kv_cache import KVCache

MODES = ("add", "scaled", "gated", "attnres", "qk", "conditioned_qk")


class KeyAttentionResidual(QKAttentionResidual):
    """Paper Section3.1 scoring, Section5 initialization, epsilon1e-6."""

    def __init__(self, width: int, *, eps: float = 1e-6) -> None:
        """Start at uniform averaging with a learned unit key gain."""
        super().__init__(width, eps=eps)
        with torch.no_grad():
            self.query.zero_()
        self.key_gain = nn.Parameter(torch.ones(width))

    def scores(self, values: torch.Tensor) -> torch.Tensor:
        """Leave the query unnormalized and use no width divisor."""
        keys = F.rms_norm(values, (self.width,), self.key_gain, eps=self.eps)
        return (keys * self.query).sum(-1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply depth softmax to a nonempty family of raw source values."""
        if values.ndim <= 1 or values.shape[-1] != self.width or values.shape[-2] == 0:
            msg = "values must have shape (..., nonempty sources, width)"
            raise ValueError(msg)
        return (self.scores(values).softmax(-1)[..., None] * values).sum(-2)


def attention_branch(
    block: Block,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: KVCache | None,
    *,
    guarded: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reuse the actual GPT attention and both Peri-RMS normalizations."""
    result = block.sa(
        block.ln_1(value), cos, sin, kv_cache=kv_cache, return_guard=guarded
    )
    if guarded:
        branch, valid = result
        return block.attn_out_norm(branch), valid
    return block.attn_out_norm(result)


def feed_forward_branch(block: Block, value: torch.Tensor) -> torch.Tensor:
    """Reuse the actual positionwise SwiGLU and both Peri-RMS normalizations."""
    return block.ffn_out_norm(block.ffwd(block.ln_2(value)))


class ResidualGPT(GPT):
    """Expose full depth aggregation while preserving ordinary GPT as a control."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        block_size: int = 32,
        n_head: int = 4,
        n_layer: int = 4,
        *,
        mixing: str = "conditioned_qk",
        use_checkpoint: bool = True,
        pad_vocab_size_to: int = 64,
        rope_base: float = 10000.0,
        rope_layers: tuple[bool, ...] | None = None,
        centered_xsa: bool = True,
    ) -> None:
        """Initialize common branch weights before drawing any new pseudo-queries."""
        if mixing not in MODES:
            msg = f"Unknown residual mixing: {mixing}"
            raise ValueError(msg)
        super().__init__(
            vocab_size,
            n_embd,
            block_size,
            n_head,
            n_layer,
            use_checkpoint=use_checkpoint,
            pad_vocab_size_to=pad_vocab_size_to,
            rope_base=rope_base,
            rope_layers=rope_layers,
            centered_xsa=centered_xsa,
        )
        self.mixing = mixing
        self.branch_scale = 1 / math.sqrt(2 * n_layer) if mixing == "scaled" else 1.0
        self.branch_logits = (
            nn.Parameter(torch.zeros(2 * n_layer)) if mixing == "gated" else None
        )
        factory = {
            "attnres": KeyAttentionResidual,
            "qk": QKAttentionResidual,
            "conditioned_qk": ConditionedQKAttentionResidual,
        }.get(mixing)
        self.residual_mixers = nn.ModuleList(
            [] if factory is None else [factory(n_embd) for _ in range(2 * n_layer)]
        )

    def scale_branch(self, index: int, normalized_branch: torch.Tensor) -> torch.Tensor:
        """Apply the branch gain after output RMS, retaining its radial derivative."""
        gain = (
            2 * self.branch_logits[index].sigmoid()
            if self.branch_logits is not None
            else self.branch_scale
        )
        return gain * normalized_branch

    def aggregate(self, index: int, sources: list[torch.Tensor]) -> torch.Tensor:
        """Read embedding plus each completed branch once, in causal depth order."""
        if index == 0:
            return sources[0]
        return self.residual_mixers[index - 1](torch.stack(sources, dim=-2))

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Keep generation/cache support; depth mixing uses only the current tokens."""
        if self.mixing == "add":
            return super().forward(idx, targets, kv_cache)
        additive = self.mixing in ("scaled", "gated")
        start = 0 if kv_cache is None else kv_cache.get_pos()
        end = start + idx.shape[1]
        if end > self.block_size:
            msg = f"positions up to {end} exceed block_size {self.block_size}"
            raise ValueError(msg)
        cos, sin = self.rope_cos[start:end], self.rope_sin[start:end]
        x = self.embedding_norm(self.token_embedding_table(idx))
        sources = [x]
        guarded = self.centered_xsa and kv_cache is None
        constant = (idx != idx[:, :1]).cumsum(dim=1) == 0 if guarded else None
        for index, block in enumerate(self.blocks):
            if not additive:
                x = self.aggregate(2 * index, sources)
            if self.training and self.use_checkpoint:
                result = checkpoint(
                    attention_branch,
                    block,
                    x,
                    cos,
                    sin,
                    kv_cache,
                    guarded=guarded,
                    use_reentrant=False,
                )
            else:
                result = attention_branch(block, x, cos, sin, kv_cache, guarded=guarded)
            if guarded:
                branch, valid = result
                constant = constant & valid[:, None]
                # Every prior source is shared on this still-constant prefix.
                # Above clipping, its XSA branch equals the first (zero) branch.
                branch = torch.where(constant[..., None], branch[:, :1], branch)
            else:
                branch = result
            if additive:
                x = x + self.scale_branch(2 * index, branch)
            else:
                sources.append(branch)
                x = self.aggregate(2 * index + 1, sources)
            branch = (
                checkpoint(feed_forward_branch, block, x, use_reentrant=False)
                if self.training and self.use_checkpoint
                else feed_forward_branch(block, x)
            )
            if guarded:
                branch = torch.where(constant[..., None], branch[:, :1], branch)
            if additive:
                x = x + self.scale_branch(2 * index + 1, branch)
            else:
                sources.append(branch)
        if not additive:
            x = self.aggregate(2 * self.n_layer, sources)
        if guarded:
            x = torch.where(constant[..., None], x[:, :1], x)
        logits = self.lm_head(self.ln_f(x))
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        )
        return logits, loss
