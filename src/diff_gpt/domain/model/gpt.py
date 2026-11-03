"""Decoder-only GPT: the `BaseGPT` interface and the plain `GPT` model."""

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diff_gpt.domain.model.mini_block import (
    LowEpsilonQKAttention,
    MiniBlock,
    MiniRMSNorm,
    fix_qk_temperature,
    initialize_swiglu,
    repair_rms_backward,
    use_native_rms_epsilon,
)
from diff_gpt.domain.model.rope import rope_tables

if TYPE_CHECKING:
    from diff_gpt.domain.model.kv_cache import KVCache
    from diff_gpt.domain.sampler.sampler import Sampler


class BaseGPT(nn.Module, ABC):
    """Interface shared by GPT models: size attributes, forward pass and sampling."""

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
    ) -> None:
        """Store the model sizes."""
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

    @abstractmethod
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Return the logits for `idx` and the loss (None when `targets` is None).

        `kv_cache`, when given, holds the keys/values of earlier positions and
        receives those of `idx`.
        """

    @abstractmethod
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        sampler: Sampler | None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Extend `idx` by `max_new_tokens` sampled tokens and return the result.

        `sampler` draws each new token from the logits; `seed`, when given,
        makes the draws reproducible.
        """

    @property
    @abstractmethod
    def device_type(self) -> str:
        """Return the device type of the model weights, such as `cpu` or `cuda`."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device of the model weights, with its index (`cuda:1`)."""


class GPTMini(BaseGPT):
    """
    Historical GPTMini with the forecasting and KV-cache interfaces.

    PreNorm with native RMSNorm (epsilon 1e-5), unit Q/K and a learned
    per-head temperature, RoPE, ordinary XSA, ReLU squared with width 4d,
    tied token/readout weights and one Normal(0.02) draw per matrix.
    """

    # RoPE tables of positions 0..block_size-1, registered in __init__
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        block_size: int = 32,
        n_head: int = 4,
        n_layer: int = 4,
        *,
        use_checkpoint: bool = True,
        pad_vocab_size_to: int = 64,
        rope_base: float = 10_000.0,
    ) -> None:
        """
        Build GPTMini, preserving historical parameter and random-draw order.

        Vocabulary alignment is caller-owned; use pad_vocab_size_to=1 for
        the unpadded historical Basis vocabulary. Checkpointing and KV caching
        preserve the same model equations. The archived normalized GPT remains
        available explicitly in normalized_gpt for replay of earlier research.
        """
        if n_layer < 1:
            msg = f"n_layer must be >= 1, got {n_layer}"
            raise ValueError(msg)
        if not 0 < rope_base < math.inf:
            msg = f"rope_base must be positive and finite, got {rope_base}"
            raise ValueError(msg)
        super().__init__(
            block_size=block_size,
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
        )
        if pad_vocab_size_to < 1:
            msg = "pad_vocab_size_to must be >= 1"
            raise ValueError(msg)
        # GPT does not pad — the encoder owns vocab_size, so the caller is
        # responsible for choosing a tensor-core-friendly multiple.
        if vocab_size % pad_vocab_size_to != 0:
            msg = (
                f"vocab_size ({vocab_size}) must be a multiple of "
                f"pad_vocab_size_to ({pad_vocab_size_to}); pad it in the encoder"
            )
            raise ValueError(msg)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # RoPE tables, shared by all layers; they follow from the sizes, so
        # they stay out of the state dict.
        cos, sin = rope_tables(n_embd // n_head, block_size, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.blocks = nn.ModuleList(
            MiniBlock(
                n_embd=n_embd,
                n_head=n_head,
                layer_idx=i,
            )
            for i in range(n_layer)
        )
        self.ln_f = MiniRMSNorm()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        # Initialize each distinct matrix once, in parameter order. Applying
        # an initializer to modules would redraw the tied embedding/readout.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.normal_(parameter, mean=0.0, std=0.02)
        self.use_checkpoint = use_checkpoint

    def estimate_flops(self) -> int:
        """
        Return the model FLOPs per training token, forward and backward.

        6 per parameter plus 12·n_layer·n_embd·block_size for attention over
        the full block (PaLM appendix B, as in nanoGPT and nanochat). The tied
        token embedding counts once, as the lm_head matmul. Elementwise work
        such as RoPE, the norms, the XSA projection and ReLU squared is not
        counted, nor is the recompute under gradient checkpointing.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.ndim > 1)
        return 6 * n_params + 12 * self.n_layer * self.n_embd * self.block_size

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        *,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply historical PreNorm updates, with causal caching.

        Optional (batch, count) positions select readout rows after the final
        norm, as in the original Basis trainer; targets must then have that
        same shape. Otherwise every input position receives logits.
        """
        start = 0 if kv_cache is None else kv_cache.get_pos()
        end = start + idx.shape[1]
        if end > self.block_size:
            msg = f"positions up to {end} exceed block_size {self.block_size}"
            raise ValueError(msg)
        cos, sin = self.rope_cos[start:end], self.rope_sin[start:end]
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(block, x, cos, sin, kv_cache, use_reentrant=False)
            else:
                x = block(x, cos, sin, kv_cache)
        x = self.ln_f(x)
        if positions is not None:
            x = x.gather(1, positions[..., None].expand(-1, -1, self.n_embd))
        logits = self.lm_head(x)
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        sampler: Sampler | None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Extend `idx` by `max_new_tokens` sampled tokens and return the result.

        Each step re-runs the model on the last `block_size` tokens, without a
        KV cache. Without `sampler`, tokens are drawn from the softmax of the
        logits; `seed` seeds a dedicated generator.
        """
        block_size = self.block_size
        device = idx.device
        rng: torch.Generator | None = None
        if seed is not None:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        B, T = idx.shape
        all_tokens = torch.full(
            (B, T + max_new_tokens),
            0,
            dtype=torch.int64,
            device=device,
        )
        all_tokens[:, :T] = idx[:, :T]
        for current_pos in range(T, T + max_new_tokens):
            start_slice = max(0, current_pos - block_size)
            idx_cond = all_tokens[:, start_slice:current_pos]
            logits, _ = self(idx_cond)
            logits = logits[:, -1]  # (B, C)
            if sampler is not None:
                idx_next = sampler(logits, rng=rng)
            else:
                probs = F.softmax(logits, dim=-1)  # (B, C)
                idx_next = torch.multinomial(
                    probs,
                    num_samples=1,
                    generator=rng,
                )  # (B, 1)
            all_tokens[:, current_pos] = idx_next.squeeze()
        return all_tokens

    @property
    def device_type(self) -> str:
        """Return the device type of the model weights, such as `cpu` or `cuda`."""
        return str(self.lm_head.weight.device.type)

    @property
    def device(self) -> torch.device:
        """Return the device of the model weights, with its index (`cuda:1`)."""
        return self.lm_head.weight.device


class GPT(GPTMini):
    """Passing SwiGLU GPTMini with repaired RMS, fixed Q/K gain and XSA clipping."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        block_size: int = 32,
        n_head: int = 4,
        n_layer: int = 4,
        *,
        use_checkpoint: bool = True,
        pad_vocab_size_to: int = 64,
        rope_base: float = 10_000.0,
    ) -> None:
        """Keep historical weights and install the passing recall features."""
        super().__init__(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            use_checkpoint=use_checkpoint,
            pad_vocab_size_to=pad_vocab_size_to,
            rope_base=rope_base,
        )
        initialize_swiglu(self.blocks, n_embd)
        repair_rms_backward(self, n_embd)
        for block in self.blocks:
            block.sa = LowEpsilonQKAttention(block.sa)
            fix_qk_temperature(block.sa)
        use_native_rms_epsilon(self)
