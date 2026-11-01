"""Decoder-only GPT: the `BaseGPT` interface and the plain `GPT` model."""

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diff_gpt.domain.model.block import Block
from diff_gpt.domain.model.rms_norm import RMSNorm
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


class GPT(BaseGPT):
    """
    Canonical GPT with parameter-free Peri-RMSNorm, QK-norm, RoPE, XSA and SwiGLU.

    A normalized learned token embedding, Peri-norm blocks on a residual
    stream, a final norm and an lm_head tied to the token embedding.
    Each block normalizes both branch inputs and outputs before residual
    addition, with coefficient 1/sqrt(2*n_layer) after the output norm.
    Positions enter only as rotations of the attention queries and
    keys (RoPE). The attention is
    exclusive (XSA): each head's output is orthogonal to the token's own
    value. The MLP is a SwiGLU with the parameter count of a 4x MLP. Every
    norm is an RMSNorm without learnable parameters, and no Linear layer has
    a bias. The model reads the token stream as ordinary text; everything
    series-specific lives in the encoder/decoder.
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
        Build the layers and initialize the weights.

        `vocab_size` must be a multiple of `pad_vocab_size_to`: the encoder owns
        the vocabulary and pads it. `use_checkpoint` turns on gradient
        checkpointing of the blocks in training mode. `n_layer` must be at
        least 1: the KV cache advances its position in the last block.
        `rope_base`, positive and finite, sets the RoPE frequencies: channel
        pair i turns by rope_base^(-2i / head_dim) radians per position.
        Parameter-free Peri-RMSNorm normalizes the embedding stream and both
        the input and output of every attention and MLP branch.
        Q/K use per-head RMSNorm followed by standard 1/sqrt(head_dim) scaling.
        Centered XSA and shared states for repeated-token prefixes avoid
        cancellation in forward and parameter gradients. Both branches use
        coefficient 1/sqrt(2*n_layer), after their output RMS normalization.
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
            Block(
                n_embd=n_embd,
                n_head=n_head,
                layer_idx=i,
                n_layer=n_layer,
            )
            for i in range(n_layer)
        )
        self.ln_f = RMSNorm(n_embd)
        self.embedding_norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        # Keep the N(0, 0.02) initialization and 1/sqrt(2·n_layer) scaling of
        # c_proj weights. Peri-RMSNorm controls the scale of their outputs.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        self.use_checkpoint = use_checkpoint
        self.branch_scale = 1 / math.sqrt(2 * n_layer)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def estimate_flops(self) -> int:
        """
        Return the model FLOPs per training token, forward and backward.

        6 per parameter plus 12·n_layer·n_embd·block_size for attention over
        the full block (PaLM appendix B, as in nanoGPT and nanochat). The tied
        token embedding counts once, as the lm_head matmul. Elementwise work
        such as RoPE, the norms, the XSA projection and the SwiGLU gate is not
        counted, nor is the recompute under gradient checkpointing.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return 6 * n_params + 12 * self.n_layer * self.n_embd * self.block_size

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply fixed post-RMS residual scaling, with causal cache support."""
        start = 0 if kv_cache is None else kv_cache.get_pos()
        end = start + idx.shape[1]
        if end > self.block_size:
            msg = f"positions up to {end} exceed block_size {self.block_size}"
            raise ValueError(msg)
        cos, sin = self.rope_cos[start:end], self.rope_sin[start:end]
        x = self.embedding_norm(self.token_embedding_table(idx))
        guarded = kv_cache is None
        constant = (idx != idx[:, :1]).cumsum(dim=1) == 0 if guarded else None
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                result = checkpoint(
                    block,
                    x,
                    cos,
                    sin,
                    kv_cache,
                    constant=constant,
                    use_reentrant=False,
                )
            else:
                result = block(x, cos, sin, kv_cache, constant=constant)
            if constant is None:
                x = result
            else:
                x, constant = result
        if constant is not None:
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
