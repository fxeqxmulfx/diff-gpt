from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diff_gpt.sampler.sampler import Sampler
from diff_gpt.model.block import Block
from diff_gpt.model.rope import precompute_freqs_cis
from diff_gpt.model.rms_norm import rms_norm
from diff_gpt.model.kv_cache import KVCache


def _gaussian_soft_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sigma: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """CE with Gaussian soft targets over bin indices — respects ordinality.

    p_soft[i] ∝ exp(-(i - y)^2 / (2σ²)); σ in bin units. σ→0 → one-hot (plain CE).
    """
    mask = targets != ignore_index
    if not mask.any():
        return logits.sum() * 0.0
    valid_logits = logits[mask]
    valid_targets = targets[mask]
    C = logits.shape[-1]
    bins = torch.arange(C, device=logits.device, dtype=valid_logits.dtype)
    diff = bins.unsqueeze(0) - valid_targets.unsqueeze(1).to(valid_logits.dtype)
    soft = F.softmax(-diff.pow(2) / (2.0 * sigma * sigma), dim=-1)
    log_probs = F.log_softmax(valid_logits, dim=-1)
    return -(soft * log_probs).sum(dim=-1).mean()


class BaseGPT(nn.Module, ABC):
    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
    ) -> None:
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
        pass

    @abstractmethod
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        sampler: Sampler | None,
        seed: int | None = None,
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def device_type(self) -> str:
        pass


class GPT(BaseGPT):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        block_size: int = 32,
        n_head: int = 4,
        n_layer: int = 4,
        rope_theta: float = 10000,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
        use_checkpoint: bool = True,
        logit_softcap: float = 15.0,
        pad_vocab_size_to: int = 64,
        attn_res_layers_per_block: int = 1,
        label_smoothing_sigma: float = 0.0,
    ) -> None:
        super().__init__(
            block_size=block_size,
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
        )
        assert logit_softcap >= 0, "logit_softcap must be non-negative (0 disables it)"
        assert label_smoothing_sigma >= 0, (
            "label_smoothing_sigma must be non-negative (0 disables it)"
        )
        assert pad_vocab_size_to >= 1, "pad_vocab_size_to must be >= 1"
        assert attn_res_layers_per_block >= 1, (
            "attn_res_layers_per_block must be >= 1"
        )
        # GPT does not pad — the encoder owns vocab_size, so the caller is
        # responsible for choosing a tensor-core-friendly multiple.
        assert vocab_size % pad_vocab_size_to == 0, (
            f"vocab_size ({vocab_size}) must be a multiple of "
            f"pad_vocab_size_to ({pad_vocab_size_to}); pad it in the encoder"
        )
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            Block(
                n_embd=n_embd,
                n_head=n_head,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                layer_idx=i,
                layers_per_block=attn_res_layers_per_block,
            )
            for i in range(n_layer)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(n_embd // n_head, block_size * 2, theta=rope_theta),
            persistent=False,
        )
        self.use_checkpoint = use_checkpoint
        self.logit_softcap = logit_softcap
        self.label_smoothing_sigma = label_smoothing_sigma

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        T_start = 0 if kv_cache is None else kv_cache.get_pos()
        T_end = T_start + T
        assert T_end <= self.freqs_cis.shape[0]  # pyright: ignore[reportIndexIssue]
        freqs_cis = (
            self.freqs_cis[T_start:T_end]  # pyright: ignore[reportIndexIssue]
        )  # (T, hs/2)
        freqs_cis = freqs_cis.view(1, T, 1, -1)
        x = self.token_embedding_table(idx)  # (B, T, C)
        x = rms_norm(x)
        # Block AttnRes state: the normalized embedding is committed as block 0
        # (v₀ = h₁ in paper notation); partial_block starts as the same embed
        # so each sub-layer's cross-block attention has at least one key.
        blocks: tuple[torch.Tensor, ...] = (x,)
        partial_block: torch.Tensor = x
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                out = checkpoint(
                    block,
                    blocks,
                    partial_block,
                    freqs_cis=freqs_cis,
                    kv_cache=kv_cache,
                    use_reentrant=False,
                )
                blocks, partial_block = out  # type: ignore[misc]
            else:
                blocks, partial_block = block(
                    blocks,
                    partial_block,
                    freqs_cis=freqs_cis,
                    kv_cache=kv_cache,
                )
        # Final residual-stream value is the current block's partial sum.
        x = rms_norm(partial_block)  # (B, T, C) # pyright: ignore[reportArgumentType]
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if self.logit_softcap > 0:
            # Smoothly bound logits to [-softcap, softcap]; stabilizes training
            # and reduces sampling temperature collapse (Gemma-style).
            cap = self.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            _logits = logits.view(B * T, C)
            _targets = targets.view(B * T)
            if self.label_smoothing_sigma > 0:
                loss = _gaussian_soft_ce(
                    logits=_logits,
                    targets=_targets,
                    sigma=self.label_smoothing_sigma,
                )
            else:
                loss = F.cross_entropy(_logits, _targets)
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        sampler: Sampler | None,
        seed: int | None = None,
    ) -> torch.Tensor:
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
                    probs, num_samples=1, generator=rng
                )  # (B, 1)
            all_tokens[:, current_pos] = idx_next.squeeze()
        return all_tokens

    @property
    def device_type(self) -> str:
        result = str(self.freqs_cis.device.type)
        return result
