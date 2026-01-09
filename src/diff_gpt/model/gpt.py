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
    ) -> None:
        super().__init__(
            block_size=block_size,
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
        )
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            Block(
                n_embd=n_embd,
                n_head=n_head,
                swiglu_alpha=swiglu_alpha,
                swiglu_limit=swiglu_limit,
                layer_idx=i,
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
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(
                    block,
                    x,
                    freqs_cis=freqs_cis,
                    kv_cache=kv_cache,
                    use_reentrant=False,
                )  # (B, T, C)
            else:
                x = block(
                    x,
                    freqs_cis=freqs_cis,
                    kv_cache=kv_cache,
                )
        x = rms_norm(x)  # (B, T, C) # pyright: ignore[reportArgumentType]
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            _logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(_logits, targets)
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        sampler: Sampler | None,
    ) -> torch.Tensor:
        block_size = self.block_size
        device = idx.device
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
                idx_next = sampler(logits)
            else:
                probs = F.softmax(logits, dim=-1)  # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            all_tokens[:, current_pos] = idx_next.squeeze()
        return all_tokens

    @property
    def device_type(self) -> str:
        result = str(self.freqs_cis.device.type)
        return result
