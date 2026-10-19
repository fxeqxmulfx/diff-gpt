import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from diff_gpt.sampler.sampler import Sampler
from diff_gpt.model.block import Block
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
        use_checkpoint: bool = True,
        pad_vocab_size_to: int = 64,
    ) -> None:
        """
        Plain GPT-2 (as in nanoGPT): learned token and position embeddings,
        pre-LayerNorm blocks on a residual stream, a final LayerNorm and an
        lm_head tied to the token embedding. The model reads the token stream
        as ordinary text; everything series-specific lives in the
        encoder/decoder.
        """
        super().__init__(
            block_size=block_size,
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
        )
        assert pad_vocab_size_to >= 1, "pad_vocab_size_to must be >= 1"
        # GPT does not pad — the encoder owns vocab_size, so the caller is
        # responsible for choosing a tensor-core-friendly multiple.
        assert vocab_size % pad_vocab_size_to == 0, (
            f"vocab_size ({vocab_size}) must be a multiple of "
            f"pad_vocab_size_to ({pad_vocab_size_to}); pad it in the encoder"
        )
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            Block(
                n_embd=n_embd,
                n_head=n_head,
                layer_idx=i,
            )
            for i in range(n_layer)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        # GPT-2 init: N(0, 0.02) weights and zero biases; the two projections
        # that write into the residual stream are scaled by 1/sqrt(2·n_layer)
        # so its variance does not grow with depth.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("sa.c_proj.weight", "ffwd.net.2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        T_start = 0 if kv_cache is None else kv_cache.get_pos()
        T_end = T_start + T
        assert T_end <= self.block_size, (
            f"positions up to {T_end} exceed block_size {self.block_size}"
        )
        pos = torch.arange(T_start, T_end, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(pos)  # (T, C)
        x = tok_emb + pos_emb
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint(block, x, kv_cache=kv_cache, use_reentrant=False)
            else:
                x = block(x, kv_cache=kv_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
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
        result = str(self.lm_head.weight.device.type)
        return result
