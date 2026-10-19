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
        head: str = "softmax",
        mixture_components: int = 8,
        attn_res_commit_embed_once: bool = False,
        n_channels: int = 1,
        feature_lags: tuple[int, ...] = (),
    ) -> None:
        """
        `head`: "softmax" (tied linear head + softcap, default) or "mixture"
        (discretised logistic mixture over the ordinal vocabulary, see
        `_mixture_log_probs`; `mixture_components` mixture components).
        """
        super().__init__(
            block_size=block_size,
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
        )
        assert head in ("softmax", "mixture"), f"unknown head {head!r}"
        assert mixture_components >= 1
        assert n_channels >= 1
        # Multivariate streams interleave F channels, so absolute position p
        # belongs to channel p % F. n_channels > 1 adds a learned channel
        # embedding indexed that way (windows must then start at channel 0,
        # see DiffDataFrameDataLoader(align_windows=True)); 1 disables it.
        self.n_channels = n_channels
        if n_channels > 1:
            self.channel_embedding = nn.Embedding(n_channels, n_embd)
        # Exact numeric features derived from the token prefix (Chen-style
        # CumSum / LookUp primitives, cf. Percepta transformer-vm): the decoder
        # computes the running integral and the seasonal lags exactly, while a
        # probe shows the model represents them only partially (R² 0.69 / 0.40).
        # Feeding them in removes that work from the attention's plate.
        # Per lag L: the level change over L steps and the token L steps back.
        self.feature_lags = tuple(int(l) for l in feature_lags)
        assert all(l >= 1 for l in self.feature_lags)
        self.n_extra_features = (1 + 2 * len(self.feature_lags)) if self.feature_lags else 0
        if self.n_extra_features:
            self.feature_proj = nn.Linear(self.n_extra_features, n_embd, bias=False)
            nn.init.zeros_(self.feature_proj.weight)  # start as a no-op
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
        self.head = head
        self.mixture_components = mixture_components
        # With the default (False) the normalised embedding is committed as
        # block 0 in forward() AND again by layer 0's boundary commit, so every
        # AttnRes softmax sees two copies of it (doubled weight). True commits
        # it once: forward() starts with no committed blocks and layer 0's
        # boundary commit is the only one.
        self.attn_res_commit_embed_once = attn_res_commit_embed_once
        if head == "softmax":
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding_table.weight
        else:
            # (mixture logit, mean, log-scale) per component, all in bin units.
            self.mix_head = nn.Linear(n_embd, 3 * mixture_components)
            # Bin q covers [q - V/2 - 1/2, q - V/2 + 1/2) in bin units.
            self.register_buffer(
                "bin_edges",
                torch.arange(vocab_size + 1, dtype=torch.float32) - vocab_size / 2 - 0.5,
                persistent=False,
            )
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
        )  # (T, hs/2, 2)
        freqs_cis = freqs_cis.view(1, T, 1, -1, 2)
        x = self.token_embedding_table(idx)  # (B, T, C)
        if self.n_extra_features:
            hist = (
                kv_cache.record_tokens(idx, T_start) if kv_cache is not None else idx
            )
            x = x + self.feature_proj(self._token_features(hist, T_start, T_end))
        if self.n_channels > 1:
            pos = torch.arange(T_start, T_end, device=idx.device)
            x = x + self.channel_embedding(pos % self.n_channels)
        x = rms_norm(x)
        # Block AttnRes state: the normalized embedding is committed as block 0
        # (v₀ = h₁ in paper notation); partial_block starts as the same embed
        # so each sub-layer's cross-block attention has at least one key.
        blocks: tuple[torch.Tensor, ...] = () if self.attn_res_commit_embed_once else (x,)
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
        if self.head == "softmax":
            logits = self.lm_head(x)  # (B, T, vocab_size)
            if self.logit_softcap > 0:
                # Smoothly bound logits to [-softcap, softcap]; stabilizes training
                # and reduces sampling temperature collapse (Gemma-style).
                cap = self.logit_softcap
                logits = cap * torch.tanh(logits / cap)
        elif targets is not None and self.label_smoothing_sigma == 0:
            # Training fast path: only the target cell's mass is needed, K
            # sigmoids per position instead of V·K (the full (B, T, V+1, K)
            # tensor is the largest in the model and dominates step time).
            return None, self._mixture_target_nll(x, targets)
        else:
            logits = self._mixture_log_probs(x)  # (B, T, vocab_size) log-probs
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

    def _token_features(self, hist: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Exact functions of the token prefix, for positions [start, end).

        `hist`: (B, N) int64 tokens at absolute positions [0, N), N >= end.
        Returns (B, end - start, 1 + 2·len(feature_lags)):
        the running integral of the tokens (what the decoder integrates), and
        per lag L the level change over L steps and the token L steps back.
        All in units of the vocabulary so the scale is O(1).
        """
        q = (hist.to(torch.float32) - self.vocab_size / 2.0) / self.vocab_size
        level = q.cumsum(dim=1)
        feats = [level]
        N = hist.size(1)
        for lag in self.feature_lags:
            if lag < N:
                pad = torch.zeros(hist.size(0), lag, dtype=level.dtype, device=hist.device)
                level_lag = torch.cat([pad, level[:, :-lag]], dim=1)
                q_lag = torch.cat([pad, q[:, :-lag]], dim=1)
            else:
                level_lag = torch.zeros_like(level)
                q_lag = torch.zeros_like(q)
            feats.append(level - level_lag)
            feats.append(q_lag)
        return torch.stack(feats, dim=-1)[:, start:end]

    def _mixture_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Discretised logistic mixture over the ordinal vocabulary.

        The next token is round(u) of a continuous quantity u (scaled
        derivative plus the sub-bin position of the current level), so the
        head models a density over u with a mixture of logistics and assigns
        each bin the mass of its cell [q - 1/2, q + 1/2). The two edge bins
        absorb the open tails. Returns log-probabilities of shape (B, T, V),
        usable wherever the softmax head's logits are.
        """
        pi_logits, mu, log_s = self.mix_head(x).float().chunk(3, dim=-1)  # (B, T, K)
        inv_s = torch.exp(-log_s.clamp(min=-5.0))
        edges = self.bin_edges.view(1, 1, -1, 1)  # (1, 1, V+1, 1)
        z = (edges - mu.unsqueeze(-2)) * inv_s.unsqueeze(-2)  # (B, T, V+1, K)
        cdf = torch.sigmoid(z)
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1, :]), cdf[..., 1:-1, :], torch.ones_like(cdf[..., :1, :])],
            dim=-2,
        )
        mass = cdf[..., 1:, :] - cdf[..., :-1, :]  # (B, T, V, K)
        log_pi = torch.log_softmax(pi_logits, dim=-1).unsqueeze(-2)  # (B, T, 1, K)
        return torch.logsumexp(torch.log(mass.clamp_min(1e-12)) + log_pi, dim=-1)

    def _mixture_target_nll(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean −log p(target bin) under the mixture, equal to cross_entropy on
        `_mixture_log_probs` but computed for the target cell only."""
        pi_logits, mu, log_s = self.mix_head(x).float().chunk(3, dim=-1)  # (B, T, K)
        inv_s = torch.exp(-log_s.clamp(min=-5.0))
        mask = targets != -100
        if not mask.any():
            return pi_logits.sum() * 0.0
        tgt = targets.clamp(min=0)
        V = self.vocab_size
        lo = (tgt.to(torch.float32) - V / 2 - 0.5).unsqueeze(-1)  # lower edge, (B, T, 1)
        hi = lo + 1.0
        cdf_lo = torch.sigmoid((lo - mu) * inv_s)
        cdf_hi = torch.sigmoid((hi - mu) * inv_s)
        # Open tails: bin 0 starts at −∞, bin V−1 ends at +∞.
        cdf_lo = torch.where((tgt == 0).unsqueeze(-1), torch.zeros_like(cdf_lo), cdf_lo)
        cdf_hi = torch.where((tgt == V - 1).unsqueeze(-1), torch.ones_like(cdf_hi), cdf_hi)
        mass = cdf_hi - cdf_lo  # (B, T, K)
        log_p = torch.logsumexp(torch.log(mass.clamp_min(1e-12)) + torch.log_softmax(pi_logits, dim=-1), dim=-1)
        return -(log_p[mask]).mean()

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        sampler: Sampler | None,
        seed: int | None = None,
    ) -> torch.Tensor:
        block_size = self.block_size
        assert not self.n_extra_features, (
            "generate() slides a window and would recompute token features from a "
            "truncated prefix; use Engine.generate, which keeps the full history"
        )
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
            # Keep the sliding window phase-aligned so position 0 is channel 0.
            start_slice += (-start_slice) % self.n_channels
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
