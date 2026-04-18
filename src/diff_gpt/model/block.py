import torch
from torch import nn

from diff_gpt.model.mha import MultiHeadAttention
from diff_gpt.model.feed_forward import FeedForward
from diff_gpt.model.rms_norm import rms_norm
from diff_gpt.model.kv_cache import KVCache


def block_attn_res(
    blocks: tuple[torch.Tensor, ...],
    partial_block: torch.Tensor,
    proj: nn.Linear,
) -> torch.Tensor:
    """
    Block Attention Residuals (Chen et al. 2026, eq. 2-5). Given the committed
    block representations `blocks` plus the running partial sum of the current
    block, compute a per-token softmax-weighted sum using a learned per-sub-
    layer pseudo-query `proj.weight` (shape (1, D)).

    Returns a tensor of shape (B, T, D).
    """
    # Short-circuit: if there is only one unique candidate (the layer-0 case
    # where partial_block is the same object as blocks[0]), the softmax is
    # provably uniform over copies of the same tensor, so h is just that
    # tensor. Saves a stack + rms_norm + two matmuls per forward at depth 0.
    if len(blocks) == 1 and partial_block is blocks[0]:
        return partial_block

    V = torch.stack([*blocks, partial_block], dim=0)  # (N+1, B, T, D)
    # RMSNorm inside ϕ(·) — prevents large-magnitude blocks from dominating
    # the attention weights (paper §3.1). A single fused kernel on the
    # stacked V beats per-block caching because the extra stack needed for a
    # pre-normed K costs more than the repeated norm work it would save.
    K = rms_norm(V)
    # logits: K @ w via matmul broadcast over the leading (N+1, B, T) dims.
    # Equivalent to einsum("d,nbtd->nbt", w, K) but lands on a native GEMV.
    w = proj.weight.view(-1)  # (D,)
    logits = K.matmul(w)  # (N+1, B, T)
    weights = logits.softmax(dim=0).unsqueeze(-1)  # (N+1, B, T, 1)
    # unsqueeze + mul + sum beats a batched matmul here: for tiny N+1 the
    # per-(B,T) bmm pays too much dispatch overhead on CPU.
    h = (weights * V).sum(dim=0)  # (B, T, D)
    return h


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        swiglu_alpha: float,
        swiglu_limit: float,
        layer_idx: int,
        layers_per_block: int = 1,
    ) -> None:
        super().__init__()
        assert layers_per_block >= 1
        self.layer_idx = layer_idx
        self.layers_per_block = layers_per_block
        self.sa = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            layer_idx=layer_idx,
        )
        self.ffwd = FeedForward(
            n_embd=n_embd, swiglu_alpha=swiglu_alpha, swiglu_limit=swiglu_limit
        )
        # One pseudo-query per sub-layer (attn and mlp). The Linear's weight
        # (shape (1, n_embd)) serves as the learnable query vector wₗ.
        self.attn_res_proj = nn.Linear(n_embd, 1, bias=False)
        self.mlp_res_proj = nn.Linear(n_embd, 1, bias=False)

    def _commit_if_boundary(
        self,
        blocks: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
        if (
            partial_block is not None
            and (self.layer_idx + 1) % self.layers_per_block == 0
        ):
            blocks = (*blocks, partial_block)
            partial_block = None
        return blocks, partial_block

    def forward(
        self,
        blocks: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: KVCache,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        # 1) Cross-block attn residual → attn sub-layer.
        h = block_attn_res(blocks, partial_block, self.attn_res_proj)
        blocks, partial_block = self._commit_if_boundary(blocks, partial_block)
        attn_out = self.sa(rms_norm(h), freqs_cis=freqs_cis, kv_cache=kv_cache)
        partial_block = (
            attn_out if partial_block is None else partial_block + attn_out
        )

        # 2) Cross-block attn residual → MLP sub-layer.
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj)
        mlp_out = self.ffwd(rms_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block
