import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.model.rope import apply_rotary_emb
from diff_gpt.model.kv_cache import KVCache
from diff_gpt.model.rms_norm import rms_norm


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, layer_idx: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        # RoPE requires head_dim to be even (pairs of dims are rotated as complex).
        assert (n_embd // n_head) % 2 == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.layer_idx = layer_idx
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_g = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        B, T, C = x.size()
        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        g = self.c_g(x)
        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        q, k = (
            apply_rotary_emb(q, freqs_cis),
            apply_rotary_emb(k, freqs_cis),
        )  # QK rotary embedding
        q, k = rms_norm(q), rms_norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)  # number of queries in this forward pass
        Tk = k.size(
            2
        )  # number of keys/values in total (in the cache + current forward pass)
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.bool, device=q.device
            )  # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = y * F.sigmoid(g)
        y = self.c_proj(y)
        return y
