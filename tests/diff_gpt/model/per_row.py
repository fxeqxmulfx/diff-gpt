"""
The per-row decode path, a reference for the shared-prefix tests.

Before KVCache.share_prefix, the engine copied each prompt's cached
positions into the rows of its samples (KVCache.prefill), and every decode
step read each copy. These helpers rerun that path, and forward passes
without a cache, on the tokens a shared-prefix run drew, so a test can
compare the logits of every step and draw the tokens again from them.
"""

from typing import TYPE_CHECKING

import torch

from diff_gpt.model.kv_cache import KVCache
from diff_gpt.sampler.sampler import Sampler

if TYPE_CHECKING:
    from diff_gpt.model.gpt import GPT


class RecordingSampler(Sampler):
    """Draw each row's token by multinomial, keeping the logits and the draws."""

    def __init__(self) -> None:
        """Start with no step recorded."""
        self.logits: list[torch.Tensor] = []
        self.tokens: list[torch.Tensor] = []

    def __call__(
        self,
        logits: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return a (B, 1) column of tokens drawn from softmax(logits) with `rng`."""
        ids = torch.multinomial(logits.softmax(dim=-1), num_samples=1, generator=rng)
        self.logits.append(logits.clone())
        self.tokens.append(ids)
        return ids


def _cache(model: GPT, batch_size: int, seq_len: int) -> KVCache:
    return KVCache(
        batch_size=batch_size,
        num_heads=model.n_head,
        seq_len=seq_len,
        head_dim=model.n_embd // model.n_head,
        num_layers=model.n_layer,
    )


@torch.inference_mode()
def per_row_logits(
    model: GPT,
    prompts: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Return the logits (steps, R, V) of the per-row path fed `tokens`.

    `prompts` (N, C) are prefilled once and copied into the R = N·S rows of
    their samples, sample s of prompt i at row i·S + s; then each step feeds
    a column of `tokens` (R, steps) through that cache. Step t holds the
    float32 logits after the first t tokens, as a sampler gets them.
    """
    num_prompts, context = prompts.shape
    rows, steps = tokens.shape
    prefill = _cache(model, num_prompts, context)
    logits, _ = model(prompts, kv_cache=prefill)
    copies = _cache(model, rows, context + steps)
    copies.prefill(prefill)
    out = [logits[:, -1].float().repeat_interleave(rows // num_prompts, dim=0)]
    for t in range(steps - 1):
        logits, _ = model(tokens[:, t : t + 1], kv_cache=copies)
        out.append(logits[:, -1].float())
    return torch.stack(out)


@torch.inference_mode()
def full_forward_logits(
    model: GPT,
    prompts: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Return the logits (steps, R, V) of forward passes without a cache.

    Step t runs each row's prompt and its first t tokens through the model
    in one pass, as training does, and keeps the last position's logits.
    """
    rows, steps = tokens.shape
    context = prompts.size(1)
    sequences = torch.cat(
        (prompts.repeat_interleave(rows // len(prompts), dim=0), tokens),
        dim=1,
    )
    return torch.stack(
        [model(sequences[:, : context + t])[0][:, -1].float() for t in range(steps)],
    )


def draw_again(logits: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Draw tokens (R, steps) from the logits (steps, R, V) of every step.

    One generator seeded with `seed` serves all the steps, as in the engine.
    """
    rng = torch.Generator(device=logits.device)
    rng.manual_seed(seed)
    sampler = RecordingSampler()
    return torch.cat([sampler(step, rng=rng) for step in logits], dim=1)
