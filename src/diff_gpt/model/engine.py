"""KV-cached sampling engine: one prompt prefill, then batched decoding."""

from collections import deque
from typing import TYPE_CHECKING, TypedDict, Unpack

import torch

from diff_gpt.model.kv_cache import KVCache

if TYPE_CHECKING:
    from collections.abc import Generator

    from diff_gpt.model.gpt import BaseGPT
    from diff_gpt.sampler.sampler import Sampler


class _GenerateKwargs(TypedDict, total=False):
    """Keyword arguments of `Engine.generate` that `generate_batch` forwards."""

    max_tokens: int | None
    sampler: Sampler | None
    seed: int
    force_schedule: list[int | None] | None


class RowState:
    """Per-row state tracking during generation."""

    __slots__ = (
        "current_tokens",
        "forced_tokens",
    )

    def __init__(self, current_tokens: list[int] | None = None) -> None:
        """Start the row at `current_tokens` with an empty queue of forced tokens."""
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()  # Queue of tokens to force inject


class Engine:
    """Sample continuations of a prompt from a `BaseGPT` with a KV cache."""

    __slots__ = ("model",)

    def __init__(self, model: BaseGPT) -> None:
        """Store the model to sample from."""
        self.model = model

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        num_samples: int = 1,
        max_tokens: int | None = None,
        sampler: Sampler | None = None,
        seed: int = 42,
        force_schedule: list[int | None] | None = None,
    ) -> Generator[tuple[list[int], list[int]]]:
        """
        Streaming generation: single prefill, then clone the KV cache for each sample.

        `force_schedule`, when provided, is a per-step list aligned with the
        generation loop:
          - entry is `None` → sample normally at that step
          - entry is an `int` → teacher-force that token at that step (across
            every sample in the batch)
        Length must be at least `max_tokens`. Useful for TimeXer-style
        exogenous forecasting where specific positions (covariate columns)
        are known a priori.

        Position embeddings cover `block_size` positions, so prompt and
        generated tokens together must fit in them: a longer prompt, or a
        `max_tokens` past `block_size - len(tokens)`, raises ValueError.
        `max_tokens=None` generates up to that limit.
        """
        if not isinstance(tokens, list):
            msg = "tokens must be a list of ints"
            raise TypeError(msg)
        if len(tokens) == 0:
            msg = "tokens must be non-empty"
            raise ValueError(msg)
        if not all(isinstance(t, int) for t in tokens):
            msg = "tokens must be a list of ints"
            raise TypeError(msg)

        # Token budget: position embeddings cover block_size positions.
        block_size = self.model.block_size
        prompt_len = len(tokens)
        if prompt_len > block_size:
            msg = f"prompt has {prompt_len} tokens but block_size is {block_size}"
            raise ValueError(msg)
        hard_ceiling = block_size - prompt_len
        if max_tokens is None:
            effective_max_tokens = hard_ceiling
        elif max_tokens < 0:
            msg = f"max_tokens must be >= 0, got {max_tokens}"
            raise ValueError(msg)
        elif max_tokens > hard_ceiling:
            msg = (
                f"max_tokens {max_tokens} exceeds the {hard_ceiling} positions "
                f"left after a {prompt_len}-token prompt (block_size {block_size})"
            )
            raise ValueError(msg)
        else:
            effective_max_tokens = max_tokens
        if effective_max_tokens == 0:
            return  # nothing to generate; skip expensive prefill+alloc
        if force_schedule is not None and len(force_schedule) < effective_max_tokens:
            msg = (
                "force_schedule must cover all generation steps "
                f"(len={len(force_schedule)} < {effective_max_tokens})"
            )
            raise ValueError(msg)

        device = self.model.device
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model
        kv_model_kwargs = {
            "num_heads": m.n_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, _ = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = prompt_len + effective_max_tokens
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while num_generated < effective_max_tokens:
            # Is this step position-forced by the schedule?
            scheduled_token: int | None = None
            if force_schedule is not None:
                scheduled_token = force_schedule[num_generated]

            if scheduled_token is None:
                # Sample the next token for each row
                if sampler is not None:
                    next_ids = sampler(logits, rng=rng)  # (B, 1)
                else:
                    probs = torch.softmax(logits, dim=-1)  # (B, C)
                    next_ids = torch.multinomial(
                        probs,
                        num_samples=1,
                        generator=rng,
                    )  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()
            else:
                # Position-forced: skip the sampler entirely, reuse across samples.
                sampled_tokens = [scheduled_token] * num_samples

            # Process each row: choose the next token, update state, optional tool use
            token_column = []  # contains the next token id along each row
            token_masks = []  # the mask along each row: sampled (1) or forced (0)
            for i, state in enumerate(row_states):
                # Per-row deque-forcing still wins over sampling (but the
                # position-forced path above has already replaced the sampled
                # values, so this just carries them through).
                is_deque_forced = len(state.forced_tokens) > 0
                # mask: 0 if the token was forced (either path), 1 if sampled
                forced = is_deque_forced or scheduled_token is not None
                token_masks.append(0 if forced else 1)
                next_token = (
                    state.forced_tokens.popleft()
                    if is_deque_forced
                    else sampled_tokens[i]
                )
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(
                1,
            )
            logits, _ = self.model.forward(ids, kv_cache=kv_cache_decode)
            logits = logits[:, -1, :]  # (B, vocab_size)

    def generate_batch(
        self,
        tokens: list[int],
        num_samples: int = 1,
        **kwargs: Unpack[_GenerateKwargs],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Run non-streaming batch generation and return the final token sequences.

        Returns (results, masks): each a list of `num_samples` lists of ints.
        `kwargs` are forwarded to `generate`.
        """
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(
                zip(token_column, token_masks, strict=True),
            ):
                results[i].append(token)
                masks[i].append(mask)
        return results, masks
