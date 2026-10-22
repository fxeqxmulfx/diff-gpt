"""KV-cached sampling engine: one prefill, then batched decoding."""

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
    """Sample continuations of prompts from a `BaseGPT` with a KV cache."""

    __slots__ = ("model",)

    def __init__(self, model: BaseGPT) -> None:
        """Store the model to sample from."""
        self.model = model

    def _prefill(
        self,
        ids: torch.Tensor,
        num_samples: int,
        new_positions: int,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Prefill the prompts `ids` (N, C) for the rows that continue them.

        Return the logits (N, 1, V) of each prompt's last position and the
        cache of the N·num_samples rows, sample s of prompt i at row
        i·num_samples + s, with room for `new_positions` more positions per
        row. One sample per prompt continues in the prompt's own cache, the
        per-row path. More share the prompt's cached positions instead
        (KVCache.share_prefix): the cache holds only the rows' own
        positions, after a view of the prompts' keys and values.
        """
        m = self.model
        shape = {
            "num_heads": m.n_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }
        prompts, prompt_len = ids.shape
        if num_samples == 1:
            kv_cache = KVCache(
                batch_size=prompts,
                seq_len=prompt_len + new_positions,
                **shape,
            )
            logits, _ = m.forward(ids, kv_cache=kv_cache)
            return logits[:, -1:, :], kv_cache
        prefill = KVCache(batch_size=prompts, seq_len=prompt_len, **shape)
        logits, _ = m.forward(ids, kv_cache=prefill)
        kv_cache = KVCache(
            batch_size=prompts * num_samples,
            seq_len=new_positions,
            **shape,
        )
        kv_cache.share_prefix(prefill)
        return logits[:, -1:, :], kv_cache

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
        Streaming generation: a single prefill, which the rows of the samples share.

        `force_schedule`, when provided, is a per-step list aligned with the
        generation loop:
          - entry is `None` → sample normally at that step
          - entry is an `int` → teacher-force that token at that step (across
            every sample in the batch)
        Length must be at least `max_tokens`. Useful for TimeXer-style
        exogenous forecasting where specific positions (covariate columns)
        are known a priori.

        The model covers `block_size` positions (its RoPE tables end there),
        so prompt and generated tokens together must fit in them: a longer
        prompt, or a `max_tokens` past `block_size - len(tokens)`, raises
        ValueError. `max_tokens=None` generates up to that limit.
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

        # Token budget: the model covers block_size positions.
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

        # 1) Run a batch 1 prefill of the prompt tokens; 2) the cache of the
        # rows of the samples, with room for every step's token
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, kv_cache_decode = self._prefill(ids, num_samples, effective_max_tokens)
        # The sampler gets float32 logits whatever the model's dtype: a float16
        # or bfloat16 softmax and multinomial would round the probabilities.
        logits = logits[:, -1, :].float().expand(num_samples, -1)  # (num_samples, V)

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
            logits = logits[:, -1, :].float()  # (B, vocab_size)

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

    @torch.inference_mode()
    def generate_many(
        self,
        prompts: list[list[int]],
        max_tokens: int,
        sampler: Sampler | None = None,
        seed: int = 42,
        num_samples: int = 1,
    ) -> list[list[int]]:
        """
        Continue each prompt by `max_tokens` tokens; return the new tokens of each.

        The prompts are the rows of one batch: one prefill covers them all and
        each step decodes the next token of every row, so many prompts take as
        many model calls as one. They must have one length, and the prompt and
        its continuation must fit in `block_size`; ValueError otherwise. The
        rows draw from one generator seeded with `seed` (multinomial when
        `sampler` is None), so their samples differ from those of `generate`.

        `num_samples`: continuations drawn per prompt. The prompts are
        prefilled once, and the rows of a prompt's samples share its cached
        positions instead of copies of them (KVCache.share_prefix), so a
        decode step reads each prompt's keys and values once; one sample per
        prompt decodes in the prompt's own cache. The result holds sample s
        of prompt i at i·num_samples + s.
        """
        if num_samples < 1:
            msg = f"num_samples must be >= 1, got {num_samples}"
            raise ValueError(msg)
        if not prompts or not all(prompts):
            msg = "prompts must be non-empty lists of ints"
            raise ValueError(msg)
        if not all(isinstance(t, int) for prompt in prompts for t in prompt):
            msg = "prompts must be lists of ints"
            raise TypeError(msg)
        lengths = sorted({len(prompt) for prompt in prompts})
        if len(lengths) > 1:
            msg = f"prompts must have one length, got lengths {lengths}"
            raise ValueError(msg)
        prompt_len = lengths[0]
        room = self.model.block_size - prompt_len
        if room < 0:
            msg = (
                f"prompts have {prompt_len} tokens but block_size is "
                f"{self.model.block_size}"
            )
            raise ValueError(msg)
        if not 0 <= max_tokens <= room:
            msg = (
                f"max_tokens {max_tokens} must lie in [0, {room}], the positions "
                f"left after a {prompt_len}-token prompt "
                f"(block_size {self.model.block_size})"
            )
            raise ValueError(msg)
        if max_tokens == 0:
            return [[] for _ in range(len(prompts) * num_samples)]
        device = self.model.device
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        m = self.model
        ids = torch.tensor(prompts, dtype=torch.long, device=device)
        # The decode cache has room for the max_tokens - 1 tokens fed back.
        logits, kv_cache = self._prefill(ids, num_samples, max_tokens - 1)
        logits = logits.repeat_interleave(num_samples, dim=0)  # (B, 1, V)
        # Then each step's tokens: a (B, 1) column.
        columns = []
        for step in range(max_tokens):
            last = logits[:, -1, :].float()  # (B, vocab_size), float32 as in generate
            if sampler is not None:
                ids = sampler(last, rng=rng)
            else:
                probs = torch.softmax(last, dim=-1)
                ids = torch.multinomial(probs, num_samples=1, generator=rng)
            columns.append(ids)
            if step + 1 < max_tokens:
                logits, _ = m.forward(ids, kv_cache=kv_cache)
        return torch.cat(columns, dim=1).tolist()
