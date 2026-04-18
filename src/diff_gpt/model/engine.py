from collections import deque
from typing import Generator

import torch

from diff_gpt.model.kv_cache import KVCache
from diff_gpt.model.gpt import BaseGPT
from diff_gpt.sampler.sampler import Sampler


class RowState:
    __slots__ = (
        "current_tokens",
        "forced_tokens",
    )

    # Per-row state tracking during generation
    def __init__(self, current_tokens: list[int] | None = None) -> None:
        self.current_tokens = current_tokens or list()
        self.forced_tokens = deque()  # Queue of tokens to force inject


class Engine:
    __slots__ = "model"

    def __init__(self, model: BaseGPT) -> None:
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
    ) -> Generator[tuple[list[int], list[int]], None, None]:
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
        """
        assert isinstance(tokens, list), "tokens must be a list of ints"
        assert len(tokens) > 0, "tokens must be non-empty"
        assert all(isinstance(t, int) for t in tokens), "tokens must be a list of ints"

        # Compute effective token budget (RoPE ceiling is block_size * 2).
        block_size = self.model.block_size
        prompt_len = len(tokens)
        hard_ceiling = max(0, block_size * 2 - prompt_len)
        if max_tokens is not None:
            effective_max_tokens = min(max_tokens, hard_ceiling)
        else:
            effective_max_tokens = hard_ceiling
        if effective_max_tokens == 0:
            return  # nothing to generate; skip expensive prefill+alloc
        if force_schedule is not None:
            assert len(force_schedule) >= effective_max_tokens, (
                f"force_schedule must cover all generation steps "
                f"(len={len(force_schedule)} < {effective_max_tokens})"
            )

        device = self.model.device_type
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
                        probs, num_samples=1, generator=rng
                    )  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()
            else:
                # Position-forced: skip the sampler entirely, reuse across samples.
                sampled_tokens = [scheduled_token] * num_samples

            # Process each row: choose the next token, update state, optional tool use
            token_column = []  # contains the next token id along each row
            token_masks = []  # contains the mask (was it sampled (1) or forced (0)?) along each row
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
                1
            )
            logits, _ = self.model.forward(ids, kv_cache=kv_cache_decode)
            logits = logits[:, -1, :]  # (B, vocab_size)

    def generate_batch(
        self, tokens: list[int], num_samples: int = 1, **kwargs
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns (results, masks): each a list of `num_samples` lists of ints.
        """
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                results[i].append(token)
                masks[i].append(mask)
        return results, masks
