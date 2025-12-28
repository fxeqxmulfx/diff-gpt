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
        "completed",
    )

    # Per-row state tracking during generation
    def __init__(self, current_tokens: list[int] | None = None) -> None:
        self.current_tokens = (
            current_tokens or list()
        )  # Current token sequence for this row
        self.forced_tokens = deque()  # Queue of tokens to force inject
        self.completed = False  # Whether this row has completed generation


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
    ) -> Generator[tuple[list[int], list[int]], None, None]:
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), (
            "expecting tuple of ints"
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
        block_size = self.model.block_size
        kv_length_hint = (
            (len(tokens) + max_tokens) if max_tokens is not None else block_size
        )
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
        while True:
            # Stop condition: we've reached max tokens
            if (
                max_tokens is not None
                and num_generated >= max_tokens
                or num_generated >= block_size
            ):
                for state in row_states:
                    state.completed = True
                break

            # Sample the next token for each row
            if sampler is not None:
                next_ids = sampler(logits, rng=rng)  # (B, 1)
            else:
                probs = torch.softmax(logits, dim=-1)  # (B, C)
                next_ids = torch.multinomial(
                    probs, num_samples=1, generator=rng
                )  # (B, 1)
            sampled_tokens = tuple(next_ids[:, 0].tolist())

            # Process each row: choose the next token, update state, optional tool use
            token_column = []  # contains the next token id along each row
            token_masks = []  # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = (
                    len(state.forced_tokens) > 0
                )  # are there tokens waiting to be forced in deque?
                token_masks.append(
                    0 if is_forced else 1
                )  # mask is 0 if forced, 1 if sampled
                next_token = (
                    state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
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
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    results[i].append(token)
                    masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks
