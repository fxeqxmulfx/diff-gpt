"""Tests for the inference Engine: batch, streaming and forced generation."""

import copy
from contextlib import nullcontext

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from diff_gpt.model.engine import Engine
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.temperature import TemperatureSampler


@pytest.fixture(scope="module")
def model_config() -> dict[str, int]:
    """Provide a shared, small model configuration for tests."""
    return {
        "vocab_size": 64,
        "n_embd": 32,
        "block_size": 16,
        "n_head": 4,
        "n_layer": 2,
    }


@pytest.fixture(scope="module")
def gpt_model(model_config: dict) -> GPT:
    """Provide a shared, initialized GPT model instance."""
    model = GPT(**model_config)
    model.eval()
    return model


def test_engine_initialization(gpt_model: GPT) -> None:
    """Tests basic engine initialization."""
    engine = Engine(model=gpt_model)
    assert engine.model is gpt_model


def test_generate_batch_output_shape(gpt_model: GPT) -> None:
    """
    Check that the output shape of generate_batch is correct.

    Let R be the result tokens, M be the masks.
    Let B be num_samples, T_prompt be prompt length, T_max be max_tokens.
    We verify:
    |R| = B
    |R_i| = T_prompt + T_max, for i in {1, ..., B}
    M has the same shape as R.
    """
    engine = Engine(model=gpt_model)
    prompt = [1, 2, 3]
    num_samples = 4
    max_tokens = 5

    # T_prompt = |prompt|
    T_prompt = len(prompt)

    # B is num_samples
    B = num_samples

    results, masks = engine.generate_batch(
        tokens=prompt,
        num_samples=num_samples,
        max_tokens=max_tokens,
    )

    # Check |R| = B
    assert len(results) == B
    # Check |M| = B
    assert len(masks) == B

    # Check |R_i| = T_prompt + T_max
    expected_len = T_prompt + max_tokens
    for i in range(B):
        assert len(results[i]) == expected_len
        assert len(masks[i]) == expected_len


def test_generate_determinism(gpt_model: GPT) -> None:
    """
    Verify that with the same seed, the generation is deterministic.

    Let G(x_0, s) be the generation function with prompt x_0 and seed s.
    We verify:
    G(x_0, s_1) = G(x_0, s_1)
    G(x_0, s_1) != G(x_0, s_2) for s_1 != s_2
    """
    engine = Engine(model=gpt_model)
    prompt = [5, 8, 12]

    # Use a sampler with high temperature to increase randomness and make the
    # test more robust.
    # This ensures that different seeds will very likely produce different outcomes.
    # Let S_T(l) be the sampling function with temperature T.
    # P(x_i) \propto exp(l_i / T)
    high_temp_sampler = TemperatureSampler(temperature=20)

    # G(x_0, s_1) once
    results1, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=2,
        max_tokens=10,
        seed=42,
        sampler=high_temp_sampler,
    )

    # G(x_0, s_1) again
    results2, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=2,
        max_tokens=10,
        seed=42,
        sampler=high_temp_sampler,
    )

    # G(x_0, s_2) with a different seed
    results3, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=2,
        max_tokens=10,
        seed=99,
        sampler=high_temp_sampler,
    )

    # G(x_0, s_1) = G(x_0, s_1)
    assert results1 == results2, "Generation with the same seed should be identical"
    # G(x_0, s_1) != G(x_0, s_2) for s_1 != s_2
    assert results1 != results3, "Generation with different seeds should be different"


def test_streaming_vs_batch_equivalence(gpt_model: GPT) -> None:
    """
    Check that consuming the streaming `generate` matches the `generate_batch` utility.

    Let G_batch(x_0) be the batch generation function.
    Let G_stream(x_0) be the streaming generator.
    We verify:
    G_batch(x_0) == Concatenate(y for y in G_stream(x_0))
    """
    engine = Engine(model=gpt_model)
    prompt = [10, 20]
    num_samples = 3
    max_tokens = 4
    seed = 123

    # --- 1. Batch mode ---
    results_batch, masks_batch = engine.generate_batch(
        tokens=prompt,
        num_samples=num_samples,
        max_tokens=max_tokens,
        seed=seed,
    )

    # --- 2. Streaming mode ---
    # Initialize with the prompt
    results_stream = [prompt.copy() for _ in range(num_samples)]
    masks_stream = [[0] * len(prompt) for _ in range(num_samples)]

    # Consume the generator
    gen = engine.generate(
        tokens=prompt,
        num_samples=num_samples,
        max_tokens=max_tokens,
        seed=seed,
    )

    for token_column, mask_column in gen:
        for i in range(num_samples):
            results_stream[i].append(token_column[i])
            masks_stream[i].append(mask_column[i])

    assert results_batch == results_stream
    assert masks_batch == masks_stream


def greedy_sampler(logits: torch.Tensor, rng: torch.Generator | None) -> torch.Tensor:
    """Return the highest-logit token of each row (deterministic argmax)."""
    del rng  # Part of the sampler interface; argmax needs no randomness.
    return torch.argmax(logits, dim=-1, keepdim=True)


def test_generate_rejects_empty_prompt(gpt_model: GPT) -> None:
    """Empty prompts must raise a clear ValueError instead of IndexError."""
    engine = Engine(model=gpt_model)
    with pytest.raises(ValueError, match="non-empty"):
        _ = engine.generate_batch(tokens=[], num_samples=1, max_tokens=2)


def test_generate_rejects_mixed_type_tokens(gpt_model: GPT) -> None:
    """All elements must be ints — a rogue float/str in the middle must fail."""
    engine = Engine(model=gpt_model)
    with pytest.raises(TypeError, match="list of ints"):
        _ = engine.generate_batch(
            tokens=[1, 2, 3.0],  # type: ignore[list-item]
            num_samples=1,
            max_tokens=2,
        )


def test_generate_early_return_when_prompt_at_position_ceiling(gpt_model: GPT) -> None:
    """
    Check that a prompt at the position ceiling makes generation short-circuit.

    If the prompt already fills all block_size positions, max_tokens=None
    leaves 0 tokens to generate, so the generator must not yield or allocate
    the decode cache.
    """
    engine = Engine(model=gpt_model)
    ceiling = gpt_model.block_size
    prompt = list(range(ceiling))  # exactly at the limit
    num_samples = 2
    results, masks = engine.generate_batch(
        tokens=prompt,
        num_samples=num_samples,
        max_tokens=None,
    )
    assert len(results) == num_samples
    # No generation happened — each row is just the prompt.
    for row, mask in zip(results, masks, strict=True):
        assert row == prompt
        assert mask == [0] * len(prompt)


def test_generate_respects_small_max_tokens(gpt_model: GPT) -> None:
    """max_tokens below the hard ceiling must be honored exactly."""
    engine = Engine(model=gpt_model)
    prompt = [1, 2, 3]
    max_tokens = 4
    results, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=1,
        max_tokens=max_tokens,
    )
    assert len(results[0]) == len(prompt) + max_tokens


def test_generate_rejects_max_tokens_past_position_ceiling(gpt_model: GPT) -> None:
    """
    Check that a max_tokens past the position ceiling raises ValueError.

    The model covers block_size positions, so total context
    (prompt + generated) cannot exceed that. The engine used to cap
    max_tokens silently, returning fewer tokens than asked for.
    """
    engine = Engine(model=gpt_model)
    prompt = [1, 2, 3]
    hard_ceiling = gpt_model.block_size - len(prompt)
    with pytest.raises(ValueError, match="exceeds"):
        _ = engine.generate_batch(
            tokens=prompt,
            num_samples=1,
            max_tokens=hard_ceiling + 1,
        )
    results, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=1,
        max_tokens=hard_ceiling,
    )
    assert len(results[0]) == gpt_model.block_size


def test_generate_rejects_prompt_longer_than_block_size(gpt_model: GPT) -> None:
    """A prompt past block_size raises instead of coming back unchanged."""
    engine = Engine(model=gpt_model)
    prompt = list(range(gpt_model.block_size + 1))
    with pytest.raises(ValueError, match="block_size"):
        _ = engine.generate_batch(tokens=prompt, num_samples=1, max_tokens=None)


def test_generate_rejects_negative_max_tokens(gpt_model: GPT) -> None:
    """A negative max_tokens raises ValueError."""
    engine = Engine(model=gpt_model)
    with pytest.raises(ValueError, match=">= 0"):
        _ = engine.generate_batch(tokens=[1, 2], num_samples=1, max_tokens=-1)


def test_generate_no_max_tokens_uses_hard_ceiling(gpt_model: GPT) -> None:
    """Without max_tokens, generation fills up to the position ceiling."""
    engine = Engine(model=gpt_model)
    block_size = gpt_model.block_size
    prompt = [1, 2, 3, 4]
    hard_ceiling = block_size - len(prompt)
    results, _ = engine.generate_batch(tokens=prompt, num_samples=1)
    assert len(results[0]) == len(prompt) + hard_ceiling


def test_force_schedule_interleaves_forced_and_sampled(gpt_model: GPT) -> None:
    """
    Check per-step forcing by force_schedule: `None` samples, an int forces.

    An alternating schedule must produce exactly the forced tokens at the
    forced positions and sampled (non-None, in-range) tokens elsewhere.
    """
    engine = Engine(model=gpt_model)
    prompt = [1, 2, 3]
    # Alternate: force 7, sample, force 11, sample, force 13, sample.
    schedule: list[int | None] = [7, None, 11, None, 13, None]
    results, masks = engine.generate_batch(
        tokens=prompt,
        num_samples=2,
        max_tokens=len(schedule),
        force_schedule=schedule,
    )
    for row, mask in zip(results, masks, strict=True):
        tail = row[len(prompt) :]
        assert tail[0] == schedule[0]
        assert tail[2] == schedule[2]
        assert tail[4] == schedule[4]
        # Forced positions' masks should be 0; sampled ones 1.
        tail_mask = mask[len(prompt) :]
        assert tail_mask == [0, 1, 0, 1, 0, 1]


def test_force_schedule_all_forced_matches_schedule(gpt_model: GPT) -> None:
    """If every slot is forced, both samples reproduce the schedule exactly."""
    engine = Engine(model=gpt_model)
    prompt = [1, 2]
    schedule = [5, 6, 7, 8]
    results, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=3,
        max_tokens=len(schedule),
        force_schedule=schedule,
    )
    for row in results:
        assert row[len(prompt) :] == schedule


def test_force_schedule_too_short_raises(gpt_model: GPT) -> None:
    """Check that a force_schedule shorter than the generation raises ValueError."""
    engine = Engine(model=gpt_model)
    with pytest.raises(ValueError, match="force_schedule must cover"):
        engine.generate_batch(
            tokens=[1, 2, 3],
            num_samples=1,
            max_tokens=10,
            force_schedule=[1, 2, 3],  # only 3 entries
        )


def test_prefill_clone_equivalence(gpt_model: GPT) -> None:
    """
    Verify that "prefill once, then clone KV cache" equals N independent runs.

    Crucial test of this optimization. We use a deterministic sampler (argmax)
    to isolate the test to the KV cache logic.
    Let f(x_0, B) be the generation with num_samples=B.
    Let S_greedy be the argmax sampler.
    We verify:
    f(x_0, B=N, sampler=S_greedy) == [f(x_0, B=1, sampler=S_greedy)] * N
    """
    engine = Engine(model=gpt_model)
    prompt = [1, 8, 4, 9]
    num_samples = 4
    max_tokens = 6
    seed = 42

    # Deterministic sampler: always picks the token with the highest logit.
    # S_greedy(l) = argmax(l)

    # --- 1. Engine's optimized batch generation (Prefill & Clone) ---
    results_engine, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=num_samples,
        max_tokens=max_tokens,
        sampler=greedy_sampler,
        seed=seed,  # seed shouldn't matter for argmax, but good practice
    )

    # --- 2. Naive batch generation (N separate runs) ---
    # One run: f(x_0, B=1, sampler=S_greedy)
    single_result, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=1,
        max_tokens=max_tokens,
        sampler=greedy_sampler,
        seed=seed,
    )

    # Repeated N times: [f(x_0, B=1, sampler=S_greedy)] * N
    results_naive = single_result * num_samples

    # With a greedy sampler, all samples in the batch must be identical.
    for i in range(1, num_samples):
        assert results_engine[0] == results_engine[i]

    # The result of the optimized batch must be the same as N naive runs.
    assert results_engine == results_naive


def test_generate_many_matches_generate_per_prompt(gpt_model: GPT) -> None:
    """
    Check that one batch of prompts continues each as `generate` alone does.

    With the argmax sampler, the new tokens of every row equal those of a
    batch-1 run on its prompt.
    """
    engine = Engine(model=gpt_model)
    prompts = [[1, 8, 4, 9], [2, 2, 7, 30], [63, 0, 5, 11]]
    max_tokens = 6
    generated = engine.generate_many(
        prompts=prompts,
        max_tokens=max_tokens,
        sampler=greedy_sampler,
    )
    for prompt, new in zip(prompts, generated, strict=True):
        alone, _ = engine.generate_batch(
            tokens=prompt,
            num_samples=1,
            max_tokens=max_tokens,
            sampler=greedy_sampler,
        )
        assert new == alone[0][len(prompt) :]


def test_cached_generation_matches_full_forward(
    model_config: dict[str, int],
) -> None:
    """
    Check cached decoding against full forward passes over the growing sequence.

    Compare `generate_batch` (a batch-1 prefill copied into each row's cache)
    and `generate_many` to the same argmax continuations.
    """
    torch.manual_seed(0)
    model = GPT(**model_config)
    model.eval()
    engine = Engine(model=model)
    prompts = [[1, 8, 4, 9], [2, 2, 7, 30]]
    max_tokens = 6
    expected = []
    for prompt in prompts:
        ids = torch.tensor([prompt])
        with torch.no_grad():
            for _ in range(max_tokens):
                logits, _ = model(ids)
                ids = torch.cat((ids, greedy_sampler(logits[:, -1], None)), dim=1)
        expected.append(ids[0, len(prompt) :].tolist())

    generated = engine.generate_many(
        prompts=prompts, max_tokens=max_tokens, sampler=greedy_sampler
    )

    assert generated == expected
    for prompt, new in zip(prompts, expected, strict=True):
        results, _ = engine.generate_batch(
            tokens=prompt,
            num_samples=2,
            max_tokens=max_tokens,
            sampler=greedy_sampler,
        )
        assert [row[len(prompt) :] for row in results] == [new, new]


def test_generate_many_is_seeded(gpt_model: GPT) -> None:
    """Sampling without a sampler repeats under one seed."""
    engine = Engine(model=gpt_model)
    prompts = [[1, 2, 3], [4, 5, 6]]
    max_tokens = 8
    first = engine.generate_many(prompts=prompts, max_tokens=max_tokens, seed=7)
    again = engine.generate_many(prompts=prompts, max_tokens=max_tokens, seed=7)
    assert first == again
    assert all(len(row) == max_tokens for row in first)


def test_generate_many_fills_the_positions_left(gpt_model: GPT) -> None:
    """
    Check that `max_tokens` may fill `block_size` but not run past it.

    0 returns an empty continuation for every prompt.
    """
    engine = Engine(model=gpt_model)
    prompts = [[1, 2, 3], [4, 5, 6]]
    room = gpt_model.block_size - len(prompts[0])
    generated = engine.generate_many(prompts=prompts, max_tokens=room)
    assert [len(row) for row in generated] == [room, room]
    assert engine.generate_many(prompts=prompts, max_tokens=0) == [[], []]
    with pytest.raises(ValueError, match="positions left"):
        engine.generate_many(prompts=prompts, max_tokens=room + 1)
    with pytest.raises(ValueError, match="positions left"):
        engine.generate_many(prompts=prompts, max_tokens=-1)


def test_generate_many_rejects_bad_prompts(gpt_model: GPT) -> None:
    """No prompt, an empty one, two lengths, a non-int token or one too long raise."""
    engine = Engine(model=gpt_model)
    with pytest.raises(ValueError, match="non-empty"):
        engine.generate_many(prompts=[], max_tokens=2)
    with pytest.raises(ValueError, match="non-empty"):
        engine.generate_many(prompts=[[1, 2], []], max_tokens=2)
    with pytest.raises(ValueError, match="one length"):
        engine.generate_many(prompts=[[1, 2], [1, 2, 3]], max_tokens=2)
    with pytest.raises(TypeError, match="lists of ints"):
        engine.generate_many(
            prompts=[[1, 2], [3, 4.0]],  # type: ignore[list-item]
            max_tokens=2,
        )
    too_long = list(range(gpt_model.block_size + 1))
    with pytest.raises(ValueError, match="block_size"):
        engine.generate_many(prompts=[too_long], max_tokens=0)


def test_generate_many_draws_samples_of_each_prompt(gpt_model: GPT) -> None:
    """
    Check that `num_samples` rows continue each prompt, sample s of i at i·S + s.

    Under the argmax sampler every sample is the one continuation of its
    prompt; multinomial samples of a prompt differ and repeat under one seed.
    """
    engine = Engine(model=gpt_model)
    prompts = [[1, 8, 4, 9], [2, 2, 7, 30], [63, 0, 5, 11]]
    max_tokens = 6
    num_samples = 3
    single = engine.generate_many(
        prompts=prompts,
        max_tokens=max_tokens,
        sampler=greedy_sampler,
    )
    samples = engine.generate_many(
        prompts=prompts,
        max_tokens=max_tokens,
        sampler=greedy_sampler,
        num_samples=num_samples,
    )
    assert samples == [row for row in single for _ in range(num_samples)]
    drawn = engine.generate_many(
        prompts=prompts,
        max_tokens=max_tokens,
        seed=7,
        num_samples=num_samples,
    )
    again = engine.generate_many(
        prompts=prompts,
        max_tokens=max_tokens,
        seed=7,
        num_samples=num_samples,
    )
    assert drawn == again
    assert len(drawn) == len(prompts) * num_samples
    assert len({tuple(row) for row in drawn[:num_samples]}) > 1
    empty = engine.generate_many(prompts=prompts, max_tokens=0, num_samples=2)
    assert empty == [[]] * (len(prompts) * 2)
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        engine.generate_many(prompts=prompts, max_tokens=2, num_samples=0)


def test_samplers_get_float32_logits_from_a_half_precision_model(
    gpt_model: GPT,
) -> None:
    """
    Check that the logits of a bfloat16 model reach the sampler as float32.

    A bfloat16 softmax and multinomial would round the probabilities and the
    sampling noise, and so change the distribution the tokens come from.
    """
    dtypes: list[torch.dtype] = []

    def recording_sampler(
        logits: torch.Tensor,
        rng: torch.Generator | None,
    ) -> torch.Tensor:
        dtypes.append(logits.dtype)
        return greedy_sampler(logits, rng)

    engine = Engine(model=copy.deepcopy(gpt_model).to(torch.bfloat16))
    engine.generate_many(
        prompts=[[1, 8, 4, 9]],
        max_tokens=3,
        sampler=recording_sampler,
        num_samples=2,
    )
    engine.generate_batch(
        [1, 8, 4, 9],
        num_samples=2,
        max_tokens=3,
        sampler=recording_sampler,
    )
    assert dtypes == [torch.float32] * 6


def _attention_backends() -> tuple[bool, bool, bool, bool]:
    """Return whether flash, memory-efficient, math and cuDNN attention are enabled."""
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )


@pytest.mark.parametrize("backends", [None, [SDPBackend.MATH]])
def test_decoding_keeps_the_enabled_attention_backends(
    gpt_model: GPT,
    backends: list[SDPBackend] | None,
) -> None:
    """
    Check that the decode steps run attention with the caller's backends.

    They put memory-efficient attention first, but every attention call, the
    prefill's and each decode step's, sees the backends the caller enabled:
    all of them by default, or the math backend alone, as after the calls.
    """
    seen: list[tuple[bool, bool, bool, bool]] = []

    def record(_module: torch.nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
        seen.append(_attention_backends())

    engine = Engine(model=gpt_model)
    handle = gpt_model.blocks[0].sa.register_forward_pre_hook(record)
    try:
        with sdpa_kernel(backends) if backends else nullcontext():
            caller = _attention_backends()
            engine.generate_many(prompts=[[1, 8, 4, 9]], max_tokens=3, num_samples=2)
            engine.generate_batch([1, 8, 4, 9], num_samples=2, max_tokens=3)
            assert _attention_backends() == caller
    finally:
        handle.remove()
    # generate_many: a prefill and 2 decode steps; generate: a prefill and 3.
    assert seen == [caller] * 7
