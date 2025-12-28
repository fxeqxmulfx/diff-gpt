import pytest
import torch

from diff_gpt.model.gpt import GPT
from diff_gpt.model.engine import Engine
from diff_gpt.sampler.temperature import TemperatureSampler


@pytest.fixture(scope="module")
def model_config() -> dict[str, int]:
    """Provides a shared, small model configuration for tests."""
    return {
        "vocab_size": 50,
        "n_embd": 32,
        "block_size": 16,
        "n_head": 4,
        "n_layer": 2,
    }


@pytest.fixture(scope="module")
def gpt_model(model_config: dict) -> GPT:
    """Provides a shared, initialized GPT model instance."""
    model = GPT(**model_config)
    model.eval()
    return model


def test_engine_initialization(gpt_model: GPT) -> None:
    """Tests basic engine initialization."""
    engine = Engine(model=gpt_model)
    assert engine.model is gpt_model


def test_generate_batch_output_shape(gpt_model: GPT) -> None:
    """
    Checks if the output shape of generate_batch is correct.
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

    # B = num_samples
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


def test_generate_determinism(gpt_model):
    """
    Verifies that with the same seed, the generation is deterministic.
    Проверяет, что при одинаковом seed генерация детерминирована.
    Let G(x_0, s) be the generation function with prompt x_0 and seed s.
    We verify:
    G(x_0, s_1) = G(x_0, s_1)
    G(x_0, s_1) != G(x_0, s_2) for s_1 != s_2
    """
    engine = Engine(model=gpt_model)
    prompt = [5, 8, 12]

    # Use a sampler with high temperature to increase randomness and make the test more robust.
    # This ensures that different seeds will very likely produce different outcomes.
    # Let S_T(l) be the sampling function with temperature T.
    # P(x_i) \propto exp(l_i / T)
    high_temp_sampler = TemperatureSampler(temperature=20)

    # G(x_0, s_1)
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

    # G(x_0, s_2)
    results3, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=2,
        max_tokens=10,
        seed=99,
        sampler=high_temp_sampler,
    )

    # G(x_0, s_1) = G(x_0, s_1)
    assert results1 == results2, "Generation with the same seed should be identical"
    # G(x_0, s_1) != G(x_0, s_2)
    assert results1 != results3, "Generation with different seeds should be different"


def test_streaming_vs_batch_equivalence(gpt_model):
    """
    Ensures that consuming the streaming generator `generate` yields
    the same result as the `generate_batch` utility function.
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
    return torch.argmax(logits, dim=-1, keepdim=True)


def test_prefill_clone_equivalence(gpt_model):
    """
    Crucial test: Verifies that the "prefill once, then clone KV cache"
    optimization yields the same result as running N independent generations.
    We use a deterministic sampler (argmax) to isolate the test to the KV cache logic.
    Ключевой тест: проверяет, что оптимизация "prefill один раз, затем клонировать KV кэш"
    даёт тот же результат, что и N независимых генераций.
    Мы используем детерминированный семплер (argmax), чтобы изолировать тест на логике KV кэша.
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
    # Детерминированный семплер: всегда выбирает токен с наибольшим логитом.
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
    # f(x_0, B=1, sampler=S_greedy)
    single_result, _ = engine.generate_batch(
        tokens=prompt,
        num_samples=1,
        max_tokens=max_tokens,
        sampler=greedy_sampler,
        seed=seed,
    )

    # [f(x_0, B=1, sampler=S_greedy)] * N
    results_naive = single_result * num_samples

    # With a greedy sampler, all samples in the batch must be identical.
    # С жадным семплером все семплы в батче должны быть идентичны.
    for i in range(1, num_samples):
        assert results_engine[0] == results_engine[i]

    # The result of the optimized batch must be the same as N naive runs.
    # Результат оптимизированного батча должен совпадать с N наивными запусками.
    assert results_engine == results_naive
