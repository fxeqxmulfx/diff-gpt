# Copyright 2025 Mikhail Borisov
"""Tests for the temperature sampler."""

import torch
from torch.testing import assert_close

from diff_gpt.sampler.temperature import TemperatureSampler


def test_output_shape() -> None:
    r"""
    Check that the output has shape (B, 1).

    L \in R^{B \times V} => S(L) \in Z^{B \times 1}
    where B is batch size, V is vocab size, S is the sampler.
    """
    B, V = 4, 50  # B = batch_size, V = vocab_size
    logits = torch.randn(B, V)
    sampler = TemperatureSampler(temperature=1.0)

    result = sampler(logits)

    assert result.shape == (B, 1), f"Expected shape ({B}, 1), but got {result.shape}"


def test_determinism_with_rng() -> None:
    """
    Check that equal seeds give equal samples and different seeds do not.

    The sampler is deterministic given a seeded generator.
    Let L be logits, G_s be a generator with seed s.
    We verify:
    1. S(L, G_{s1}) == S(L, G_{s1})
    2. S(L, G_{s1}) != S(L, G_{s2}) for s1 != s2
    """
    B, V = 2, 100
    logits = torch.randn(B, V)
    sampler = TemperatureSampler(temperature=1.5)

    # 1. Check for equality with the same seed
    rng1 = torch.Generator().manual_seed(42)
    result1 = sampler(logits, rng=rng1)

    rng2 = torch.Generator().manual_seed(42)
    result2 = sampler(logits, rng=rng2)

    assert torch.equal(result1, result2)

    # 2. Check for inequality with different seeds
    rng3 = torch.Generator().manual_seed(99)
    result3 = sampler(logits, rng=rng3)

    assert not torch.equal(result1, result3)


def test_temperature_zero_is_argmax() -> None:
    """
    Check the edge case T=0: the sampled tokens equal argmax(L).

    lim_{T->0} P(x_i) = 1 if l_i = max(L), else 0.
    This is equivalent to argmax(L).
    """
    B, V = 8, 128
    logits = torch.randn(B, V)
    sampler = TemperatureSampler(temperature=0)

    expected_tokens = torch.argmax(logits, dim=-1, keepdim=True)
    sampled_tokens = sampler(logits)

    assert torch.equal(sampled_tokens, expected_tokens)


def test_high_temperature_approaches_uniform() -> None:
    """
    Check the edge case T -> infinity: the samples are close to uniform.

    lim_{T->inf} P(x_i) = 1/|V| (uniform distribution)
    We check this statistically over N samples.
    """
    B, V = 1, 10
    logits = torch.randn(B, V)  # The specific logits shouldn't matter

    # Use a very high temperature to simulate infinity
    sampler = TemperatureSampler(temperature=1000.0)

    N = 20_000  # Number of samples for statistical test
    # Replicate logits N times to sample in one batch
    logits_batch = logits.expand(N, -1)

    samples = sampler(logits_batch, rng=torch.Generator().manual_seed(123))

    # Count occurrences of each token
    counts = torch.bincount(samples.flatten(), minlength=V)
    empirical_probs = counts.float() / N

    # Theoretical probability for a uniform distribution
    expected_probs = torch.full((V,), 1.0 / V)

    # Check if empirical distribution is close to uniform
    assert_close(empirical_probs, expected_probs, atol=0.01, rtol=0.1)


def test_statistical_correctness() -> None:
    r"""
    Check that the empirical distribution matches the theoretical one.

    Let C(i) be the count of token i after N samples.
    We verify: C(i)/N \approx softmax(L/T)_i
    """
    _, V = 1, 5
    T = 0.8
    # Use simple, non-random logits for a reproducible theoretical distribution
    logits = torch.tensor([[0.0, -1.0, 2.0, 1.5, 0.5]], dtype=torch.float32)

    sampler = TemperatureSampler(temperature=T)

    # Theoretical probabilities: P = softmax(L/T).
    expected_probs = torch.softmax(logits / T, dim=-1).flatten()

    N = 50_000  # Number of samples for statistical test
    logits_batch = logits.expand(N, -1)

    samples = sampler(logits_batch, rng=torch.Generator().manual_seed(314))

    # Empirical probabilities
    counts = torch.bincount(samples.flatten(), minlength=V)
    empirical_probs = counts.float() / N

    # Check if the empirical distribution is close to the theoretical one
    assert_close(empirical_probs, expected_probs, atol=0.01, rtol=0.1)


def test_samples_logits_with_several_leading_dims() -> None:
    """
    Check that logits of shape [..., n_tokens] give one sample per row.

    torch.multinomial takes 1-D or 2-D input only, so (2, 3, V) logits
    raised although the Sampler contract allows any leading dims.
    """
    logits = torch.randn(2, 3, 50)
    rng = torch.Generator().manual_seed(0)
    ids = TemperatureSampler(temperature=1.0)(logits, rng=rng)
    assert ids.shape == (2, 3, 1)
    rng = torch.Generator().manual_seed(0)
    flat = TemperatureSampler(temperature=1.0)(logits.reshape(6, 50), rng=rng)
    assert torch.equal(ids.reshape(6, 1), flat)
