import torch

from diff_gpt.sampler.nucleus import NucleusSampler
from diff_gpt.sampler.temperature import TemperatureSampler


def test_output_shape():
    r"""
    Verifies the output tensor shape.
    L \in R^{B \times V} => S(L) \in Z^{B \times 1}
    where B is batch size, V is vocab size, S is the sampler.
    """
    B, V = 4, 50
    logits = torch.randn(B, V)
    inner_sampler = TemperatureSampler(temperature=1.0)
    sampler = NucleusSampler(p=0.9, sampler=inner_sampler)

    result = sampler(logits)

    # This test expects the consistent (B, 1) shape.
    # It will fail if the original code's .squeeze(-1) is present.
    assert result.shape == (B, 1), f"Expected shape ({B}, 1), but got {result.shape}"


def test_determinism_with_rng():
    """
    Verifies that the composite sampler is deterministic given a seeded generator.
    Let L be logits, G_s be a generator with seed s.
    We verify:
    1. S(L, G_{s1}) == S(L, G_{s1})
    2. S(L, G_{s1}) != S(L, G_{s2}) for s1 != s2
    """
    B, V = 2, 100
    logits = torch.randn(B, V)
    # Use a random inner sampler to ensure the whole chain is random
    inner_sampler = TemperatureSampler(temperature=1.5)
    sampler = NucleusSampler(p=0.9, sampler=inner_sampler)

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


def test_nucleus_selection_logic():
    r"""
    Verifies that only tokens from the nucleus V^{(p)} are ever sampled.
    Let P = softmax(L). Let V^{(p)} be the smallest set where sum_{i in V^{(p)}} P_i >= p.
    We verify: For any sample s, s \in V^{(p)}.
    """
    # Use direct probabilities for clarity, then convert to logits.
    # P = [0.6, 0.25, 0.1, 0.05], Indices = [0, 1, 2, 3]
    probs = torch.tensor([[0.6, 0.25, 0.1, 0.05]])
    logits = torch.log(probs)

    # Set p=0.9.
    # cumsum(P) = [0.6, 0.85, 0.95, 1.0]
    # The nucleus V^{(p)} must contain tokens until cumsum >= 0.9.
    # This includes tokens with probs 0.6, 0.25, and 0.1.
    # V^{(0.9)} = {0, 1, 2}
    nucleus_indices = {0, 1, 2}

    inner_sampler = TemperatureSampler(temperature=1.0)
    sampler = NucleusSampler(p=0.9, sampler=inner_sampler)

    N = 2000  # Number of samples for statistical verification
    logits_batch = logits.expand(N, -1)

    samples = sampler(logits_batch, rng=torch.Generator().manual_seed(123))

    # Find all unique tokens that were sampled
    unique_sampled_indices = set(torch.unique(samples).tolist())

    # The set of sampled tokens must be a subset of the calculated nucleus.
    assert unique_sampled_indices.issubset(nucleus_indices)
    # Also assert that it's not empty, i.e., sampling did happen.
    assert len(unique_sampled_indices) > 0


def test_edge_case_p_equals_one():
    r"""
    Verifies that for p=1, the sampler is equivalent to its inner sampler.
    p=1 => V^{(p)} = V (the entire vocabulary).
    Therefore, S_{nucleus}(L, p=1) \equiv S_{inner}(L).
    """
    B, V = 10, 20
    logits = torch.randn(B, V)
    rng_seed = 1337

    inner_sampler = TemperatureSampler(temperature=1.2)
    nucleus_sampler = NucleusSampler(p=1.0, sampler=inner_sampler)

    # Run inner sampler
    rng1 = torch.Generator().manual_seed(rng_seed)
    result_inner = inner_sampler(logits, rng=rng1)

    # Run nucleus sampler
    rng2 = torch.Generator().manual_seed(rng_seed)
    result_nucleus = nucleus_sampler(logits, rng=rng2)

    assert torch.equal(result_inner, result_nucleus)


def test_edge_case_p_very_small_is_argmax():
    """
    Verifies that for a very small p, the result is always the most likely token.
    p -> 0 => V^{(p)} = {argmax_i P_i}.
    Therefore, S_{nucleus}(L, p->0) should be equivalent to argmax(L),
    regardless of the inner sampler's randomness.
    """
    B, V = 4, 30
    logits = torch.randn(B, V)

    # Use a highly random inner sampler. The nucleus logic must override it.
    random_inner_sampler = TemperatureSampler(temperature=2.0)
    # A p small enough to only ever include the top token.
    sampler = NucleusSampler(p=1e-9, sampler=random_inner_sampler)

    sampled_tokens = sampler(logits)
    expected_tokens = torch.argmax(logits, dim=-1, keepdim=True)

    assert torch.equal(sampled_tokens, expected_tokens)
