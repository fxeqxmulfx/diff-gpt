"""Check the continuous generator and its quantized conditional Bayes law."""

from dataclasses import replace

import numpy as np
import pytest

from diff_gpt.application.grokking import series_gaussian as gaussian
from diff_gpt.application.grokking.series_data import DataConfig, dataset
from diff_gpt.domain.encoder_decoder import encode


@pytest.mark.parametrize("tokenizer", ["lgt", "mash"])
def test_gaussian_bin_edges_match_actual_codec_boundaries(tokenizer: str) -> None:
    """The oracle changes output cells at the encoder's actual discontinuities."""
    vocab, mean, std = 64, 2.0, 1.5
    domain = np.array([3.0])
    _, grid, _ = encode(np.zeros((1, 1)), vocab, domain, 0, tokenizer=tokenizer)
    edges = gaussian.bin_edges(mean, std, float(grid.scale[0]), vocab)
    epsilon = std / grid.scale[0] * 1e-6
    probes = edges[1:-1, None] + np.array([-epsilon, epsilon])[None, :]
    _, _, tokens = encode(
        ((probes.ravel() - mean) / std)[:, None],
        vocab,
        domain,
        0,
        tokenizer=tokenizer,
    )
    expected = np.column_stack((np.arange(vocab - 1), np.arange(1, vocab)))
    np.testing.assert_array_equal(tokens.reshape(-1, 2), expected)


@pytest.mark.parametrize("lag", [1, 24])
def test_gaussian_oracle_matches_generated_transition_frequencies(lag: int) -> None:
    """Both generators agree with the independently integrated conditional law."""
    values = gaussian.generate(100000, 37, lag)[:, 0]
    edges = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    counts, _, _ = np.histogram2d(values[:-lag], values[lag:], bins=(edges, edges))
    observed = counts / counts.sum(1, keepdims=True)
    expected, _ = gaussian.transition(tuple(edges))
    np.testing.assert_allclose(observed, expected, atol=0.02, rtol=0)


def test_independent_gaussian_transition_is_the_marginal() -> None:
    """With zero AR coefficient, every input bin has the same output law."""
    edges = (-np.inf, -1.0, 0.0, 1.0, np.inf)
    probabilities, diagnostics = gaussian.transition(edges, coefficient=0.0)
    expected = np.diff(gaussian.normal_cdf(np.asarray(edges)))
    np.testing.assert_allclose(probabilities, np.tile(expected, (4, 1)), atol=1e-12)
    assert diagnostics["max_probability_delta"] < gaussian.PROBABILITY_TOLERANCE


def test_gaussian_transition_preserves_stationarity_and_detailed_balance() -> None:
    """The integrated law agrees with two identities of a stationary normal AR."""
    edges = np.array([-np.inf, -2.0, -0.5, 0.0, 0.5, 2.0, np.inf])
    probabilities, _ = gaussian.transition(tuple(edges))
    marginal = np.diff(
        gaussian.normal_cdf(edges * np.sqrt(1 - gaussian.COEFFICIENT**2))
    )
    joint = marginal[:, None] * probabilities
    np.testing.assert_allclose(joint, joint.T, atol=1e-7)
    np.testing.assert_allclose(marginal @ probabilities, marginal, atol=1e-7)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-12)
    assert np.all(probabilities >= 0)


@pytest.mark.parametrize(
    ("name", "lag"), [("gaussian_ar", 1), ("gaussian_seasonal_ar", 24)]
)
def test_gaussian_dataset_uses_relevant_phase_and_independent_calibration(
    name: str, lag: int
) -> None:
    """The oracle conditions on the lagged input and calibration never enters test."""
    config = DataConfig(dataset=name, context=lag, max_train=256, max_eval=64)
    data = dataset(config)
    fit = data.metadata["fit"][0]
    assert data.metadata["independent_split_realizations"]
    assert fit["fit_population"] == "independent training-only calibration realization"
    more = dataset(replace(config, max_train=512))
    assert fit == more.metadata["fit"][0]
    # Sampling is fixed before coverage filtering; a larger training set can
    # make additional heldout windows eligible without changing their values.
    common = set(map(tuple, data.splits["test"].ids)) & set(
        map(tuple, more.splits["test"].ids)
    )
    assert len(common) > config.max_eval // 2
    split = data.splits["test"]
    # Equal relevant-phase tokens have equal predictive laws even if all the
    # intervening phases differ. Distinct tokens change that law.
    by_token = {}
    for token, law in zip(split.x[:, 0], split.bayes[:, 0], strict=True):
        if token in by_token:
            np.testing.assert_array_equal(law, by_token[token])
        by_token[token] = law
    assert len(by_token) > 1
    assert np.max(np.ptp(np.stack(list(by_token.values())), axis=0)) > 0
    assert data.references["test"]["x"]["floor_loss"] > 0
    assert data.metadata["bayes_conditioned_on_target_coverage"]
    seen = np.unique(
        np.concatenate((data.splits["train"].x.ravel(), data.splits["train"].y[:, -1]))
    )
    missing = np.setdiff1d(np.arange(config.vocab_size), seen)
    assert np.all(split.bayes[..., missing] == 0)
    np.testing.assert_allclose(split.bayes.sum(-1), 1, atol=1e-12)
    with pytest.raises(ValueError, match="context = lag"):
        dataset(replace(config, context=lag + 1))
    with pytest.raises(ValueError, match="shuffled"):
        dataset(replace(config, shuffled_time=True))


def test_gaussian_generator_is_deterministic_and_seasonal() -> None:
    """A seasonal process has correlation at its specified phase rather than lag one."""
    values = gaussian.generate(30000, 11, 24)[:, 0]
    np.testing.assert_array_equal(values, gaussian.generate(30000, 11, 24)[:, 0])
    assert np.corrcoef(values[24:], values[:-24])[0, 1] == pytest.approx(
        0.75, abs=0.025
    )
    assert np.corrcoef(values[1:], values[:-1])[0, 1] == pytest.approx(0, abs=0.06)
