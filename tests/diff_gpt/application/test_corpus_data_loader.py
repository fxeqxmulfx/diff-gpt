"""The corpus loader: windows of many series, each in the units of its context."""

import numpy as np
import pytest
import torch

from diff_gpt.application.data_loader import DiffCorpusDataLoader, corpus_domain
from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.domain.encoder_decoder import observed_tokens
from diff_gpt.domain.model.gpt import GPT

IGNORE_INDEX = -100
BLOCK_SIZE = 16
VOCAB_SIZE = 64
# A window of BLOCK_SIZE + 1 tokens of order 1 reads one row more.
ROWS = BLOCK_SIZE + 2
# The windows a batch of the weights test draws, and the share and the
# tolerance of the source of weight 3 among weights 1 and 3.
WEIGHTED_BATCH = 4000
WEIGHTED_SHARE = 0.75
SHARE_TOLERANCE = 0.05
# Under missing_prob 1, more than this share of windows masks a target.
MASKED_WINDOWS_SHARE = 0.9


def _random_walks(n: int, length: int, seed: int) -> np.ndarray:
    """Return `n` random walks of `length` values, float64 of shape (n, length)."""
    return np.cumsum(np.random.default_rng(seed).normal(size=(n, length)), axis=1)


def _loader(
    train: dict[str, np.ndarray],
    *,
    batch_size: int = 8,
    seed: int = 0,
    val: dict[str, np.ndarray] | None = None,
    weights: dict[str, float] | None = None,
    missing_prob: float = 0.0,
) -> DiffCorpusDataLoader:
    """Return a loader of order 1 over `train`, on the CPU."""
    return DiffCorpusDataLoader(
        train=train,
        block_size=BLOCK_SIZE,
        batch_size=batch_size,
        vocab_size=VOCAB_SIZE,
        order_of_derivative=1,
        domain_of_definition=np.array([4.0]),
        device="cpu",
        val=val,
        weights=weights,
        missing_prob=missing_prob,
        rng=np.random.default_rng(seed),
    )


def test_batches_are_windows_shifted_by_one_token() -> None:
    """Check the shape, the dtype and the range of a batch, and y = x shifted."""
    loader = _loader({"walks": _random_walks(4, 100, seed=0)})
    x, y = loader.get_batch()
    assert x.shape == (8, BLOCK_SIZE)
    assert y.shape == (8, BLOCK_SIZE)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert x.min() >= 0
    assert x.max() < VOCAB_SIZE
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert not loader.has_val()


def test_one_seed_repeats_the_batches() -> None:
    """Check that a seed draws the same windows, and another seed others."""
    corpus = {"walks": _random_walks(4, 100, seed=0)}
    first = _loader(corpus, seed=5).get_batch()
    again = _loader(corpus, seed=5).get_batch()
    other = _loader(corpus, seed=6).get_batch()
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not torch.equal(first[0], other[0])


def test_series_in_other_units_give_the_same_tokens() -> None:
    """
    Check that the corpus 1024·x tokenizes as x: one vocabulary for every scale.

    A power of 2 scales every statistic exactly, so the tokens are equal.
    """
    walks = _random_walks(4, 100, seed=0)
    small = _loader({"walks": walks}, seed=1).get_batch()
    large = _loader({"walks": 1024.0 * walks}, seed=1).get_batch()
    assert all(torch.equal(a, b) for a, b in zip(small, large, strict=True))


def test_targets_that_read_a_missing_value_are_masked() -> None:
    """
    Check that y is -100 exactly where a token reads a NaN of the series.

    The one series is one window long, so every window is that series; x
    holds a token at every position, of the filled values.
    """
    series = _random_walks(1, ROWS, seed=0)
    series[0, [3, 9, 10]] = np.nan
    x, y = _loader({"gappy": series}).get_batch()
    observed = observed_tokens(series.T, 1)[1:, 0]
    assert np.array_equal((y != IGNORE_INDEX).numpy(), np.tile(observed, (8, 1)))
    assert not observed.all()
    kept = y[:, :-1] != IGNORE_INDEX
    assert torch.equal(x[:, 1:][kept], y[:, :-1][kept])


def test_missing_prob_takes_values_out_of_the_windows() -> None:
    """
    Check that missing_prob 1 masks targets of the windows, and 0 none.

    The targets left still continue the inputs.
    """
    corpus = {"walks": _random_walks(4, 100, seed=0)}
    _, complete = _loader(corpus).get_batch()
    assert (complete != IGNORE_INDEX).all()
    x, y = _loader(corpus, batch_size=64, missing_prob=1.0).get_batch()
    masked = y == IGNORE_INDEX
    assert masked.any(dim=1).double().mean() > MASKED_WINDOWS_SHARE
    kept = ~masked[:, :-1]
    assert torch.equal(x[:, 1:][kept], y[:, :-1][kept])


def test_weights_set_the_share_of_each_source() -> None:
    """
    Check that a source's windows come in proportion to its weight.

    The windows of the source of NaN have every target masked, so they count
    by their rows of -100: none at weight 0, about 3/4 at the weights 1:3.
    """
    corpus = {
        "walks": _random_walks(4, 100, seed=0),
        "gaps": np.full((2, 50), np.nan),
    }
    _, y = _loader(corpus, batch_size=64, weights={"walks": 1, "gaps": 0}).get_batch()
    assert (y != IGNORE_INDEX).all()
    _, y = _loader(
        corpus,
        batch_size=WEIGHTED_BATCH,
        weights={"walks": 1, "gaps": 3},
    ).get_batch()
    share = (y == IGNORE_INDEX).all(dim=1).double().mean().item()
    assert abs(share - WEIGHTED_SHARE) < SHARE_TOLERANCE


def test_val_draws_from_its_own_sources() -> None:
    """Check that val windows come from `val`, and the errors of the splits."""
    corpus = {"walks": _random_walks(4, 100, seed=0)}
    loader = _loader(corpus, val={"walks": np.full((1, ROWS), np.nan)})
    assert loader.has_val()
    _, y = loader.get_batch("val")
    assert (y == IGNORE_INDEX).all()
    _, y = loader.get_batch("train")
    assert (y != IGNORE_INDEX).all()
    with pytest.raises(ValueError, match="unknown split"):
        loader.get_batch("test")
    with pytest.raises(RuntimeError, match="no val split"):
        _loader(corpus).get_batch("val")


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"order_of_derivative": [1]}, TypeError, "univariate"),
        ({"domain_of_definition": np.array([1.0, 2.0])}, ValueError, "shape"),
        ({"normalization": "max"}, ValueError, "unknown normalization"),
        ({"min_context": 0}, ValueError, "min_context"),
        ({"min_context": ROWS + 1}, ValueError, "min_context"),
        ({"missing_prob": 1.5}, ValueError, "missing_prob"),
        ({"max_missing_fraction": 1.0}, ValueError, "max_missing_fraction"),
        ({"max_missing_run": 0}, ValueError, "max_missing_run"),
        ({"train": {}}, ValueError, "at least one source"),
        ({"train": {"short": np.zeros((2, ROWS - 1))}}, ValueError, f"{ROWS} rows"),
        ({"weights": {"other": 1.0}}, ValueError, "no entry"),
        ({"weights": {"walks": -1.0}}, ValueError, "non-negative"),
    ],
)
def test_loader_rejects_bad_settings(
    change: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """Check every setting the loader checks before it draws a window."""
    settings = {
        "train": {"walks": _random_walks(2, 50, seed=0)},
        "block_size": BLOCK_SIZE,
        "batch_size": 4,
        "vocab_size": VOCAB_SIZE,
        "order_of_derivative": 1,
        "domain_of_definition": np.array([4.0]),
        "device": "cpu",
    }
    with pytest.raises(error, match=match):
        DiffCorpusDataLoader(**(settings | change))


def test_corpus_domain_bounds_the_steps_within_windows() -> None:
    """
    Check that the domain holds the steps of the windows and none across two.

    Every step of a ramp is 1 in the units of any of its windows, while the
    step from the end of one window to the start of the next is about -ROWS.
    """
    ramps = np.arange(100.0) + np.array([[0.0], [50.0], [-7.0]])
    domain = corpus_domain({"ramps": ramps}, BLOCK_SIZE, 64, order_of_derivative=1)
    np.testing.assert_allclose(domain, [1.0])


def test_corpus_domain_quantile_and_ranges() -> None:
    """
    Check the shapes of the domains, and that a quantile holds fewer steps.

    The steps of a window have the mean size 1 in its units, so their
    median lies below the largest.
    """
    walks = {"walks": _random_walks(8, 200, seed=0)}
    full = corpus_domain(walks, BLOCK_SIZE, 256, order_of_derivative=1)
    half = corpus_domain(walks, BLOCK_SIZE, 256, order_of_derivative=1, quantile=0.5)
    ranges = corpus_domain(
        walks,
        BLOCK_SIZE,
        256,
        order_of_derivative=1,
        symmetric=False,
    )
    assert full.shape == (1,)
    assert half.shape == (1,)
    assert half[0] < full[0]
    assert ranges.shape == (1, 2)
    assert ranges[0, 0] < 0 < ranges[0, 1]


def test_a_model_trains_on_the_corpus() -> None:
    """Check that `DiffGPT.train` drives the loader, with missing values and val."""
    torch.manual_seed(0)
    walks = {"walks": _random_walks(8, 200, seed=0)}
    domain = corpus_domain(walks, BLOCK_SIZE, 64, order_of_derivative=1, quantile=0.99)
    model = DiffGPT(
        model=GPT(
            vocab_size=VOCAB_SIZE,
            n_embd=16,
            block_size=BLOCK_SIZE,
            n_head=2,
            n_layer=1,
        ),
        order_of_derivative=1,
        domain_of_definition=domain,
    )
    loader = DiffCorpusDataLoader(
        train=walks,
        block_size=BLOCK_SIZE,
        batch_size=4,
        vocab_size=VOCAB_SIZE,
        order_of_derivative=1,
        domain_of_definition=domain,
        device=model.model.device_type,
        val={"walks": _random_walks(2, 200, seed=1)},
        missing_prob=0.5,
        tokenizer=model.tokenizer,
    )
    val_loss, _ = model.train(
        loader=loader,
        max_iters=4,
        eval_interval=4,
        eval_iters=2,
        use_tqdm=False,
    )
    assert np.isfinite(val_loss)
