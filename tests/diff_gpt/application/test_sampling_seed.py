"""
Sampling seed (step 16): the `seed` argument of the Monte-Carlo forecasts.

Repeated calls were identical because the Engine always seeded with 42, which
silently correlated independent Monte-Carlo runs.
"""

import numpy as np
import pandas as pd
import torch

from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.domain.encoder_decoder import get_domain_of_definition
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.sampler.temperature import TemperatureSampler


def _model() -> tuple[DiffGPT, pd.DataFrame]:
    torch.manual_seed(0)
    idx = np.arange(80, dtype=np.float64)
    df = pd.DataFrame({"y": np.sin(idx / 5.0)})
    dod = get_domain_of_definition(df.to_numpy(dtype=np.float64), 1)
    base = GPT(
        vocab_size=64,
        n_embd=16,
        block_size=64,
        n_head=2,
        n_layer=1,
        use_checkpoint=False,
    )
    return DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=dod,
    ), df


def test_same_seed_reproduces_and_different_seed_does_not() -> None:
    """Check that equal seeds give equal samples and different seeds do not."""
    model, df = _model()
    kw = {
        "df": df.iloc[:20],
        "max_new_points": 6,
        "num_samples": 8,
        "sampler": TemperatureSampler(temperature=1.0),
    }
    a = model.predict_samples(**kw, seed=1)
    b = model.predict_samples(**kw, seed=1)
    c = model.predict_samples(**kw, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_default_seed_is_still_reproducible() -> None:
    """Check that two calls with the default seed return identical samples."""
    model, df = _model()
    kw = {
        "df": df.iloc[:20],
        "max_new_points": 5,
        "num_samples": 4,
        "sampler": TemperatureSampler(temperature=1.0),
    }
    assert np.array_equal(model.predict_samples(**kw), model.predict_samples(**kw))


def test_seed_reaches_quantiles_and_point_readouts() -> None:
    """Check that the seed reaches predict_quantiles and predict_point."""
    model, df = _model()
    kw = {
        "df": df.iloc[:20],
        "max_new_points": 5,
        "num_samples": 8,
        "sampler": TemperatureSampler(temperature=1.0),
    }
    q1 = model.predict_quantiles(**kw, quantiles=(0.5,), seed=3)[0.5].to_numpy()
    q2 = model.predict_quantiles(**kw, quantiles=(0.5,), seed=4)[0.5].to_numpy()
    assert not np.array_equal(q1, q2)
    p1 = model.predict_point(**kw, metric="mae", seed=3).to_numpy()
    assert np.allclose(p1, q1)  # the MAE Bayes point is the marginal median
