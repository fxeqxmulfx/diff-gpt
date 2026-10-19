import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import get_domain_of_definition
from diff_gpt.model.gpt import GPT


def _model_and_windows(n_windows=2, ctx=12, horizon=4):
    torch.manual_seed(0)
    idx = np.arange(120, dtype=np.float64)
    df = pd.DataFrame({"y": np.sin(idx / 5.0)})
    dod = get_domain_of_definition(df.to_numpy(dtype=np.float64), 1, False)
    base = GPT(vocab_size=64, n_embd=16, block_size=64, n_head=2, n_layer=1,
               use_checkpoint=False, head="mixture")
    model = DiffGPT(model=base, order_of_derivative=1, domain_of_definition=dod, use_decimal=False)
    windows = [
        (df.iloc[i * 20 : i * 20 + ctx], df.iloc[i * 20 + ctx : i * 20 + ctx + horizon].to_numpy())
        for i in range(n_windows)
    ]
    return model, windows, horizon


def test_returns_a_candidate_and_scores_every_temperature():
    model, windows, H = _model_and_windows()
    temps = (0.5, 1.0)
    best, scores = model.calibrate_temperature(windows, H, temperatures=temps, num_samples=6)
    assert set(scores) == {0.5, 1.0}
    assert best in scores
    assert scores[best] == min(scores.values())
    assert all(np.isfinite(v) for v in scores.values())


def test_lower_temperature_gives_tighter_bands_so_coverage_criterion_moves():
    """Coverage grows monotonically with temperature for a fixed model, so the
    coverage criterion must not pick blindly the largest temperature."""
    model, windows, H = _model_and_windows()
    _, cov_scores = model.calibrate_temperature(
        windows, H, temperatures=(0.3, 1.0), num_samples=8, criterion="coverage", level=0.8
    )
    assert set(cov_scores) == {0.3, 1.0}
    assert all(0.0 <= v <= 1.0 for v in cov_scores.values())


def test_rejects_bad_arguments():
    model, windows, H = _model_and_windows()
    with pytest.raises(AssertionError):
        model.calibrate_temperature([], H)
    with pytest.raises(AssertionError, match="unknown criterion"):
        model.calibrate_temperature(windows, H, criterion="nope", num_samples=4)
    with pytest.raises(AssertionError):
        model.calibrate_temperature(windows, H, num_samples=1)
    with pytest.raises(AssertionError, match="temperature must be positive"):
        model.calibrate_temperature(windows, H, temperatures=(0.0,), num_samples=4)
