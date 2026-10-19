"""Exact token-derived features (step 11): running integral and seasonal lags."""
import numpy as np
import pytest
import torch
from torch.testing import assert_close

from diff_gpt.model.engine import Engine
from diff_gpt.model.gpt import GPT
from diff_gpt.model.kv_cache import KVCache


def _gpt(lags, **kw):
    torch.manual_seed(0)
    return GPT(vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=2,
               use_checkpoint=False, feature_lags=lags, **kw)


def test_no_features_by_default():
    m = _gpt(())
    assert m.n_extra_features == 0
    assert not hasattr(m, "feature_proj")


def test_features_match_the_closed_form():
    m = _gpt((4,))
    assert m.n_extra_features == 3
    idx = torch.randint(0, 64, (2, 10))
    f = m._token_features(idx, 0, 10).numpy()
    for b in range(2):
        q = (idx[b].numpy() - 32) / 64.0
        level = np.cumsum(q)
        ref = np.stack([
            level,
            level - np.concatenate([np.zeros(4), level[:-4]]),
            np.concatenate([np.zeros(4), q[:-4]]),
        ], axis=-1)
        assert np.allclose(f[b], ref, atol=1e-6)


def test_lag_longer_than_history_is_zero_padded():
    m = _gpt((9,))
    idx = torch.randint(0, 64, (1, 4))
    f = m._token_features(idx, 0, 4)[0]
    assert torch.allclose(f[:, 1], f[:, 0])  # level - 0
    assert torch.count_nonzero(f[:, 2]) == 0


def test_zero_init_makes_features_a_no_op_at_initialisation():
    """feature_proj starts at zero, so a fresh model matches one without features
    and training can only add the signal if it helps."""
    a, b = _gpt((4,)), _gpt(())
    b.load_state_dict({k: v for k, v in a.state_dict().items() if not k.startswith("feature_proj")})
    idx = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        assert_close(a(idx)[0], b(idx)[0], atol=1e-6, rtol=1e-6)


def test_incremental_kv_decoding_matches_full_forward():
    m = _gpt((3,)).eval()
    idx = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        full, _ = m(idx)
        kv = KVCache(batch_size=1, num_heads=2, seq_len=40, head_dim=8, num_layers=2)
        steps = torch.cat([m(idx[:, t : t + 1], kv_cache=kv)[0] for t in range(12)], dim=1)
    assert_close(steps, full, atol=1e-5, rtol=1e-5)


def test_engine_prefill_expands_the_token_history_across_samples():
    m = _gpt((3,)).eval()
    out, _ = Engine(model=m).generate_batch([5, 9, 13, 21], num_samples=3, max_tokens=6, sampler=None)
    assert len(out) == 3 and all(len(r) == 10 for r in out)
    assert all(0 <= t < 64 for r in out for t in r)


def test_gradient_reaches_the_feature_projection():
    m = _gpt((3,))
    idx = torch.randint(0, 64, (2, 10))
    _, loss = m(idx, torch.randint(0, 64, (2, 10)))
    loss.backward()
    assert m.feature_proj.weight.grad is not None
    assert m.feature_proj.weight.grad.abs().sum() > 0


def test_sliding_window_generate_is_refused_with_features():
    m = _gpt((3,)).eval()
    with pytest.raises(AssertionError, match="Engine.generate"):
        m.generate(torch.randint(0, 64, (1, 4)), max_new_tokens=3, sampler=None)


def test_features_combine_with_channel_embedding():
    m = _gpt((6,), n_channels=3).eval()
    idx = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        full, _ = m(idx)
        kv = KVCache(batch_size=1, num_heads=2, seq_len=40, head_dim=8, num_layers=2)
        steps = torch.cat([m(idx[:, t : t + 1], kv_cache=kv)[0] for t in range(9)], dim=1)
    assert_close(steps, full, atol=1e-5, rtol=1e-5)
