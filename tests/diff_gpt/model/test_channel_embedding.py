import torch
from torch.testing import assert_close

from diff_gpt.model.engine import Engine
from diff_gpt.model.gpt import GPT
from diff_gpt.model.kv_cache import KVCache


def _gpt(n_channels, **kw):
    torch.manual_seed(0)
    return GPT(vocab_size=64, n_embd=16, block_size=24, n_head=2, n_layer=2,
               use_checkpoint=False, n_channels=n_channels, **kw)


def test_default_has_no_channel_embedding():
    m = _gpt(1)
    assert not hasattr(m, "channel_embedding")


def test_channel_embedding_shapes_and_gradient():
    m = _gpt(3)
    idx = torch.randint(0, 64, (2, 12))
    logits, loss = m(idx, torch.randint(0, 64, (2, 12)))
    assert logits.shape == (2, 12, 64) and torch.isfinite(loss)
    loss.backward()
    assert m.channel_embedding.weight.grad.abs().sum() > 0


def test_incremental_kv_decoding_keeps_channel_phase():
    """Feeding tokens one at a time through the KV cache must give the same
    logits as one full forward: the channel index comes from the absolute
    position, which the cache tracks."""
    m = _gpt(3).eval()
    idx = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        full, _ = m(idx)
        kv = KVCache(batch_size=1, num_heads=2, seq_len=32, head_dim=8, num_layers=2)
        steps = [m(idx[:, t : t + 1], kv_cache=kv)[0] for t in range(9)]
    assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)


def test_engine_generation_and_phase_aligned_sliding_window():
    m = _gpt(3).eval()
    out, _ = Engine(model=m).generate_batch([1, 2, 3, 4, 5, 6], num_samples=2, max_tokens=6, sampler=None)
    assert len(out[0]) == 12
    # generate() slides a window longer than block_size: must not crash and
    # must keep the window start on a channel boundary.
    idx = torch.randint(0, 64, (1, 6))
    gen = m.generate(idx, max_new_tokens=30, sampler=None, seed=0)
    assert gen.shape == (1, 36)
