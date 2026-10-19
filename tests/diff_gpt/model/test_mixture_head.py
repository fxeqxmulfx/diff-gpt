import torch
import torch.nn.functional as F

from diff_gpt.model.engine import Engine
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.median import MedianSampler
from diff_gpt.sampler.temperature import TemperatureSampler


def _model(V=64, K=4):
    torch.manual_seed(0)
    return GPT(vocab_size=V, n_embd=16, block_size=16, n_head=2, n_layer=1,
               use_checkpoint=False, head="mixture", mixture_components=K)


def test_mixture_head_outputs_normalised_log_probs():
    m = _model()
    idx = torch.randint(0, 64, (3, 10))
    logits, loss = m(idx)
    assert logits.shape == (3, 10, 64)
    assert loss is None
    total = logits.exp().sum(-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)


def test_mixture_head_loss_and_gradients():
    m = _model()
    idx = torch.randint(0, 64, (2, 12))
    tgt = torch.randint(0, 64, (2, 12))
    tgt[0, :3] = -100  # ignore_index must be honoured
    _, loss = m(idx, tgt)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.mix_head.weight.grad is not None and m.mix_head.weight.grad.abs().sum() > 0
    assert not hasattr(m, "lm_head")


def test_mixture_head_mass_follows_component_mean():
    """Force one sharp component at bin 40: that bin must dominate."""
    m = _model(V=64, K=1)
    with torch.no_grad():
        m.mix_head.weight.zero_()
        m.mix_head.bias.copy_(torch.tensor([0.0, 40.0 - 32.0, -2.0]))  # pi, mu (bin units from centre), log_s
    logits, _ = m(torch.randint(0, 64, (1, 5)))
    assert (logits.argmax(-1) == 40).all()
    assert logits[0, 0, 40].exp() > 0.9


def test_mixture_head_generation_with_engine_and_samplers():
    m = _model().eval()
    eng = Engine(model=m)
    for sampler in (None, TemperatureSampler(0.0), TemperatureSampler(1.0), MedianSampler()):
        out, _ = eng.generate_batch([1, 2, 3], num_samples=2, max_tokens=5, sampler=sampler)
        assert len(out) == 2 and len(out[0]) == 8
        assert all(0 <= t < 64 for row in out for t in row)


def test_softmax_head_unchanged_by_default():
    torch.manual_seed(0)
    m = GPT(vocab_size=64, n_embd=16, block_size=16, n_head=2, n_layer=1, use_checkpoint=False)
    assert m.head == "softmax" and hasattr(m, "lm_head") and not hasattr(m, "mix_head")
    logits, _ = m(torch.randint(0, 64, (1, 4)))
    assert logits.abs().max() <= 15.0  # softcap still applied


def test_mixture_target_nll_matches_full_cross_entropy():
    """The training fast path must equal cross-entropy on the full bin log-probs,
    including ignore_index and the open-tail edge bins."""
    m = _model(V=64, K=4)
    idx = torch.randint(0, 64, (3, 12))
    tgt = torch.randint(0, 64, (3, 12))
    tgt[0, :2] = -100
    tgt[1, 3] = 0
    tgt[2, 5] = 63
    logits_none, fast = m(idx, tgt)
    assert logits_none is None
    m.label_smoothing_sigma = 1e-9  # force the full-distribution path
    full_logits, _ = m(idx)
    ref = F.cross_entropy(full_logits.reshape(-1, 64), tgt.reshape(-1), ignore_index=-100)
    assert torch.allclose(fast, ref, atol=1e-5), (fast.item(), ref.item())
