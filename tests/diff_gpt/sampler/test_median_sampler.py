import torch

from diff_gpt.sampler.median import MedianSampler
from diff_gpt.sampler.temperature import TemperatureSampler


def test_median_of_known_distribution():
    # probs: 0.1, 0.2, 0.3, 0.4 -> cdf 0.1, 0.3, 0.6, 1.0 -> median bin = 2
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    out = MedianSampler()(probs.log())
    assert out.shape == (1, 1)
    assert out.item() == 2


def test_median_differs_from_argmax_on_skewed_distribution():
    # Mode at bin 0 but most of the mass sits on the right.
    probs = torch.tensor([[0.3, 0.15, 0.25, 0.2, 0.1]])  # cdf 0.30, 0.45, 0.70: median bin 2
    logits = probs.log()
    assert TemperatureSampler(0.0)(logits).item() == 0
    assert MedianSampler()(logits).item() == 2


def test_median_equals_argmax_for_symmetric_peaked_distribution():
    logits = torch.zeros(3, 64)
    logits[:, 20] = 10.0
    logits[:, 19] = logits[:, 21] = 5.0
    assert torch.equal(MedianSampler()(logits), TemperatureSampler(0.0)(logits))


def test_median_batch_shape_and_range():
    torch.manual_seed(0)
    logits = torch.randn(8, 256)
    out = MedianSampler()(logits)
    assert out.shape == (8, 1)
    assert out.dtype == torch.int64
    assert (out >= 0).all() and (out < 256).all()
