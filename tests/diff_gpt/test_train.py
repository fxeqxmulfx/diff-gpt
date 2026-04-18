import pytest

from diff_gpt.train import get_batch


def test_get_batch_raises_on_short_data():
    """Regression: previously crashed deep inside torch.randint with a cryptic error."""
    data = tuple(range(8))
    with pytest.raises(ValueError, match="block_size"):
        get_batch(data=data, batch_size=2, block_size=16, device="cpu")


def test_get_batch_shapes():
    data = tuple(range(100))
    batch_size = 4
    block_size = 8
    x, y = get_batch(data=data, batch_size=batch_size, block_size=block_size, device="cpu")
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    # y is x shifted by one
    for i in range(batch_size):
        start = int(x[i, 0].item())
        for j in range(block_size):
            assert int(x[i, j].item()) == start + j
            assert int(y[i, j].item()) == start + j + 1
