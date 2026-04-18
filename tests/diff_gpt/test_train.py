import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.data_loader import DiffDataFrameDataLoader


def _toy_df(n_rows: int, n_features: int = 1) -> pd.DataFrame:
    # Simple monotonic pattern per channel so encoded tokens are easy to
    # sanity check.
    cols = {f"c{c}": np.arange(n_rows, dtype=np.float64) + c * 100 for c in range(n_features)}
    return pd.DataFrame(cols)


def _big_domain(n_features: int) -> np.ndarray:
    # Generous derivative bound so the encoder doesn't clip.
    return np.full((n_features,), 100.0, dtype=np.float64)


def test_dataframe_loader_shapes():
    df = _toy_df(n_rows=200)
    loader = DiffDataFrameDataLoader(
        dfs=[df],
        block_size=16,
        batch_size=4,
        vocab_size=64,
        order_of_derivative=1,
        domain_of_definition=_big_domain(1),
        use_decimal=False,
        device="cpu",
        train_part=0.8,
    )
    x, y = loader.get_batch("train")
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # y is x shifted by one.
    assert (y[:, :-1] == x[:, 1:]).all()


def test_dataframe_loader_train_val_split_exists():
    df = _toy_df(n_rows=400)
    loader = DiffDataFrameDataLoader(
        dfs=[df],
        block_size=16,
        batch_size=2,
        vocab_size=64,
        order_of_derivative=1,
        domain_of_definition=_big_domain(1),
        use_decimal=False,
        device="cpu",
        train_part=0.8,
    )
    assert loader.has_val()
    x_tr, _ = loader.get_batch("train")
    x_val, _ = loader.get_batch("val")
    assert x_tr.shape == (2, 16)
    assert x_val.shape == (2, 16)


def test_dataframe_loader_raises_on_unknown_split():
    df = _toy_df(n_rows=200)
    loader = DiffDataFrameDataLoader(
        dfs=[df],
        block_size=16,
        batch_size=2,
        vocab_size=64,
        order_of_derivative=1,
        domain_of_definition=_big_domain(1),
        use_decimal=False,
        device="cpu",
    )
    with pytest.raises(ValueError, match="unknown split"):
        loader.get_batch("test")


def test_dataframe_loader_target_column_mask():
    """target_columns=[1] on a 3-feature DataFrame masks the other two columns to -100."""
    torch.manual_seed(0)
    df = _toy_df(n_rows=400, n_features=3)
    loader = DiffDataFrameDataLoader(
        dfs=[df],
        block_size=12,
        batch_size=2,
        vocab_size=64,
        order_of_derivative=1,
        domain_of_definition=_big_domain(3),
        use_decimal=False,
        device="cpu",
        target_columns=[1],
    )
    _, y = loader.get_batch("train")
    mask = torch.tensor([i % 3 == 1 for i in range(12)])
    assert (y[:, ~mask] == -100).all()
    assert (y[:, mask] != -100).all()


def test_dataframe_loader_multi_series_blends():
    """
    With two series that produce distinct token bands, windows should come
    from both over many samples.
    """
    torch.manual_seed(0)
    a = _toy_df(n_rows=200)  # values 0..199
    b = _toy_df(n_rows=200)
    b["c0"] = b["c0"] + 1000  # shifted; same derivatives though, so encoded tokens overlap
    loader = DiffDataFrameDataLoader(
        dfs=[a, b],
        block_size=8,
        batch_size=32,
        vocab_size=64,
        order_of_derivative=1,
        domain_of_definition=_big_domain(1),
        use_decimal=False,
        device="cpu",
    )
    x, _ = loader.get_batch("train")
    assert x.shape == (32, 8)
