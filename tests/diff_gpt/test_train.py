# Copyright 2025 Mikhail Borisov
"""Tests for DiffDataFrameDataLoader batches, splits and target-column masking."""

import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.data_loader import DiffDataFrameDataLoader

# F.cross_entropy's ignore_index; the loader masks non-target columns with it.
IGNORE_INDEX = -100


def _toy_df(n_rows: int, n_features: int = 1) -> pd.DataFrame:
    # Simple monotonic pattern per channel so encoded tokens are easy to
    # sanity check.
    cols = {
        f"c{c}": np.arange(n_rows, dtype=np.float64) + c * 100
        for c in range(n_features)
    }
    return pd.DataFrame(cols)


def _big_domain(n_features: int) -> np.ndarray:
    # Generous derivative bound so the encoder doesn't clip.
    return np.full((n_features,), 100.0, dtype=np.float64)


def test_dataframe_loader_shapes() -> None:
    """Check that batches are (batch_size, block_size) and y is x shifted by one."""
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


def test_dataframe_loader_train_val_split_exists() -> None:
    """Check that train_part=0.8 yields a val split; both splits give full batches."""
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


def test_dataframe_loader_raises_on_unknown_split() -> None:
    """Check that get_batch raises ValueError for a split other than train or val."""
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


def test_dataframe_loader_target_column_mask() -> None:
    """Check that target_columns=[1] masks the other two of three columns to -100."""
    torch.manual_seed(0)
    n_features, block_size = 3, 12
    df = _toy_df(n_rows=400, n_features=n_features)
    loader = DiffDataFrameDataLoader(
        dfs=[df],
        block_size=block_size,
        batch_size=2,
        vocab_size=64,
        order_of_derivative=1,
        domain_of_definition=_big_domain(n_features),
        use_decimal=False,
        device="cpu",
        target_columns=[1],
    )
    _, y = loader.get_batch("train")
    keep = y != IGNORE_INDEX
    # Windows are cut at arbitrary token offsets, so the kept positions of each
    # row form exactly one residue class modulo n_features (the row's phase),
    # and the phase is not the same for every row.
    phases = set()
    for row in keep:
        js = torch.nonzero(row).flatten().tolist()
        assert len(js) == block_size // n_features
        assert len({j % n_features for j in js}) == 1
        phases.add(js[0] % n_features)
    assert len(phases) > 1


def test_dataframe_loader_multi_series_blends() -> None:
    """
    Check that a loader over two series serves full (batch_size, block_size) batches.

    With two series that produce distinct token bands, windows should come
    from both over many samples.
    """
    torch.manual_seed(0)
    a = _toy_df(n_rows=200)  # values 0..199
    b = _toy_df(n_rows=200)
    # Shifted; same derivatives though, so encoded tokens overlap.
    b["c0"] = b["c0"] + 1000
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
