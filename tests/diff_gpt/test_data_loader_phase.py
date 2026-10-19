"""Window phase: the target mask must follow the real column of each token, and
align_windows must start every window at column 0."""
import numpy as np
import pandas as pd
import torch

from diff_gpt.data_loader import DiffDataFrameDataLoader

# Three columns with distinguishable tokens under k=1, V=64, D=31 (scale = 1):
# constant -> 32, slope +1 -> 33, slope -1 -> 31.
T = 600
_t = np.arange(T, dtype=np.float64)
DF = pd.DataFrame({"c0": np.zeros(T), "c1": _t, "c2": -_t})
DOD = np.array([31.0, 31.0, 31.0])
TOKEN = {0: 32, 1: 33, 2: 31}


def _loader(**kw):
    return DiffDataFrameDataLoader(
        dfs=[DF], block_size=32, batch_size=64, vocab_size=64, order_of_derivative=1,
        domain_of_definition=DOD, use_decimal=False, device="cpu", train_part=1.0,
        rng=torch.Generator().manual_seed(0), **kw,
    )


def test_target_mask_keeps_only_the_target_column_at_any_phase():
    for target in (0, 1, 2):
        loader = _loader(target_columns=[target])
        for _ in range(20):
            _, y = loader.get_batch("train")
            kept = y[y != -100]
            assert kept.numel() > 0
            assert (kept == TOKEN[target]).all(), (
                f"target {target}: unmasked tokens {kept.unique().tolist()} != {TOKEN[target]}"
            )


def test_unaligned_windows_start_at_every_phase():
    loader = _loader()
    first_cols = set()
    for _ in range(20):
        x, _ = loader.get_batch("train")
        first_cols |= set(x[:, 0].tolist())
    assert first_cols == {31, 32, 33}


def test_aligned_windows_always_start_at_column_zero():
    loader = _loader(align_windows=True)
    for _ in range(20):
        x, y = loader.get_batch("train")
        assert (x[:, 0] == TOKEN[0]).all()
        assert (x[:, 1] == TOKEN[1]).all()
        assert (y[:, 0] == TOKEN[1]).all()  # y is x shifted by one token


def test_aligned_windows_with_target_mask():
    loader = _loader(align_windows=True, target_columns=[2])
    _, y = loader.get_batch("train")
    kept = y[y != -100]
    assert (kept == TOKEN[2]).all()
    # y[:, j] holds column (1 + j) % 3, so column 2 sits at j = 1, 4, 7, ...
    assert (y[:, 1] != -100).all() and (y[:, 0] == -100).all() and (y[:, 2] == -100).all()


def test_aligned_split_point_keeps_val_windows_on_column_zero():
    """The train/val cut must land on a column boundary: otherwise val windows
    start mid-row and carry a different column phase than the train windows."""
    # 401 rows * 3 columns = 1200 tokens; train_part 0.7 -> 840, already a
    # multiple of 3, so use a length whose 70% is not.
    T2 = 335
    t = np.arange(T2, dtype=np.float64)
    df = pd.DataFrame({"c0": np.zeros(T2), "c1": t, "c2": -t})
    common = dict(dfs=[df], block_size=12, batch_size=8, vocab_size=64, order_of_derivative=1,
                  domain_of_definition=DOD, use_decimal=False, device="cpu", train_part=0.7)
    plain = DiffDataFrameDataLoader(**common, rng=torch.Generator().manual_seed(0))
    aligned = DiffDataFrameDataLoader(**common, align_windows=True, rng=torch.Generator().manual_seed(0))
    n_plain = len(plain._train_serieses[0])
    n_aligned = len(aligned._train_serieses[0])
    assert n_plain % 3 != 0, "pick a length where the naive split is misaligned"
    assert n_aligned % 3 == 0
    assert 0 <= n_plain - n_aligned < 3
    for _ in range(10):
        x, _ = aligned.get_batch("val")
        assert (x[:, 0] == TOKEN[0]).all()
        assert (x[:, 1] == TOKEN[1]).all()
