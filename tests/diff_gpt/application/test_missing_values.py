"""
Missing values in the loaders and in DiffGPT: filled, and kept out of the loss.

NaN in a DataFrame, or NULL in a SQLite table, marks a missing value. The
series is encoded as `fill_missing` fills it; the targets whose tokens read
a missing value are masked with -100, and their anomaly scores are NaN.
"""

import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.application.data_loader import DiffDataFrameDataLoader
from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.domain.encoder_decoder import encode, fill_missing
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.infrastructure.sqlite_data_loader import DiffSQLiteDataLoader

if TYPE_CHECKING:
    from pathlib import Path

V = 64
# F.cross_entropy's ignore_index; the loaders mask targets with it.
IGNORE_INDEX = -100
# sin(t) at the order 1 on [-1, 1]: 20 rows, 19 tokens. The NaN at row 10
# enters the steps at rows 10 and 11, tokens 9 and 10.
N_ROWS = 20
NAN_ROW = 10
UNOBSERVED = (9, 10)
SIN_DOMAIN = np.array([1.0])
# The model's columns: cov steps by at most 1/3, y by -0.15 to 0.25.
DOMAIN = np.array([[-1.0, 1.0], [-0.5, 1.5]])
# One level of the cov column, 2/(V - 2): the teacher-forced covariates
# decode within half of it.
COV_LEVEL = 2 / (V - 2)
CONTEXT_ROWS = 30
# Rows of each training frame: 398 tokens, so that both splits have windows.
TRAIN_ROWS = 200


def _sin_with_a_gap() -> np.ndarray:
    """Return sin(t) for t = 0..19 with the row 10 missing."""
    x = np.sin(np.arange(N_ROWS, dtype=np.float64))
    x[NAN_ROW] = np.nan
    return x


def _sin_tokens() -> np.ndarray:
    """Return the tokens of the filled `_sin_with_a_gap`."""
    _, _, tokens = encode(fill_missing(_sin_with_a_gap()[:, None]), V, SIN_DOMAIN, 1)
    return tokens[:, 0]


def test_dataframe_loader_masks_the_targets_that_read_a_nan() -> None:
    """
    Check the one window of the series: inputs filled, targets 8 and 9 masked.

    Targets y[i] are the tokens i + 1, so tokens 9 and 10 are y[8] and y[9].
    """
    batch_size = 4
    loader = DiffDataFrameDataLoader(
        dfs=[pd.DataFrame({"x": _sin_with_a_gap()})],
        block_size=N_ROWS - 2,
        batch_size=batch_size,
        vocab_size=V,
        order_of_derivative=1,
        domain_of_definition=SIN_DOMAIN,
        device="cpu",
        train_part=1.0,
    )
    inputs, targets = loader.get_batch("train")
    tokens = torch.from_numpy(_sin_tokens())
    expected = tokens[1:].clone()
    expected[[t - 1 for t in UNOBSERVED]] = IGNORE_INDEX
    assert torch.equal(inputs, tokens[:-1].repeat(batch_size, 1))
    assert torch.equal(targets, expected.repeat(batch_size, 1))


def test_sqlite_loader_masks_the_targets_that_read_a_null(tmp_path: Path) -> None:
    """
    Check get_data and get_batch on a table whose row 10 is NULL.

    get_data marks tokens 9 and 10 unobserved, and a window starts at token
    0 or 1, so it masks y[8] and y[9], or y[7] and y[8].
    """
    db_file = tmp_path / "gaps.db"
    connection = sqlite3.connect(db_file)
    connection.execute("CREATE TABLE gaps (dt INTEGER PRIMARY KEY, x REAL)")
    connection.executemany(
        "INSERT INTO gaps VALUES (?, ?)",
        [(t, None if np.isnan(v) else v) for t, v in enumerate(_sin_with_a_gap())],
    )
    connection.commit()
    connection.close()
    loader = DiffSQLiteDataLoader(
        block_size=N_ROWS - 3,
        batch_size=8,
        vocab_size=V,
        order_of_derivative=1,
        domain_of_definition=SIN_DOMAIN,
        device="cpu",
        database=str(db_file),
    )
    values, observed = loader.get_data("gaps", 0, N_ROWS - 1)
    assert values == tuple(_sin_tokens().tolist())
    assert observed == tuple(t not in UNOBSERVED for t in range(N_ROWS - 1))
    _, targets = loader.get_batch()
    masked = {
        tuple(torch.nonzero(row == IGNORE_INDEX)[:, 0].tolist()) for row in targets
    }
    assert masked <= {(7, 8), (8, 9)}
    loader.close()


def _model(block_size: int = 128) -> DiffGPT:
    torch.manual_seed(0)
    return DiffGPT(
        model=GPT(vocab_size=V, n_embd=16, block_size=block_size, n_head=2, n_layer=1),
        order_of_derivative=1,
        domain_of_definition=DOMAIN,
    )


def _frame(n_rows: int = CONTEXT_ROWS) -> pd.DataFrame:
    t = np.arange(n_rows, dtype=np.float64)
    return pd.DataFrame({"cov": np.sin(t / 3.0), "y": np.cos(t / 5.0) + 0.05 * t})


def _filled(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(fill_missing(df.to_numpy(dtype=np.float64)), columns=df.columns)


def test_predict_forecasts_a_context_with_gaps_as_its_fill() -> None:
    """
    Check that predict reads the NaN as filled and returns the context as given.

    A gap on the last row holds the last value, so the forecast continues
    from there; predict_many forecasts the same.
    """
    model = _model()
    df = _frame()
    df.iloc[[3, 4], 0] = np.nan
    df.iloc[-1, 1] = np.nan
    out = model.predict(df=df, max_new_points=5)
    expected = model.predict(df=_filled(df), max_new_points=5)
    pd.testing.assert_frame_equal(out.iloc[: len(df)], df)
    pd.testing.assert_frame_equal(out.iloc[len(df) :], expected.iloc[len(df) :])
    pd.testing.assert_frame_equal(model.predict_many([df], max_new_points=5)[0], out)


def test_teacher_forced_covariates_take_the_line_across_their_gaps() -> None:
    """Check that a NaN future covariate is forced as the line across its gap."""
    model = _model()
    df = _frame(CONTEXT_ROWS + 6)
    ctx, fut = df.iloc[:CONTEXT_ROWS], df.iloc[CONTEXT_ROWS:].copy()
    fut.iloc[[2, 3], 0] = np.nan
    out = model.predict(df=ctx, future_covariates_df=fut, target_columns=["y"])
    filled = fill_missing(np.concatenate([ctx.to_numpy(), fut.to_numpy()]))
    np.testing.assert_allclose(
        out["cov"].iloc[CONTEXT_ROWS:],
        filled[CONTEXT_ROWS:, 0],
        atol=COV_LEVEL,
    )


def test_anomaly_scores_are_nan_at_the_steps_that_read_a_nan() -> None:
    """Check that the steps into and out of a NaN score NaN, the rest as filled."""
    model = _model()
    df = _frame()
    df.iloc[NAN_ROW, 1] = np.nan
    expected = model.anomaly_scores(_filled(df))
    expected.iloc[[NAN_ROW, NAN_ROW + 1], 1] = np.nan
    pd.testing.assert_frame_equal(model.anomaly_scores(df), expected)


def test_calibration_scores_only_the_truth_that_is_not_nan() -> None:
    """
    Check calibrate_temperature on truth with gaps.

    A NaN in the truth leaves the scores finite, a window of NaN changes
    nothing, and windows of NaN alone raise.
    """
    model = _model()
    df = _frame(CONTEXT_ROWS + 4)
    ctx = df.iloc[:CONTEXT_ROWS]
    truth = df.iloc[CONTEXT_ROWS:].to_numpy(copy=True)
    truth[1, 0] = np.nan
    missing = np.full_like(truth, np.nan)
    settings = {"max_new_points": 4, "temperatures": (0.5, 1.0), "num_samples": 4}
    best, scores = model.calibrate_temperature([(ctx, truth)], **settings)
    assert np.isfinite(list(scores.values())).all()
    assert model.calibrate_temperature(
        [(ctx, truth), (ctx, missing)],
        **settings,
    ) == (best, scores)
    with pytest.raises(ValueError, match="no calibration window"):
        model.calibrate_temperature([(ctx, missing)], **settings)


def test_training_on_frames_with_gaps_gives_a_finite_loss() -> None:
    """Check that the masked targets keep the NaN in the data out of the loss."""
    model = _model()
    rng = np.random.default_rng(0)
    dfs = []
    for _ in range(3):
        df = _frame(TRAIN_ROWS)
        for c in range(2):
            df.iloc[rng.choice(TRAIN_ROWS, 10, replace=False), c] = np.nan
        dfs.append(df)
    loader = DiffDataFrameDataLoader(
        dfs=dfs,
        block_size=model.model.block_size,
        batch_size=4,
        vocab_size=V,
        order_of_derivative=model.order_of_derivative,
        domain_of_definition=model.domain_of_definition,
        device=model.model.device_type,
        train_part=0.6,
    )
    val_loss, _ = model.train(
        loader=loader,
        max_iters=5,
        eval_interval=5,
        eval_iters=2,
        use_tqdm=False,
        use_early_stopping=False,
    )
    assert np.isfinite(val_loss)
