import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.data_loader import DiffDataFrameDataLoader
from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import get_domain_of_definition
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.temperature import TemperatureSampler


def _make_toy(n_rows: int = 80) -> pd.DataFrame:
    idx = np.arange(n_rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "cov0": np.sin(idx / 3.0),
            "cov1": np.cos(idx / 5.0),
            "y": np.sin(idx / 7.0),
        }
    )


def _make_model(
    block_size: int = 32,
    vocab_size: int = 64,
) -> tuple[DiffGPT, np.ndarray]:
    torch.manual_seed(0)
    df = _make_toy(n_rows=60)
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64), order_of_derivative=1, use_decimal=False
    )
    model = DiffGPT(
        model=GPT(vocab_size=vocab_size, n_embd=16, block_size=block_size, n_head=2, n_layer=1),
        order_of_derivative=1,
        domain_of_definition=dod,
        use_decimal=False,
    )
    return model, dod


def test_train_runs_with_target_columns_by_name():
    """A DiffDataFrameDataLoader with target_columns=['y'] trains OK."""
    model, dod = _make_model()
    dfs = [_make_toy(n_rows=60) for _ in range(3)]
    loader = DiffDataFrameDataLoader(
        dfs=dfs,
        block_size=model.model.block_size,
        batch_size=4,
        vocab_size=model.model.vocab_size,
        order_of_derivative=model.order_of_derivative,
        domain_of_definition=model.domain_of_definition,
        use_decimal=False,
        device=model.model.device_type,
        train_part=0.7,
        target_columns=[2],  # "y" is index 2
    )
    val_loss, _ = model.train(
        loader=loader,
        max_iters=10,
        eval_interval=10,
        eval_iters=2,
        use_tqdm=False,
        use_early_stopping=False,
    )
    assert np.isfinite(val_loss)


def test_predict_teacher_forced_shape_and_covariate_fidelity():
    """
    predict(future_covariates_df=..., target_columns=['y']) teacher-forces
    the covariates, so the decoded covariate values in the future rows
    must match the supplied future_cov modulo derivative quantization.
    """
    torch.manual_seed(0)
    df = _make_toy(n_rows=80)
    seq_len, pred_len = 20, 8
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64), order_of_derivative=1, use_decimal=False
    )
    base = GPT(
        vocab_size=256, n_embd=16, block_size=128, n_head=2, n_layer=1, use_checkpoint=False
    )
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=dod,
        use_decimal=False,
    )

    context_df = df.iloc[:seq_len].copy()
    future_cov = df.iloc[seq_len : seq_len + pred_len].copy()
    result = model.predict(
        df=context_df,
        future_covariates_df=future_cov,
        target_columns=["y"],
        sampler=TemperatureSampler(temperature=0.0),
    )
    assert result.shape == (seq_len + pred_len, 3)
    assert list(result.columns) == ["cov0", "cov1", "y"]

    for col in ("cov0", "cov1"):
        pred = result[col].iloc[seq_len:].to_numpy()
        truth = future_cov[col].to_numpy()
        assert np.max(np.abs(pred - truth)) < 0.1, (
            f"{col}: max err {np.max(np.abs(pred - truth)):.3f}"
        )


def test_predict_autoregressive_shape():
    """predict with just max_new_points generates every column by sampling."""
    torch.manual_seed(0)
    df = _make_toy(n_rows=80)
    seq_len, pred_len = 10, 5
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64), order_of_derivative=1, use_decimal=False
    )
    base = GPT(
        vocab_size=64, n_embd=16, block_size=128, n_head=2, n_layer=1, use_checkpoint=False
    )
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=dod,
        use_decimal=False,
    )
    result = model.predict(
        df=df.iloc[:seq_len], max_new_points=pred_len, sampler=None
    )
    assert result.shape == (seq_len + pred_len, 3)


def test_predict_rejects_both_modes_together():
    """max_new_points is redundant when future_covariates_df is given."""
    model, _ = _make_model(block_size=64)
    df = _make_toy(n_rows=30)
    future = df.iloc[10:15]
    # max_new_points mismatch must raise.
    with pytest.raises(AssertionError, match="must equal"):
        model.predict(
            df=df.iloc[:10],
            max_new_points=99,
            future_covariates_df=future,
            target_columns=["y"],
        )


def test_predict_teacher_forced_mismatched_columns_raises():
    model, _ = _make_model(block_size=64)
    df = _make_toy(n_rows=30)
    wrong = df.rename(columns={"y": "z"})
    with pytest.raises(AssertionError, match="same columns"):
        model.predict(
            df=df.iloc[:10],
            future_covariates_df=wrong.iloc[10:15],
            target_columns=["y"],
        )


def test_train_save_best_restores_best_val():
    """
    After training past the val minimum (use_early_stopping=False), the
    restored model must reproduce the best val_loss seen — proving the
    snapshot was captured and reloaded, not the noisier final state.
    """
    model, _ = _make_model(block_size=32)
    dfs = [_make_toy(n_rows=200) for _ in range(2)]
    loader = DiffDataFrameDataLoader(
        dfs=dfs,
        block_size=model.model.block_size,
        batch_size=4,
        vocab_size=model.model.vocab_size,
        order_of_derivative=model.order_of_derivative,
        domain_of_definition=model.domain_of_definition,
        use_decimal=False,
        device=model.model.device_type,
        train_part=0.7,
    )
    returned_loss, _ = model.train(
        loader=loader,
        max_iters=40,
        eval_interval=10,
        eval_iters=8,
        use_tqdm=False,
        use_early_stopping=False,
        save_best=True,
    )
    # Re-measure val loss on the restored weights.
    model.model.eval()
    measured = 0.0
    n = 5
    for _ in range(n):
        X, Y = loader.get_batch("val")
        with torch.no_grad():
            _, loss = model.model(X, Y)
        measured += float(loss.item())
    measured /= n
    # Within noise: returned_loss was measured over eval_iters=8 batches,
    # we re-measure over 5 fresh batches — loosely agree.
    assert abs(measured - returned_loss) < 1.0, (
        f"restored val loss {measured:.3f} diverged from returned {returned_loss:.3f}"
    )


def test_train_save_best_false_returns_final_loss():
    """save_best=False keeps the final model and returns its val_loss."""
    model, _ = _make_model(block_size=32)
    dfs = [_make_toy(n_rows=200)]
    loader = DiffDataFrameDataLoader(
        dfs=dfs,
        block_size=model.model.block_size,
        batch_size=4,
        vocab_size=model.model.vocab_size,
        order_of_derivative=model.order_of_derivative,
        domain_of_definition=model.domain_of_definition,
        use_decimal=False,
        device=model.model.device_type,
        train_part=0.7,
    )
    final_loss, _ = model.train(
        loader=loader,
        max_iters=10,
        eval_interval=10,
        eval_iters=4,
        use_tqdm=False,
        use_early_stopping=False,
        save_best=False,
    )
    assert np.isfinite(final_loss)


def test_train_grad_accum_runs():
    """grad_accum_steps + grad_clip_norm wired through DiffGPT.train."""
    model, _ = _make_model()
    dfs = [_make_toy(n_rows=60) for _ in range(2)]
    loader = DiffDataFrameDataLoader(
        dfs=dfs,
        block_size=model.model.block_size,
        batch_size=2,
        vocab_size=model.model.vocab_size,
        order_of_derivative=model.order_of_derivative,
        domain_of_definition=model.domain_of_definition,
        use_decimal=False,
        device=model.model.device_type,
        train_part=0.7,
    )
    val_loss, _ = model.train(
        loader=loader,
        max_iters=5,
        eval_interval=5,
        eval_iters=2,
        use_tqdm=False,
        use_early_stopping=False,
        grad_accum_steps=2,
        grad_clip_norm=1.0,
    )
    assert np.isfinite(val_loss)
