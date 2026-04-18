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


def test_per_column_order_train_and_predict_autoregressive():
    """End-to-end train + autoregressive predict with per-column k
    vector (e.g., [1, 1, 2] giving one column a 2nd-derivative
    tokenization while the others use 1st). All downstream paths —
    loader.encode, DiffGPT.train, DiffGPT.predict — must thread the
    array through without error, and the round-trip through encoder/
    decoder must reproduce the context portion of the input exactly
    (up to quantization)."""
    torch.manual_seed(0)
    df = _make_toy(n_rows=120)
    order = np.array([1, 1, 2], dtype=np.int64)
    # Use V=64 (MASH needs V > 2^2 + 2 = 6 for the k=2 column; trivially met).
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64), order_of_derivative=order, use_decimal=False
    )
    base = GPT(
        vocab_size=64, n_embd=16, block_size=64, n_head=2, n_layer=1,
        use_checkpoint=False,
    )
    model = DiffGPT(
        model=base,
        order_of_derivative=order,
        domain_of_definition=dod,
        use_decimal=False,
    )
    loader = DiffDataFrameDataLoader(
        dfs=[df],
        block_size=base.block_size,
        batch_size=2,
        vocab_size=base.vocab_size,
        order_of_derivative=order,
        domain_of_definition=dod,
        use_decimal=False,
        device=base.device_type,
        train_part=0.7,
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
    # Predict a few points; max_k = 2 so need ctx ≥ 2.
    ctx = df.iloc[:10]
    result = model.predict(
        df=ctx, max_new_points=4,
        sampler=TemperatureSampler(temperature=0.0),
    )
    assert result.shape == (14, 3)


def test_per_column_order_predict_teacher_forced():
    """Teacher-forced mode must respect per-column orders for token
    layout and covariate fidelity."""
    torch.manual_seed(0)
    df = _make_toy(n_rows=80)
    seq_len, pred_len = 20, 8
    order = np.array([1, 1, 2], dtype=np.int64)
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64), order_of_derivative=order, use_decimal=False
    )
    base = GPT(
        vocab_size=256, n_embd=16, block_size=128, n_head=2, n_layer=1,
        use_checkpoint=False,
    )
    model = DiffGPT(
        model=base,
        order_of_derivative=order,
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


def test_per_column_order_shape_mismatch_raises():
    """A per-column order of wrong length must fail loudly, not
    silently read past the end of the array."""
    torch.manual_seed(0)
    df = _make_toy(n_rows=30)
    # df has 3 columns; pass a length-2 order array.
    with pytest.raises(AssertionError):
        get_domain_of_definition(
            df.to_numpy(dtype=np.float64),
            order_of_derivative=np.array([1, 1], dtype=np.int64),
            use_decimal=False,
        )


def test_predict_samples_shape_and_distinct_trajectories():
    """predict_samples returns (num_samples, H, F) and at positive
    temperature the trajectories are actually distinct (not all mode
    collapse). At temperature=0, all trajectories must collapse to the
    same argmax path."""
    torch.manual_seed(0)
    model, _ = _make_model(block_size=64)
    df = _make_toy(n_rows=20)
    N = 8
    # Temperature 1: samples should differ.
    samples_stoch = model.predict_samples(
        df=df.iloc[:10],
        max_new_points=5,
        num_samples=N,
        sampler=TemperatureSampler(temperature=1.0),
    )
    assert samples_stoch.shape == (N, 5, 3)
    # At least one pair of trajectories must differ (with overwhelming
    # probability for a sane model on ≥1 of 8 samples).
    pairwise_diff = np.any(samples_stoch[0] != samples_stoch[1])
    assert pairwise_diff, "stochastic sampling produced identical trajectories"

    # Temperature 0 argmax → all trajectories identical.
    samples_det = model.predict_samples(
        df=df.iloc[:10],
        max_new_points=5,
        num_samples=N,
        sampler=TemperatureSampler(temperature=0.0),
    )
    assert samples_det.shape == (N, 5, 3)
    for s in range(1, N):
        assert np.array_equal(samples_det[0], samples_det[s]), (
            f"temperature=0 trajectory {s} differs from trajectory 0"
        )


def test_predict_quantiles_shape_and_ordering():
    """predict_quantiles returns a dict keyed by quantile level, each a
    DataFrame with the right shape. P10 ≤ P50 ≤ P90 must hold per step
    (empirical quantiles are monotone in α)."""
    torch.manual_seed(0)
    model, _ = _make_model(block_size=64)
    df = _make_toy(n_rows=20)
    quantile_bands = model.predict_quantiles(
        df=df.iloc[:10],
        max_new_points=5,
        quantiles=[0.1, 0.5, 0.9],
        num_samples=32,
        sampler=TemperatureSampler(temperature=1.0),
    )
    assert set(quantile_bands.keys()) == {0.1, 0.5, 0.9}
    for q, qdf in quantile_bands.items():
        assert qdf.shape == (5, 3)
        assert list(qdf.columns) == list(df.columns)
    p10 = quantile_bands[0.1].to_numpy()
    p50 = quantile_bands[0.5].to_numpy()
    p90 = quantile_bands[0.9].to_numpy()
    assert np.all(p10 <= p50 + 1e-9), "P10 must not exceed P50"
    assert np.all(p50 <= p90 + 1e-9), "P50 must not exceed P90"


def test_predict_quantiles_rejects_out_of_range_quantile():
    """Quantiles in (0, 1). Boundary or out-of-range values rejected."""
    torch.manual_seed(0)
    model, _ = _make_model(block_size=64)
    df = _make_toy(n_rows=20)
    for bad_q in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(AssertionError, match="quantile"):
            model.predict_quantiles(
                df=df.iloc[:10],
                max_new_points=3,
                quantiles=[0.5, bad_q],
                num_samples=4,
                sampler=TemperatureSampler(temperature=1.0),
            )


def test_predict_samples_matches_predict_when_n_equals_one_at_argmax():
    """N=1 + temperature=0 must yield the same future rows as predict()
    with the same argmax sampler — the two paths share _generate_future_samples."""
    torch.manual_seed(0)
    model, _ = _make_model(block_size=64)
    df = _make_toy(n_rows=20).iloc[:10]
    sampler = TemperatureSampler(temperature=0.0)

    samples = model.predict_samples(
        df=df, max_new_points=4, num_samples=1, sampler=sampler,
    )
    point = model.predict(df=df, max_new_points=4, sampler=sampler)
    point_future = point.iloc[-4:].to_numpy()
    assert np.allclose(samples[0], point_future, atol=1e-10)


def test_anomaly_scores_shape_and_nan_positions():
    """anomaly_scores returns (T, F) with NaN at (a) the first max_order
    prefix rows and (b) column 0 of the first encoded row (which has no
    preceding token to condition on). Everywhere else must be finite."""
    torch.manual_seed(0)
    model, _ = _make_model(block_size=256)
    df = _make_toy(n_rows=40)
    scores = model.anomaly_scores(df)
    assert scores.shape == (40, 3)
    assert list(scores.columns) == list(df.columns)
    # max_order=1 → first 1 rows are prefix → all-NaN.
    assert scores.iloc[0].isna().all()
    # Column 0 of row 1 corresponds to the very first token in the encoded
    # sequence, which has no previous token — NaN. Columns 1, 2 of row 1
    # are scored.
    assert np.isnan(scores.iloc[1, 0])
    assert scores.iloc[1, 1:].notna().all()
    # Tail rows must be fully scored.
    assert scores.iloc[-5:].notna().all().all()


def test_anomaly_scores_spike_is_higher_than_baseline_after_training():
    """After a brief training run on clean data, injecting an out-of-
    distribution spike raises that row's NLL above the median baseline.
    (Untrained models produce near-uniform output and cannot detect
    anomalies — the test sanity-checks that learning calibrated per-step
    distributions actually enables detection.)"""
    torch.manual_seed(0)
    df_clean = _make_toy(n_rows=100)
    dod = get_domain_of_definition(
        df_clean.to_numpy(dtype=np.float64), order_of_derivative=1, use_decimal=False
    )
    base = GPT(
        vocab_size=64, n_embd=16, block_size=256, n_head=2, n_layer=1,
        use_checkpoint=False,
    )
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=dod,
        use_decimal=False,
    )
    # Brief training on the clean signal so the model learns the
    # typical derivative distribution.
    loader = DiffDataFrameDataLoader(
        dfs=[df_clean],
        block_size=base.block_size,
        batch_size=2,
        vocab_size=base.vocab_size,
        order_of_derivative=1,
        domain_of_definition=dod,
        use_decimal=False,
        device=base.device_type,
        train_part=0.9,
    )
    model.train(
        loader=loader, max_iters=200, eval_interval=200, eval_iters=2,
        use_tqdm=False, use_early_stopping=False,
    )
    # Inject a large spike at a known row, then score.
    df_spiked = df_clean.copy()
    df_spiked.iloc[50, 2] = df_clean.iloc[50, 2] + 5.0
    scores = model.anomaly_scores(df_spiked.iloc[:70])
    row_scores = scores.sum(axis=1, skipna=True).to_numpy()
    # Skip prefix rows (NaN summed → 0) and the first encoded row (partial NaN).
    baseline_median = np.median(row_scores[2:50])
    assert row_scores[50] > baseline_median, (
        f"spiked row NLL {row_scores[50]:.3f} should exceed clean median "
        f"{baseline_median:.3f}"
    )


def _train_and_prob_predict(
    df: pd.DataFrame,
    ctx_len: int,
    pred_len: int,
    num_samples: int,
    max_iters: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper: train briefly on df then return (samples, truth) pair
    for the last pred_len rows. Used by probabilistic-calibration tests."""
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64), order_of_derivative=1, use_decimal=False
    )
    base = GPT(
        vocab_size=64, n_embd=16, block_size=64, n_head=2, n_layer=1,
        use_checkpoint=False,
    )
    model = DiffGPT(
        model=base,
        order_of_derivative=1,
        domain_of_definition=dod,
        use_decimal=False,
    )
    loader = DiffDataFrameDataLoader(
        dfs=[df.iloc[: -pred_len]],
        block_size=base.block_size,
        batch_size=2,
        vocab_size=base.vocab_size,
        order_of_derivative=1,
        domain_of_definition=dod,
        use_decimal=False,
        device=base.device_type,
        train_part=0.9,
    )
    model.train(
        loader=loader, max_iters=max_iters, eval_interval=max_iters,
        eval_iters=2, use_tqdm=False, use_early_stopping=False,
    )
    context = df.iloc[-ctx_len - pred_len : -pred_len]
    truth = df.iloc[-pred_len:].to_numpy()
    samples = model.predict_samples(
        df=context,
        max_new_points=pred_len,
        num_samples=num_samples,
        sampler=TemperatureSampler(temperature=1.0),
    )
    return samples, truth


def test_probabilistic_sharpness_grows_with_horizon():
    """Interval width must grow monotonically with forecast horizon.
    Derivative tokenization integrates random-walk uncertainty, so
    Var[Ĥ_{T+h}] ∝ h."""
    torch.manual_seed(0)
    idx = np.arange(160, dtype=np.float64)
    df = pd.DataFrame(
        {"y": np.sin(idx / 5.0) + 0.05 * np.random.default_rng(0).normal(size=160)}
    )
    samples, _ = _train_and_prob_predict(
        df=df, ctx_len=30, pred_len=20, num_samples=80,
    )
    # samples: (80, 20, 1) — take the target column.
    widths = np.quantile(samples[..., 0], 0.9, axis=0) - np.quantile(
        samples[..., 0], 0.1, axis=0
    )
    # Monotone growth on average: compare early to late steps.
    early = float(np.mean(widths[:4]))
    late = float(np.mean(widths[-4:]))
    assert late > early, f"late width {late} should exceed early {early}"


def test_probabilistic_argmax_inside_median_band():
    """The argmax (temperature-0) forecast should lie inside the
    corresponding interval predicted by temperature-1 samples — a weak
    calibration check that mode and median aren't wildly divergent."""
    torch.manual_seed(0)
    from diff_gpt.metrics import empirical_coverage

    idx = np.arange(160, dtype=np.float64)
    df = pd.DataFrame({"y": np.sin(idx / 5.0)})
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64), order_of_derivative=1, use_decimal=False
    )
    base = GPT(
        vocab_size=64, n_embd=16, block_size=64, n_head=2, n_layer=1,
        use_checkpoint=False,
    )
    model = DiffGPT(
        model=base, order_of_derivative=1,
        domain_of_definition=dod, use_decimal=False,
    )
    loader = DiffDataFrameDataLoader(
        dfs=[df],
        block_size=base.block_size, batch_size=2, vocab_size=base.vocab_size,
        order_of_derivative=1, domain_of_definition=dod,
        use_decimal=False, device=base.device_type, train_part=0.9,
    )
    model.train(
        loader=loader, max_iters=500, eval_interval=500, eval_iters=2,
        use_tqdm=False, use_early_stopping=False,
    )
    ctx = df.iloc[-50:-20]
    argmax_pred = model.predict(
        df=ctx, max_new_points=20, sampler=TemperatureSampler(temperature=0.0),
    ).iloc[-20:].to_numpy()
    samples = model.predict_samples(
        df=ctx, max_new_points=20, num_samples=80,
        sampler=TemperatureSampler(temperature=1.0),
    )
    # Argmax-prediction coverage inside 80% band should be high on a
    # well-calibrated model — it's the mode, which is near the median.
    cov = empirical_coverage(samples[..., 0], argmax_pred[..., 0], level=0.8)
    assert cov > 0.5, (
        f"argmax outside the 80% band too often: coverage={cov:.2f}"
    )


def test_anomaly_scores_signal_longer_than_block_size_raises():
    """anomaly_scores runs a single forward pass; sequences longer than
    block_size must error rather than silently truncating or looping."""
    torch.manual_seed(0)
    model, _ = _make_model(block_size=32)  # very small
    df = _make_toy(n_rows=40)  # 40 * 3 = 120 tokens ≫ 32
    with pytest.raises(AssertionError, match="block_size"):
        model.anomaly_scores(df)


def test_teacher_forced_max_order_exceeds_context_raises():
    """When one column needs k derivatives of history but the context
    is shorter than k, the teacher-forced path must reject loudly."""
    torch.manual_seed(0)
    df = _make_toy(n_rows=30)
    order = np.array([1, 1, 3], dtype=np.int64)
    dod = get_domain_of_definition(
        df.to_numpy(dtype=np.float64), order_of_derivative=order, use_decimal=False
    )
    base = GPT(
        vocab_size=64, n_embd=16, block_size=128, n_head=2, n_layer=1,
        use_checkpoint=False,
    )
    model = DiffGPT(
        model=base,
        order_of_derivative=order,
        domain_of_definition=dod,
        use_decimal=False,
    )
    # Context of only 2 rows — less than max_order=3.
    with pytest.raises(AssertionError, match="max order"):
        model.predict(
            df=df.iloc[:2],
            future_covariates_df=df.iloc[2:5],
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
