"""
Tests for how predict and anomaly_scores read the context.

The forecast must continue from the last context rows whatever context steps
clipped, targets may be named by negative position, the output is float64,
and the context needs a row past the derivative prefix.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from diff_gpt.diff_gpt import DiffGPT
from diff_gpt.encoder_decoder import TokenClippingWarning, encode
from diff_gpt.model.gpt import GPT
from diff_gpt.sampler.sampler import Sampler

V = 64
# Max distance of a decoded value from the expected one: a few quantization
# steps (Δ = 1/31 for D = 1), far below the shifts of 2 and more that the
# decoding from the first context row produced.
STEP_TOLERANCE = 0.1


class _ConstantSampler(Sampler):
    """Pick `token` at every position, whatever the logits."""

    def __init__(self, token: int) -> None:
        self.token = token

    def __call__(
        self,
        logits: torch.Tensor,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        del rng
        return torch.full(
            (*logits.shape[:-1], 1),
            self.token,
            dtype=torch.long,
            device=logits.device,
        )


def _zero_step_sampler(order: int, n_features: int = 1) -> _ConstantSampler:
    """Return a sampler that always picks the token of a zero k-th difference."""
    _, _, tokens = encode(
        np.zeros((order + 1, n_features)),
        V,
        np.ones(n_features),
        order,
    )
    return _ConstantSampler(int(tokens[0, 0]))


def _model(
    order: int,
    domain: np.ndarray,
    *,
    tokenizer: str | None = None,
) -> DiffGPT:
    torch.manual_seed(0)
    return DiffGPT(
        model=GPT(vocab_size=V, n_embd=16, block_size=64, n_head=2, n_layer=1),
        order_of_derivative=order,
        domain_of_definition=domain,
        tokenizer=tokenizer,
    )


def _covariate_frame() -> pd.DataFrame:
    """Return 12 rows: a 0.3-slope covariate and a constant target at 5."""
    t = np.arange(12, dtype=np.float64)
    return pd.DataFrame({"cov": 0.3 * t, "y": np.full(12, 5.0)})


def test_forecast_stays_at_the_last_context_value_after_a_clipped_step() -> None:
    """
    Check that a zero-step forecast stays at the last context value (k = 1).

    The context steps by 3 against a domain of 1, so that step clips. predict
    decoded the context and the forecast as one stream from the first row,
    which moved the forecast to 2.52 instead of 4.5.
    """
    model = _model(1, np.array([1.0]))
    ctx = pd.DataFrame({"x": [0.0, 0.5, 1.0, 4.0, 4.5, 4.5, 4.5, 4.5]})
    with pytest.warns(TokenClippingWarning):
        out = model.predict(df=ctx, max_new_points=4, sampler=_zero_step_sampler(1))
    np.testing.assert_array_equal(out["x"].iloc[:8], ctx["x"])
    np.testing.assert_allclose(out["x"].iloc[8:], 4.5, atol=STEP_TOLERANCE)


def test_forecast_continues_the_last_slope_after_clipped_steps() -> None:
    """
    Check that zero second differences continue the last context slope (k = 2).

    The second differences 3 and -2 clip against a domain of 1; the forecast
    came out as 3.07, 3.03, 3.00, 2.97 instead of 13, 14, 15, 16.
    """
    model = _model(2, np.array([1.0]))
    ctx = pd.DataFrame({"x": [0.0, 0, 0, 3, 6, 9, 10, 11, 12]})
    with pytest.warns(TokenClippingWarning):
        out = model.predict(df=ctx, max_new_points=4, sampler=_zero_step_sampler(2))
    np.testing.assert_allclose(
        out["x"].iloc[9:],
        [13.0, 14.0, 15.0, 16.0],
        atol=STEP_TOLERANCE,
    )


def test_teacher_forced_covariates_come_back_as_supplied_after_a_clipped_step() -> None:
    """
    Check that the future covariate is returned as supplied after a context jump.

    A +3 level shift of the covariate clips inside the context; the
    teacher-forced path decoded its future from the first context row as
    [3.10, 3.42, 3.71, 4.00] instead of the supplied [5.4, 5.7, 6.0, 6.3].
    """
    model = _model(1, np.array([1.0, 1.0]))
    df = _covariate_frame()
    df.loc[4:, "cov"] += 3.0
    ctx, fut = df.iloc[:8], df.iloc[8:]
    with pytest.warns(TokenClippingWarning):
        out = model.predict(
            df=ctx,
            future_covariates_df=fut,
            target_columns=["y"],
            sampler=_zero_step_sampler(1, 2),
        )
    np.testing.assert_allclose(out["cov"].iloc[8:], fut["cov"], atol=STEP_TOLERANCE)
    np.testing.assert_allclose(out["y"].iloc[8:], 5.0, atol=STEP_TOLERANCE)


def test_mash_k1_covariates_come_back_as_supplied_after_a_jump_at_the_end() -> None:
    """
    Check the teacher-forced covariates after a jump on the last context step.

    MASH at k = 1 makes a cut step up over the next tokens. Context and future
    were encoded as one frame, so the future covariate tokens carried that
    catch-up, and decoded from the last context row they came out up to 2.28
    above the supplied values.
    """
    model = _model(1, np.array([1.0, 1.0]), tokenizer="mash")
    df = _covariate_frame()
    df.loc[7:, "cov"] += 3.0
    ctx, fut = df.iloc[:8], df.iloc[8:]
    with pytest.warns(TokenClippingWarning):
        out = model.predict(
            df=ctx,
            future_covariates_df=fut,
            target_columns=["y"],
            sampler=_zero_step_sampler(1, 2),
        )
    np.testing.assert_allclose(out["cov"].iloc[8:], fut["cov"], atol=STEP_TOLERANCE)
    np.testing.assert_allclose(out["y"].iloc[8:], 5.0, atol=STEP_TOLERANCE)


def test_negative_target_column_is_sampled_like_its_name() -> None:
    """
    Check that target_columns=[-1] samples the last column, as its name does.

    p % n_features is never -1, so every future position used to be forced
    to the target's placeholder, and nothing was sampled.
    """
    model = _model(1, np.array([1.0, 1.0]))
    df = _covariate_frame()
    ctx, fut = df.iloc[:8], df.iloc[8:]
    sampler = _zero_step_sampler(1, 2)
    by_name = model.predict(
        df=ctx,
        future_covariates_df=fut,
        target_columns=["y"],
        sampler=sampler,
    )
    by_position = model.predict(
        df=ctx,
        future_covariates_df=fut,
        target_columns=[-1],
        sampler=sampler,
    )
    pd.testing.assert_frame_equal(by_name, by_position)
    for bad in ([2], [-3]):
        with pytest.raises(ValueError, match="out of range for 2 columns"):
            model.predict(
                df=ctx,
                future_covariates_df=fut,
                target_columns=bad,
                sampler=sampler,
            )


def test_predict_returns_float64_on_both_paths() -> None:
    """Check that predict returns float64 when teacher-forced and autoregressive."""
    model = _model(1, np.array([1.0, 1.0]))
    df = _covariate_frame()
    ctx, fut = df.iloc[:8], df.iloc[8:]
    sampler = _zero_step_sampler(1, 2)
    teacher_forced = model.predict(
        df=ctx,
        future_covariates_df=fut,
        target_columns=["y"],
        sampler=sampler,
    )
    autoregressive = model.predict(df=ctx, max_new_points=4, sampler=sampler)
    assert (teacher_forced.dtypes == np.float64).all()
    assert (autoregressive.dtypes == np.float64).all()
    np.testing.assert_allclose(teacher_forced["y"].iloc[8:], 5.0, atol=STEP_TOLERANCE)


def test_a_context_of_max_order_rows_is_rejected() -> None:
    """
    Check that a context without a row past the derivative prefix raises.

    The guards used > instead of >=, so such a context reached the Engine
    ("tokens must be non-empty") or failed to reshape 0 elements.
    """
    model = _model(2, np.array([1.0, 1.0]))
    df = _covariate_frame()
    ctx, fut = df.iloc[:2], df.iloc[2:5]
    with pytest.raises(ValueError, match="context needs a row"):
        model.predict(df=ctx, max_new_points=2)
    with pytest.raises(ValueError, match="context needs a row"):
        model.predict(df=ctx, future_covariates_df=fut, target_columns=["y"])
    with pytest.raises(ValueError, match="nothing to score"):
        model.anomaly_scores(ctx)
