"""
Per-window normalization: one vocabulary for series of every scale.

The tokens hold the steps c(B) x on one grid (`encoder_decoder.encode`), so a
model that serves many series needs their values in common units.
`window_scale` reads a location and a scale off a forecast's context,
`normalize` puts the context in those units and `denormalize` takes the
forecast back. Two modules of diff_gpt.application use them:
`normalization.predict_samples_normalized` does all three around a model, and
`data_loader.DiffCorpusDataLoader` divides each training window by the
statistics of a prefix of it, with the same function, so that training and
forecasting see the same units.

The methods, per column:
- "mean_abs": location 0 and the mean |x|, as Chronos scales;
- "z": the mean and the standard deviation of x;
- "mean_abs_step": the mean of x and the mean |x_t - x_{t-1}|, the error of
  the naive forecast (the denominator of MASE at lag 1). The steps that the
  tokens hold above the order 0 then have a mean size of 1 in every context.

Only the observed values count: NaN marks a missing one, and a step needs
both of its values. A scale that is 0 or has nothing to read falls back to
the mean |x|, then to 1, so a constant context keeps its level's units.
"""

from typing import NamedTuple

import numpy as np

NORMALIZATIONS = ("mean_abs", "z", "mean_abs_step")


class WindowScale(NamedTuple):
    """The location and the scale of each column, float64 of shape (F,)."""

    loc: np.ndarray
    scale: np.ndarray  # > 0


def _mean(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Return the mean over the rows where `observed`, per column; 0 where none."""
    count = observed.sum(axis=0)
    total = np.where(observed, values, 0.0).sum(axis=0)
    return np.divide(total, count, out=np.zeros(values.shape[1]), where=count > 0)


def window_scale(context: np.ndarray, method: str = "mean_abs_step") -> WindowScale:
    """
    Return the location and scale of each column of the float64 (T, F) `context`.

    `method` is one of `NORMALIZATIONS` (see the module docstring). NaN marks a
    missing value; infinities raise ValueError, as in `fill_missing`.
    """
    if method not in NORMALIZATIONS:
        msg = f"unknown normalization {method!r}; expected one of {NORMALIZATIONS}"
        raise ValueError(msg)
    if context.ndim != 2:  # noqa: PLR2004
        msg = f"context must have shape (T, F), got {context.shape}"
        raise ValueError(msg)
    if np.isinf(context).any():
        msg = "context holds infinity: only NaN marks a missing value"
        raise ValueError(msg)
    observed = ~np.isnan(context)
    mean = _mean(context, observed)
    mean_abs = _mean(np.abs(context), observed)
    if method == "mean_abs":
        loc, scale = np.zeros_like(mean), mean_abs
    elif method == "z":
        loc, scale = mean, np.sqrt(_mean((context - mean) ** 2, observed))
    else:
        both = observed[1:] & observed[:-1]
        loc, scale = mean, _mean(np.abs(np.diff(context, axis=0)), both)
    scale = np.where(scale > 0, scale, mean_abs)
    return WindowScale(loc=loc, scale=np.where(scale > 0, scale, 1.0))


def normalize(values: np.ndarray, window: WindowScale) -> np.ndarray:
    """Return (values - loc) / scale, per column on the last axis of `values`."""
    return (values - window.loc) / window.scale


def denormalize(values: np.ndarray, window: WindowScale) -> np.ndarray:
    """Return values · scale + loc, the inverse of `normalize`."""
    return values * window.scale + window.loc
