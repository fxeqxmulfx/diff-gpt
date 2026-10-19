"""
Bayes-optimal point readouts from Monte-Carlo samples of the predictive
distribution.

A point forecast is a decision: for a loss ℓ the optimal point at each
horizon is argmin_y E[ℓ(X, y)] under the predictive distribution, estimated
here from N sampled trajectories. Feeding a per-step statistic back into the
model ("argmax path", "median path") is a plug-in heuristic that computes a
path statistic, not a marginal one, and is optimal for no loss in general.

  metric="mae"  (MAE / MASE)  -> marginal median
  metric="mse"                -> marginal mean
  metric="smape"              -> argmin of the expected sMAPE (upward-biased)
  metric="owa"                -> argmin of ½(E sMAPE / smape_ref + E|X−y| / (naive_scale·mase_ref)),
                                  M4's OWA with the caller's Naive2 reference values
"""

from __future__ import annotations

import numpy as np


def _expected_smape(samples: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """samples (N,), cands (K,) -> expected sMAPE (in percent) for each candidate, (K,)."""
    x = samples[None, :]
    y = cands[:, None]
    den = np.abs(x) + np.abs(y)
    num = 200.0 * np.abs(x - y)
    out = np.zeros_like(den)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out.mean(axis=1)


def bayes_point(
    samples: np.ndarray,
    metric: str = "mae",
    *,
    naive_scale: float | np.ndarray | None = None,
    smape_ref: float | None = None,
    mase_ref: float | None = None,
    grid: int = 401,
) -> np.ndarray:
    """Point forecast minimising the expected `metric` under the sample distribution.

    `samples`: shape (N, ...) with N trajectories on axis 0. Returns shape (...).
    For "owa", `naive_scale` is the MASE denominator (scalar, or one value per
    trailing column when samples are (N, H, F)), and `smape_ref` / `mase_ref`
    are the reference sMAPE and MASE that normalise the two terms.
    """
    samples = np.asarray(samples, dtype=np.float64)
    assert samples.ndim >= 2 and samples.shape[0] >= 1
    tail_shape = samples.shape[1:]
    flat = samples.reshape(samples.shape[0], -1)
    if metric in ("mae", "median"):
        return np.median(flat, axis=0).reshape(tail_shape)
    if metric in ("mse", "mean"):
        return flat.mean(axis=0).reshape(tail_shape)
    if metric not in ("smape", "owa"):
        raise ValueError(f"unknown metric {metric!r}")
    if metric == "owa":
        assert naive_scale is not None and smape_ref and mase_ref, (
            "owa needs naive_scale, smape_ref and mase_ref"
        )
        n_cols = tail_shape[-1] if len(tail_shape) >= 1 else 1
        ns = np.broadcast_to(np.asarray(naive_scale, dtype=np.float64), (n_cols,))
    out = np.empty(flat.shape[1], dtype=np.float64)
    for j in range(flat.shape[1]):
        s = flat[:, j]
        lo, hi = float(s.min()), float(s.max())
        cands = np.unique(np.concatenate([s, np.linspace(lo, hi, grid)]))
        obj = _expected_smape(s, cands)
        if metric == "owa":
            col = j % n_cols
            mae = np.abs(s[None, :] - cands[:, None]).mean(axis=1)
            obj = 0.5 * (obj / smape_ref + mae / (ns[col] * mase_ref))
        out[j] = cands[int(obj.argmin())]
    return out.reshape(tail_shape)
