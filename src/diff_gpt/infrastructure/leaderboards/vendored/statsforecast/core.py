"""
statsforecast's forecast of series stored end to end, one model at a time.

This is statsforecast 2.1.1's `statsforecast/core.py` (Apache-2.0, a copy of
the license in leaderboards/LICENSE-statsforecast), cut to GroupedArray's
`__init__`, `_get_cols`, `_output_fcst` and `forecast`, changed in two places
besides the style of leaderboards.vendored:
- StatsForecast, the DataFrame interface around GroupedArray, is left out:
  leaderboards builds the GroupedArray of its series itself and reads their
  forecasts from the array `forecast` returns, by row and column name.
  StatsForecast.forecast sorts the rows of a DataFrame by series and date,
  makes a GroupedArray of their float32 values, forecasts it with
  GroupedArray.forecast as here, and returns the same values in a DataFrame
  with each series' future dates. With it go its checks (of the dates
  against the frequency, of the model names, and that each `level` is in
  (0, 100)), its future dates, and its parallel backends (n_jobs > 1, and
  fugue's Dask, Ray and Spark): leaderboards ran it with one job, levels 20
  to 80 and a model of its own name.
- `forecast` takes no `fitted`, `X`, `verbose` and `target_col`: leaderboards
  asks for no in-sample values and has no exogenous features. With `X` go the
  `X` of `_output_fcst`, which only passes it on, and the `X` and `h` of
  `_get_cols`, which never reads them; with `verbose`, tqdm's progress bar, off
  unless asked for; and the result has no "times", the seconds it measured
  each model take.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np

from leaderboards.vendored.statsforecast.utils import _ensure_float
from leaderboards.vendored.utilsforecast.grouped_array import (
    GroupedArray as BaseGroupedArray,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from leaderboards.vendored.statsforecast.models import SeasonalNaive


class GroupedArray(BaseGroupedArray):
    """Series stored end to end in `data`, series i up to `indptr[i + 1]`."""

    def __init__(self, data: np.ndarray, indptr: np.ndarray) -> None:
        """Hold the series of `data` that `indptr` delimits."""
        self.data = _ensure_float(data)
        self.indptr = indptr
        self.n_groups = len(indptr) - 1

    def _get_cols(
        self,
        models: Sequence[SeasonalNaive],
        attr: str,
        level: Sequence[int] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        n_models = len(models)
        cuts = np.full(n_models + 1, fill_value=0, dtype=np.int32)
        has_level_models = np.full(n_models, fill_value=False, dtype=bool)
        cuts[0] = 0
        for i_model, model in enumerate(models):
            len_cols = 1  # mean
            has_level = (
                "level" in inspect.signature(getattr(model, attr)).parameters
                and len(level) > 0
            )
            has_level_models[i_model] = has_level
            if has_level:
                len_cols += 2 * len(level)  # levels
            cuts[i_model + 1] = len_cols + cuts[i_model]
        return cuts, has_level_models

    def _output_fcst(
        self,
        models: Sequence[SeasonalNaive],
        attr: str,
        h: int,
        level: Sequence[int] = (),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # returns empty output according to method
        cuts, has_level_models = self._get_cols(models=models, attr=attr, level=level)
        out = np.full(
            (self.n_groups * h, cuts[-1]), fill_value=np.nan, dtype=self.data.dtype
        )
        return out, cuts, has_level_models

    def forecast(
        self,
        models: Sequence[SeasonalNaive],
        h: int,
        fallback_model: SeasonalNaive | None = None,
        level: Sequence[int] = (),
    ) -> dict[str, np.ndarray | list[str]]:
        """
        Forecast `h` steps of each series with each of `models`.

        The forecasts are rows i * h to (i + 1) * h of "forecasts" for series
        i, a column per model and bound, named in "cols": the model's name for
        its mean, "<name>-lo-<level>" and "<name>-hi-<level>" for its bounds.
        A model that fails on a series is replaced by `fallback_model`.
        """
        fcsts, cuts, has_level_models = self._output_fcst(
            models=models, attr="forecast", h=h, level=level
        )
        matches = ["mean", "lo", "hi"]
        for i, grp in enumerate(self):
            y_train = grp[:, 0] if grp.ndim == 2 else grp  # noqa: PLR2004 - a 2-D series has its target in column 0.
            cols = []
            for i_model, model in enumerate(models):
                has_level = has_level_models[i_model]
                kwargs = {}
                if has_level:
                    kwargs["level"] = level
                try:
                    res_i = model.forecast(h=h, y=y_train, **kwargs)
                except Exception:
                    if fallback_model is not None:
                        res_i = fallback_model.forecast(h=h, y=y_train, **kwargs)
                    else:
                        raise
                cols_m = [
                    key for key in res_i if any(key.startswith(m) for m in matches)
                ]
                fcsts_i = np.vstack([res_i[key] for key in cols_m]).T
                cols_m = [
                    f"{model!r}" if col == "mean" else f"{model!r}-{col}"
                    for col in cols_m
                ]
                if fcsts_i.ndim == 1:
                    fcsts_i = fcsts_i[:, None]
                fcsts[i * h : (i + 1) * h, cuts[i_model] : cuts[i_model + 1]] = fcsts_i
                cols += cols_m
        return {"forecasts": fcsts, "cols": cols}
