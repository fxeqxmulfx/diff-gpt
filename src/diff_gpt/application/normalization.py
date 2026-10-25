"""
Forecasts in the units of each context, from a model trained on normalized windows.

`predict_samples_normalized` divides each context by its own
`window_scale`, has a `DiffGPT` forecast the normalized contexts and takes
the samples back to the context's units with `denormalize`; the methods are
those of `diff_gpt.domain.normalization`.
"""

from typing import TYPE_CHECKING

import pandas as pd

from diff_gpt.domain.normalization import denormalize, normalize, window_scale

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import torch

    from diff_gpt.application.diff_gpt import DiffGPT
    from diff_gpt.domain.sampler.sampler import Sampler


def predict_samples_normalized(
    model: DiffGPT,
    contexts: Sequence[np.ndarray],
    max_new_points: int,
    num_samples: int,
    method: str = "mean_abs_step",
    *,
    sampler: Sampler | None = None,
    batch_size: int = 256,
    seed: int = 42,
    dtype: torch.dtype | None = None,
) -> list[np.ndarray]:
    """
    Draw `num_samples` futures of each context, each in the context's units.

    A context is a float64 (T, F) array, NaN for a missing value, of any
    scale. It is divided by its own `window_scale` with `method`, the
    normalization the model was trained with, forecast by
    `DiffGPT.predict_samples_many` (`sampler`, `batch_size`, `seed` and
    `dtype` as there), and its samples come back in its units. Returns, in
    the order of `contexts`, arrays of shape (num_samples, max_new_points, F).
    """
    scales = [window_scale(context, method) for context in contexts]
    samples = model.predict_samples_many(
        [
            pd.DataFrame(normalize(context, scale))
            for context, scale in zip(contexts, scales, strict=True)
        ],
        max_new_points=max_new_points,
        num_samples=num_samples,
        sampler=sampler,
        batch_size=batch_size,
        seed=seed,
        dtype=dtype,
    )
    return [
        denormalize(drawn, scale) for drawn, scale in zip(samples, scales, strict=True)
    ]
