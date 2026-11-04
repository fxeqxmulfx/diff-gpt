"""Free-running forecasts from the actual public GPT and train-fitted codec."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from diff_gpt.application.diff_gpt import DiffGPT
from diff_gpt.application.grokking.series_data import _raw_series
from diff_gpt.application.grokking.series_rollout import HORIZONS
from diff_gpt.domain.encoder_decoder import TokenClippingWarning, encode
from diff_gpt.domain.sampler.temperature import TemperatureSampler

if TYPE_CHECKING:
    from research.current_grokking.adapters import SeriesTask
    from research.current_grokking.train import Logits


def errors(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Score actual generated points, without teacher-forced future columns."""
    delta = predicted - truth
    return {"mse": float(np.mean(delta**2)), "mae": float(np.mean(np.abs(delta)))}


@torch.no_grad()
def evaluate(task: SeriesTask, model: Logits, maximum: int = 8) -> dict:
    """Use the historical eligible-origin rule and equal-origin naive controls."""
    config = task.config
    if config.dataset not in HORIZONS or config.shuffled_time:
        return {
            "applicable": False,
            "reason": "No chronological real-dataset rollout for this case",
        }
    raw, columns, _, period = _raw_series(config)
    horizon = HORIZONS[config.dataset]
    train = task.dataset.splits["train"]
    features = len(columns)
    seen = [
        set(train.x[:, index::features].ravel()) | set(train.y[train.column == index])
        for index in range(features)
    ]
    ids = task.dataset.splits["test"].ids
    fits = task.dataset.metadata["fit"]
    rng = np.random.default_rng(config.seed ^ 0xA0FE)
    selected = []
    excluded = {"horizon_past_end": 0, "unseen_training_token": 0}
    for index in rng.permutation(len(ids)):
        series_id, first, origin = map(int, ids[index])
        values = raw[series_id]
        if origin + horizon > len(values):
            excluded["horizon_past_end"] += 1
            continue
        fit = fits[series_id]
        mean, std, domain = (np.asarray(fit[key]) for key in ("mean", "std", "domain"))
        normalized = (values[first : origin + horizon] - mean) / std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TokenClippingWarning)
            _, _, tokens = encode(
                normalized,
                config.vocab_size,
                domain,
                config.order,
                tokenizer=config.tokenizer,
            )
        if any(
            not set(tokens[:, feature]) <= seen[feature] for feature in range(features)
        ):
            excluded["unseen_training_token"] += 1
            continue
        selected.append(
            {
                "series": series_id,
                "first": first,
                "origin": origin,
                "mean": mean,
                "std": std,
                "domain": domain,
                "context": values[first:origin].copy(),
                "truth": values[origin : origin + horizon].copy(),
            }
        )
        if len(selected) >= maximum:
            break
    result = {
        "applicable": True,
        "horizon_rows": horizon,
        "origins": len(selected),
        "teacher_forced_future": False,
        "excluded_before_limit": excluded,
        "population": (
            "Chronological test origins with complete horizon and "
            "training-vocabulary coverage, matching the historical "
            "eligibility rule."
        ),
    }
    if not selected:
        return result
    truth = np.stack([row["truth"] for row in selected])
    means = np.stack([row["mean"] for row in selected])[:, None]
    stds = np.stack([row["std"] for row in selected])[:, None]
    predictions = np.empty_like(truth)
    groups = {}
    for index, row in enumerate(selected):
        groups.setdefault(row["series"], []).append(index)
    for indices in groups.values():
        first = selected[indices[0]]
        wrapper = DiffGPT(
            model.model, config.order, first["domain"], tokenizer=config.tokenizer
        )
        frames = [
            pd.DataFrame(
                (selected[index]["context"] - selected[index]["mean"])
                / selected[index]["std"],
                columns=columns,
            )
            for index in indices
        ]
        context_rows = len(frames[0])
        for step in range(horizon):
            generated = wrapper.predict_many(
                frames, 1, sampler=TemperatureSampler(0.0), batch_size=maximum
            )
            for index, frame in zip(indices, generated, strict=True):
                predictions[index, step] = (
                    frame.to_numpy()[-1] * selected[index]["std"]
                    + selected[index]["mean"]
                )
            frames = [
                frame.iloc[-context_rows:].reset_index(drop=True) for frame in generated
            ]
    persistence = np.stack(
        [np.repeat(row["context"][-1:], horizon, axis=0) for row in selected]
    )
    seasonal = np.stack(
        [
            raw[row["series"]][row["origin"] - period + np.arange(horizon) % period]
            for row in selected
        ]
    )
    result["scores"] = {
        name: {
            "raw": errors(value, truth),
            "train_normalized": errors((value - means) / stds, (truth - means) / stds),
        }
        for name, value in (
            ("current_gpt", predictions),
            ("persistence", persistence),
            ("seasonal_naive", seasonal),
        )
    }
    result["selected_origins"] = [
        {key: row[key] for key in ("series", "first", "origin")} for row in selected
    ]
    return result
