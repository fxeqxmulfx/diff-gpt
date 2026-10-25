"""
Training-only single-operand controls for independent weather equations.

Separate per-column normalization can make an answer nearly identical to one
input token. A high unigram-relative skill then need not require both operands.
Report literal token copying and a smoothed conditional table fitted on one
training operand at a time, without using any heldout labels in that fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.grokking.div97 import write_json
from benchmarks.grokking.series_data import array_hash

PRIOR_COUNT = 0.5
PRIOR_STRENGTHS = (0.01, 0.1, 1.0, 10.0, 32.0, 100.0)


def _conditional_table(counts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Choose shrinkage to the training marginal by training leave-one-out CE."""
    marginal = counts.sum(0)
    total = counts.sum()
    prior = (marginal + PRIOR_COUNT) / (total + PRIOR_COUNT * len(marginal))
    loo_prior = (marginal - 1 + PRIOR_COUNT) / (total - 1 + PRIOR_COUNT * len(marginal))
    observed = counts > 0
    losses = []
    for strength in PRIOR_STRENGTHS:
        numerator = counts - 1 + strength * loo_prior[None, :]
        denominator = counts.sum(1, keepdims=True) - 1 + strength
        probability = np.maximum(
            numerator[observed] / np.broadcast_to(denominator, counts.shape)[observed],
            1e-300,
        )
        losses.append(float(-(counts[observed] * np.log(probability)).sum() / total))
    index = int(np.argmin(losses))
    strength = PRIOR_STRENGTHS[index]
    table = (counts + strength * prior[None, :]) / (
        counts.sum(1, keepdims=True) + strength
    )
    return table, strength, losses[index]


def _metrics(
    tokens: np.ndarray,
    probability: np.ndarray | None,
    target: np.ndarray,
    truth: np.ndarray,
    base: np.ndarray,
    step: np.ndarray,
    vocab: int,
) -> dict[str, float | None]:
    expected = tokens if probability is None else probability @ np.arange(vocab)
    prediction = base + step * (expected - vocab // 2)
    error = prediction - truth
    return {
        "loss": float(-np.log(probability[np.arange(len(target)), target]).mean())
        if probability is not None
        else None,
        "accuracy": float((tokens == target).mean()),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
    }


def formula_shortcuts(directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Fit each input-only baseline on the frozen optimized training sample."""
    config = metadata["config"]["data"]
    if not config["dataset"].startswith("weather_formula"):
        return {}
    path = directory / "frozen_splits.npz"
    provenance = {
        "frozen_splits_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = directory / "shortcut-baselines.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["provenance"] == provenance:
            return previous
    vocab = config["vocab_size"]
    result = {
        "provenance": provenance,
        "fit_population": "optimized training examples only",
        "marginal_prior_count_per_output": PRIOR_COUNT,
        "conditional_prior_strength_grid": list(PRIOR_STRENGTHS),
        "prior_strength_selection": "training-only leave-one-out cross-entropy",
        "operands": {},
    }
    with np.load(path, allow_pickle=False) as frozen:
        names = [
            name for name in ("train", "validation", "test") if f"{name}__x" in frozen
        ]
        inputs = {name: frozen[f"{name}__x"].copy() for name in names}
        if config["dataset"] == "weather_formula_bits":
            bits = metadata["data"]["bits_per_input"]
            inputs = {
                name: values.reshape(len(values), 2, bits)
                @ (2 ** np.arange(bits - 1, -1, -1))
                for name, values in inputs.items()
            }
        for position, operand in enumerate(metadata["data"]["input_columns"]):
            counts = np.zeros((vocab, vocab), dtype=np.float64)
            np.add.at(
                counts, (inputs["train"][:, position], frozen["train__y"][:, -1]), 1
            )
            table, strength, loo_loss = _conditional_table(counts)
            entry = {
                "training_probability_sha256": array_hash(table),
                "selected_prior_strength": strength,
                "leave_one_out_training_loss": loo_loss,
                "native_token_copy": {},
                "training_fitted_conditional": {},
            }
            for name in names:
                copied = inputs[name][:, position]
                probability = table[copied]
                arguments = (
                    frozen[f"{name}__y"][:, -1],
                    frozen[f"{name}__truth"][:, 0],
                    frozen[f"{name}__base"][:, 0],
                    frozen[f"{name}__step"][:, 0],
                    vocab,
                )
                entry["native_token_copy"][name] = _metrics(copied, None, *arguments)
                entry["training_fitted_conditional"][name] = _metrics(
                    probability.argmax(1), probability, *arguments
                )
            result["operands"][operand] = entry
    write_json(destination, result)
    return result
