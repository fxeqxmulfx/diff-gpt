"""Full-population conditional references, evaluated on frozen input contexts."""

from __future__ import annotations

import numpy as np


def conditional_reference(
    table_contexts: np.ndarray,
    table_targets: np.ndarray,
    contexts: np.ndarray,
    targets: np.ndarray,
    vocab: int,
) -> dict[str, float]:
    """
    Use the entire table's conditional distribution, even for a one-row test.

    The entropy and majority probability are averaged over the evaluated input
    contexts. Realized oracle scores can differ due to finite label sampling.
    No model is fitted from this diagnostic table.
    """
    if not len(table_targets) or not len(targets):
        msg = "conditional reference requires nonempty table and evaluation"
        raise ValueError(msg)
    keys, groups = np.unique(table_contexts, axis=0, return_inverse=True)
    counts = np.zeros((len(keys), vocab), dtype=np.int64)
    np.add.at(counts, (groups, table_targets), 1)
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum(axis=1)
    lookup = {tuple(key): index for index, key in enumerate(keys)}
    try:
        evaluated = np.array([lookup[tuple(row)] for row in contexts], dtype=np.int64)
    except KeyError as error:
        msg = "evaluated context is absent from the full table"
        raise ValueError(msg) from error
    actual = probabilities[evaluated, targets]
    if np.any(actual <= 0):
        msg = "evaluated target is outside the full table's conditional support"
        raise ValueError(msg)
    return {
        "floor_loss": float(entropy[evaluated].mean()),
        "ceiling_accuracy": float(probabilities.max(axis=1)[evaluated].mean()),
        "oracle_realized_loss": float(-np.log(actual).mean()),
        "oracle_realized_accuracy": float(
            (probabilities.argmax(axis=1)[evaluated] == targets).mean()
        ),
    }
