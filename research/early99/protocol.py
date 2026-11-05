"""Frozen task tiers and exact sample-exposure accounting."""

import json
from pathlib import Path

from research.current_grokking.protocol import Case

MANIFEST = Path(__file__).with_name("suite.json")
SEEDS = (0, 1, 2)
TARGET = 0.99
SHORT_SECONDS = 60
LONG_SECONDS = 300


def suite() -> dict:
    """Load the task selection without training or modifying old results."""
    return json.loads(MANIFEST.read_text())


def selected(tier: str = "all") -> list[dict]:
    """Return the disjoint short/long tiers, or their union."""
    if tier not in {"short", "long", "all"}:
        message = f"Unknown tier: {tier}"
        raise ValueError(message)
    return [row for row in suite()["tasks"] if tier == "all" or row["tier"] == tier]


def specification(row: dict) -> Case:
    """Instantiate the recorded benchmark settings, including strict 99% exactness."""
    return Case(**row["case"])


def exposure(updates: int, train_examples: int, batch: int) -> dict:
    """Count actual presented examples with nonwrapping, short-tail epochs."""
    if updates < 0 or train_examples <= 0 or batch <= 0:
        message = (
            "Updates must be nonnegative; training size and batch must be positive"
        )
        raise ValueError(message)
    batches = (train_examples + batch - 1) // batch
    full_epochs, partial_updates = divmod(updates, batches)
    examples = full_epochs * train_examples + min(
        partial_updates * batch, train_examples
    )
    return {
        "updates": updates,
        "train_examples": train_examples,
        "batch": batch,
        "updates_per_epoch": batches,
        "examples_seen": examples,
        "epochs": examples / train_examples,
        "complete_epochs": examples // train_examples,
    }


def first99(history: list[dict], train_examples: int, batch: int) -> dict | None:
    """Accept the first exact held-out crossing, even when a later score drops."""
    previous = None
    for row in history:
        accuracy = row["validation"]["sequence_accuracy"]
        if accuracy >= TARGET:
            counts = exposure(row["step"], train_examples, batch)
            return {
                **counts,
                "accuracy": accuracy,
                "recorded_seconds": row["seconds"],
                "previous_observed_updates": previous,
                "previous_observed_epochs": exposure(previous, train_examples, batch)[
                    "epochs"
                ]
                if previous is not None
                else None,
            }
        previous = row["step"]
    return None
