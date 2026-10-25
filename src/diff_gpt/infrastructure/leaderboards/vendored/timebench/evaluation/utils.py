"""
The terms of a TIME dataset that its configuration gives a horizon.

This is timebench c633e23's `src/timebench/evaluation/utils.py` (MIT, a copy of
the license in leaderboards/LICENSE-timebench), cut to `get_available_terms`.
Unchanged but for the style of leaderboards.vendored.
"""

from __future__ import annotations


def get_available_terms(dataset_name: str, config: dict) -> list[str]:
    """Get the terms that are actually defined in the config for a dataset."""
    datasets_config = config.get("datasets", {})
    if dataset_name not in datasets_config:
        return []
    dataset_config = datasets_config[dataset_name]
    available_terms = []
    for term in ["short", "medium", "long"]:
        if (
            term in dataset_config
            and dataset_config[term].get("prediction_length") is not None
        ):
            available_terms.append(term)  # noqa: PERF401 - upstream's loop.
    return available_terms
