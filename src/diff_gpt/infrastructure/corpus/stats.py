r"""
Value ranges and duplicates of the exported corpus, to plan its normalization.

Usage:

    uv run python -m diff_gpt.interfaces.corpus.stats \
        [--root datasets/tempopfn-synthetic]

Per (split, source) of npy/: the minimum, median and maximum of |x| and of
|dx| (dx_t = x_t - x_{t-1}) over all values; the median of |dx_t| / |x_{t-1}|
over the steps from a value other than 0; the median over series of
mean |dx| / mean |x|; the share of constant series. Then the series equal,
byte for byte, to another of their split, and to one of another split.
Prints a table and writes stats.json under --root.
"""

import hashlib
import json
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from diff_gpt.infrastructure.corpus import sources

if TYPE_CHECKING:
    from pathlib import Path


def value_ranges(series: np.ndarray) -> dict[str, object]:
    """Summarize the magnitudes of the values and steps of the (N, L) `series`."""
    magnitude = np.abs(series)
    step = np.abs(np.diff(series, axis=1))
    previous = magnitude[:, :-1]
    relative = step[previous > 0] / previous[previous > 0]
    mean_magnitude = magnitude.mean(axis=1, dtype=np.float64)
    mean_step = step.mean(axis=1, dtype=np.float64)
    level = mean_magnitude > 0
    return {
        "abs_x": [float(f(magnitude)) for f in (np.min, np.median, np.max)],
        "abs_dx": [float(f(step)) for f in (np.min, np.median, np.max)],
        "median_abs_dx_over_abs_previous_x": float(np.median(relative)),
        "median_mean_abs_dx_over_mean_abs_x": float(
            np.median(mean_step[level] / mean_magnitude[level]),
        ),
        "constant_share": float((mean_step == 0).mean()),
    }


def digests(series: np.ndarray) -> list[bytes]:
    """Hash each series by its bytes."""
    return [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in series]


def write_stats(root: Path) -> None:
    """Print and save the statistics of the corpus under `root`."""
    ranges: dict[str, dict[str, object]] = {}
    split_digests: dict[str, Counter[bytes]] = {}
    print(
        "| split | source | min |x| | median |x| | max |x| | min |dx| | median |dx| "
        "| max |dx| | median |dx_t|/|x_t-1| | median mean|dx|/mean|x| | constant |",
    )
    print("|" + " --- |" * 11)
    for split in sources.SPLITS:
        counter: Counter[bytes] = Counter()
        for source in sources.SOURCES:
            path = root / "npy" / split / f"{source}.npy"
            if not path.exists():
                continue
            series = np.load(path)
            entry = value_ranges(series)
            ranges.setdefault(split, {})[source] = entry
            counter.update(digests(series))
            numbers = [
                *entry["abs_x"],
                *entry["abs_dx"],
                entry["median_abs_dx_over_abs_previous_x"],
                entry["median_mean_abs_dx_over_mean_abs_x"],
            ]
            cells = [f"{value:.3g}" for value in numbers]
            print(
                f"| {split} | {source} | {' | '.join(cells)} "
                f"| {entry['constant_share']:.2%} |",
            )
        split_digests[split] = counter
    duplicates = {
        "within_split": {
            split: sum(n for n in counter.values() if n > 1)
            for split, counter in split_digests.items()
        },
        "across_splits": sum(
            min(counter[key] for counter in split_digests.values())
            for key in set.intersection(*(set(c) for c in split_digests.values()))
        )
        if split_digests
        else 0,
    }
    print(f"series equal to another: {json.dumps(duplicates)}")
    stats = {"value_ranges": ranges, "duplicates": duplicates}
    (root / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
