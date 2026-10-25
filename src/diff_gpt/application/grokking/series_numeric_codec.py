"""
Optional lossless binary prompts for the independent weather equations.

The ordinary codec still quantizes each scalar and the answer remains one
quantized-value token. Only the two input tokens become fixed-width binary
strings. This exposes numeric structure without changing resolution, labels,
splits, the model, or the optimizer. Run this module with the series CLI flags.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.grokking import series
from benchmarks.grokking.series_data import REGISTRY, token_coverage, weather_formula

if TYPE_CHECKING:
    from benchmarks.grokking.series_data import DataConfig, Dataset


def binary_prompts(tokens: np.ndarray, vocab: int) -> np.ndarray:
    """Encode each integer MSB first, preserving all original scalar levels."""
    bits = (vocab - 1).bit_length()
    shifts = np.arange(bits - 1, -1, -1)
    return ((tokens[..., None] >> shifts) & 1).reshape(len(tokens), -1)


def bits_dataset(config: DataConfig) -> Dataset:
    """Adapt the exact same equation populations and their conditional oracle."""
    data = weather_formula(config)
    for split in data.splits.values():
        x = binary_prompts(split.x, config.vocab_size)
        y, clean, column = (
            np.full_like(x, -100),
            np.full_like(x, -100),
            np.full_like(x, -1),
        )
        y[:, -1], clean[:, -1], column[:, -1] = split.y[:, -1], split.clean_y[:, -1], 0
        split.x, split.y, split.clean_y, split.column = x, y, clean, column
    data.metadata.update(
        {
            "input_representation": (
                "MSB-first bits of each original quantized input; "
                "answer remains one original value token"
            ),
            "bits_per_input": (config.vocab_size - 1).bit_length(),
            "quantization_changed": False,
            "split_assignment_changed": False,
            "hashes": data.fingerprint(),
            "token_coverage": token_coverage(data.splits),
        }
    )
    return data


def register() -> None:
    """Extend the public dataset registry and measured-source provenance."""
    if REGISTRY.get("weather_formula_bits") is bits_dataset:
        return
    REGISTRY["weather_formula_bits"] = bits_dataset
    previous = series.source_hashes
    relative = str(Path(__file__).resolve().relative_to(series.ROOT))

    def with_codec() -> dict[str, str]:
        return {
            **previous(),
            relative: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    series.source_hashes = with_codec


def main() -> None:
    """Use the unchanged shared training/resume/verifier after registering codec."""
    register()
    series.main()


if __name__ == "__main__":
    main()
