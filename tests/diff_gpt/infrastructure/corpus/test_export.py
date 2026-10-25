"""The export of TempoPFN's Arrow batches to the .npy arrays of the corpus."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from diff_gpt.infrastructure.corpus.tempopfn.synthetic_generation.continuous_generation import (  # noqa: E501 - the module's name alone is wider than the line.
    TimeSeriesDatasetManager,
)

from diff_gpt.infrastructure.corpus import export, sources

if TYPE_CHECKING:
    from pathlib import Path

LENGTH = 8


def write_source(root: Path, split: str, source: str, files: list[np.ndarray]) -> None:
    """Write each of `files` as an Arrow batch of TempoPFN's, and their record."""
    directory = sources.source_dir(root, split, source)
    manager = TimeSeriesDatasetManager(str(directory), batch_size=len(files[0]))
    for rows in files:
        manager.append_batch(
            [
                {
                    "series_id": index,
                    "values": [row.tolist()],
                    "length": LENGTH,
                    "num_channels": 1,
                    "generator_type": source,
                    "start": pd.Timestamp("2020-01-01"),
                    "frequency": "D",
                    "generation_timestamp": pd.Timestamp.now(),
                }
                for index, row in enumerate(rows)
            ],
        )
    sources.write_record(
        directory,
        {
            "split": split,
            "source": source,
            "run_id": sources.run_id(split, source),
            "file_size": len(files[0]),
            "length": LENGTH,
            "series": [0, sum(map(len, files))],
            "first_file": 0,
            "files": len(files),
            "seconds": 1.5,
            "cpu_seconds": 1.4,
            "max_rss_mib": 300,
            "environment": {"tempopfn_commit": "c", "python_hash_seed": 0},
        },
    )


def test_export_keeps_the_finite_series_in_order(tmp_path: Path) -> None:
    """Rows that overflow float32 or are not finite are dropped and counted."""
    rows = np.linspace(-3.0, 3.0, 6 * LENGTH).reshape(6, LENGTH)
    rows[1, 2] = np.nan
    rows[2, 5] = 1e39  # finite in float64 only
    rows[3, 0] = -np.inf
    write_source(tmp_path, "holdout", "kernel", [rows[:3], rows[3:]])

    manifest = export.export(tmp_path)

    series = np.load(tmp_path / "npy" / "holdout" / "kernel.npy", mmap_mode="r")
    assert series.dtype == np.float32
    assert series.flags.c_contiguous
    np.testing.assert_array_equal(series, rows[[0, 4, 5]].astype(np.float32))
    entry = manifest["splits"]["holdout"]["kernel"]
    assert (
        entry["series"],
        entry["generated"],
        entry["nonfinite_float64"],
        entry["nonfinite_float32"],
        entry["run_id"],
        entry["generation_seconds"],
        entry["cpu_seconds"],
        entry["max_rss_mib"],
    ) == (3, 6, 2, 3, 2**28 + 2 * 2**24, 1.5, 1.4, 300)
    assert manifest["python_hash_seed"] == 0
    assert manifest["training_proportions"]["augmented_temp_batch_2048"] == 2.0  # noqa: PLR2004
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
