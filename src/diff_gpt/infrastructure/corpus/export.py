"""
Export the Arrow batches of the corpus to float32 .npy arrays, with a manifest.

Usage, from synthetic/:

    uv run python -m tempopfn_corpus.export [--root ../datasets/tempopfn-synthetic]

For each split and each source with recorded runs, npy/<split>/<source>.npy
gets its series in the order of their files: float32 of shape (N, LENGTH),
C-contiguous, for np.load(mmap_mode="r"). Series with a value that is not
finite, in float64 or once cast to float32, are dropped and counted.
manifest.json records the TempoPFN commit and the PYTHONHASHSEED of the
runs, and per (split, source) the run_id (the --global-seed of an augmented
source), the series, the dropped series, the generation and CPU seconds summed
over its runs and their largest peak RSS, next to the training proportions of
TempoPFN's configs/example.yaml.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from pyarrow import feather

from tempopfn_corpus import sources


def read_values(path: Path, length: int) -> np.ndarray:
    """Read the float64 (series, length) values of a batch of univariate series."""
    column = feather.read_table(path, columns=["values"])["values"].combine_chunks()
    channels = column.flatten()
    values = channels.flatten()
    if (
        column.null_count + channels.null_count + values.null_count
        or (pc.list_value_length(column).to_numpy() != 1).any()
        or (pc.list_value_length(channels).to_numpy() != length).any()
    ):
        msg = (
            f"{path} holds series that are null, multivariate or not of length {length}"
        )
        raise ValueError(msg)
    return values.to_numpy().reshape(len(column), length)


def export_source(directory: Path, output: Path) -> dict[str, object]:
    """Write the finite series of a source to `output`; return its manifest entry."""
    records = sources.read_records(directory)
    length = records[0]["length"]
    kept, generated, nonfinite_float64, nonfinite_float32 = [], 0, 0, 0
    for index in range(sum(record["files"] for record in records)):
        values = read_values(sources.batch_path(directory, index), length)
        with np.errstate(over="ignore"):
            values32 = values.astype(np.float32)
        finite32 = np.isfinite(values32).all(axis=1)
        generated += len(values)
        nonfinite_float64 += int((~np.isfinite(values).all(axis=1)).sum())
        nonfinite_float32 += int((~finite32).sum())
        kept.append(values32[finite32])
    series = np.ascontiguousarray(np.concatenate(kept))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("wb") as file:
        np.save(file, series)
    partial.replace(output)
    entry = {
        "run_id": records[0]["run_id"],
        "series": len(series),
        "generated": generated,
        "nonfinite_float64": nonfinite_float64,
        "nonfinite_float32": nonfinite_float32,
        "generation_seconds": round(sum(record["seconds"] for record in records), 1),
        "cpu_seconds": round(sum(record["cpu_seconds"] for record in records), 1),
        "max_rss_mib": max(record["max_rss_mib"] for record in records),
        "file_size": records[0]["file_size"],
        "runs": len(records),
    }
    if "base_series" in records[0]:
        entry["base_series"] = records[0]["base_series"]
    return entry


def export(root: Path) -> dict[str, object]:
    """Export every recorded (split, source) under `root`; return the manifest."""
    splits: dict[str, dict[str, object]] = {}
    environments = set()
    for split in sources.SPLITS:
        for source in sources.SOURCES:
            directory = sources.source_dir(root, split, source)
            if not directory.exists():
                continue
            if sources.staged(root, split, source):
                msg = f"{directory} has a run in progress, or one that was killed"
                raise RuntimeError(msg)
            output = root / "npy" / split / f"{source}.npy"
            entry = export_source(directory, output)
            entry["npy"] = str(output.relative_to(root))
            splits.setdefault(split, {})[source] = entry
            record = json.loads(next(directory.glob("record_*.json")).read_text())
            environments.add(json.dumps(record["environment"], sort_keys=True))
    if len(environments) != 1:
        msg = f"the runs under {root} differ in their environments: {environments}"
        raise ValueError(msg)
    environment = json.loads(environments.pop())
    manifest = {
        "tempopfn": {
            "repository": "https://github.com/automl/TempoPFN",
            "commit": environment.pop("tempopfn_commit"),
            "license": "Apache-2.0",
        },
        "python_hash_seed": environment.pop("python_hash_seed"),
        "environment": environment,
        "length": sources.LENGTH,
        "dtype": "float32",
        "seeds": (
            "series j of a base source has the seed run_id + j, with run_id = "
            "split_index * 2**28 + source_index * 2**24 in the orders of `splits` "
            "and `training_proportions`; an augmented source is TempoPFN's script "
            "run with --global-seed run_id on the base_series of its split"
        ),
        "dropped": (
            "series not finite in float64 (nonfinite_float64) or once cast to "
            "float32 (nonfinite_float32, which includes the former)"
        ),
        "training_proportions": sources.training_proportions(),
        "splits": splits,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Export the corpus under --root."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=sources.DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = export(args.root)
    for split, entries in manifest["splits"].items():
        for source, entry in entries.items():
            dropped = entry["nonfinite_float32"], entry["nonfinite_float64"]
            print(
                f"{split}/{source}: {entry['series']} series, dropped "
                f"{dropped[0]} ({dropped[1]} of them in float64)",
            )


if __name__ == "__main__":
    main()
