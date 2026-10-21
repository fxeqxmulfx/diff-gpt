"""
The splits and sources of the corpus, the seeds of its runs and their records.

The sources are the 16 of TempoPFN's configs/example.yaml, in its order: the
14 base generators of continuous_generation.py, then its two offline
augmentations of their output. TempoPFN's ContinuousGenerator starts each
chunk of 64 series at the seed run_id + series_in_run, and its wrappers seed
series i of a chunk with that seed + i, so series j of a run gets the seed
run_id + j. A run_id of split_index * 2**28 + source_index * 2**24 gives each
(split, source) its own 2**24 seeds, all below 2**32 as np.random.seed
requires; the augmentations take theirs as --global-seed.

A run writes the series [start, stop) of a (split, source) as Arrow files of
`file_size` series, arrow/<split>/<source>/batch_<k>.arrow with k from
start / file_size on, and records itself in record_<k>.json next to them.
"""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from src import TEMPOPFN_ROOT

TEMPOPFN_COMMIT = "5969ec6ddbded5a8d2976d42504c1ae436a2f1b3"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "datasets" / "tempopfn-synthetic"
STAND_INS = Path(__file__).resolve().parents[1] / "src"

LENGTH = 2048
SPLITS = ("train", "holdout")
BASE_SOURCES = (
    "forecast_pfn",
    "gp",
    "kernel",
    "sawtooth",
    "sinewave",
    "step",
    "anomaly",
    "spike",
    "cauker_univariate",
    "ou_process",
    "audio_financial_volatility",
    "audio_multi_scale_fractal",
    "audio_network_topology",
    "audio_stochastic_rhythm",
)
AUGMENTED_SOURCES = ("augmented_per_sample_2048", "augmented_temp_batch_2048")
SOURCES = BASE_SOURCES + AUGMENTED_SOURCES

CHUNK = 64  # the series ContinuousGenerator asks a wrapper for at a time
SEEDS_PER_SOURCE = 2**24
SEEDS_PER_SPLIT = 2**28
_HASH_SEED_BOUND = 2**32

LIBRARIES = (
    "gpytorch",
    "networkx",
    "numpy",
    "pandas",
    "pyarrow",
    "pyo",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "torch",
)


def run_id(split: str, source: str) -> int:
    """Return the seed of series 0 of `source` in `split`; series j gets run_id + j."""
    return (
        SPLITS.index(split) * SEEDS_PER_SPLIT + SOURCES.index(source) * SEEDS_PER_SOURCE
    )


def python_hash_seed() -> int:
    """
    Return the PYTHONHASHSEED this interpreter started with, if it is fixed.

    TempoPFN's wrappers salt the seed of their parameter generator with
    hash(class name), which differs between interpreters unless it is.
    """
    value = os.environ.get("PYTHONHASHSEED", "")
    if not value.isdecimal() or int(value) >= _HASH_SEED_BOUND:
        msg = (
            f"PYTHONHASHSEED={value!r}: set it to an integer in [0, 2**32) before "
            "Python starts, e.g. PYTHONHASHSEED=0, since TempoPFN's wrappers salt "
            "their parameter seeds with hash(class name)"
        )
        raise SystemExit(msg)
    return int(value)


def tempopfn_commit() -> str:
    """Return the commit of the TempoPFN clone, if it is TEMPOPFN_COMMIT and clean."""
    git = shutil.which("git") or "/usr/bin/git"
    head, status = (
        subprocess.check_output(  # noqa: S603 - fixed git argv, no shell.
            [git, *argv],
            cwd=TEMPOPFN_ROOT,
            text=True,
        ).strip()
        for argv in (["rev-parse", "HEAD"], ["status", "--porcelain"])
    )
    if head != TEMPOPFN_COMMIT or status:
        msg = f"{TEMPOPFN_ROOT}: {head}, changes {status!r}; expected {TEMPOPFN_COMMIT}"
        raise SystemExit(msg)
    return head


def training_proportions() -> dict[str, float]:
    """Read the generator_proportions of TempoPFN's configs/example.yaml."""
    path = TEMPOPFN_ROOT / "configs" / "example.yaml"
    proportions = yaml.safe_load(path.read_text())["generator_proportions"]
    if tuple(proportions) != SOURCES:
        msg = f"{path} has the sources {tuple(proportions)}, expected {SOURCES}"
        raise ValueError(msg)
    return {source: float(weight) for source, weight in proportions.items()}


def stand_ins() -> dict[str, str]:
    """Return the SHA-256 of each module of `src` here: the stand-ins and loaders."""
    return {
        path.relative_to(STAND_INS).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(STAND_INS.rglob("*.py"))
    }


def environment() -> dict[str, object]:
    """Return what the output of a run depends on besides its arguments."""
    return {
        "tempopfn_commit": tempopfn_commit(),
        "stand_ins": stand_ins(),
        "python_hash_seed": python_hash_seed(),
        "python": sys.version.split()[0],
        "libraries": {name: importlib.metadata.version(name) for name in LIBRARIES},
    }


def source_dir(root: Path, split: str, source: str) -> Path:
    """Return the folder of the Arrow files of `source` in `split`."""
    return root / "arrow" / split / source


def recorded_series(root: Path, split: str, source: str) -> int:
    """Return the series of the recorded runs of a source, 0 if it has none."""
    directory = source_dir(root, split, source)
    if not any(directory.glob("record_*.json")):
        return 0
    records = read_records(directory)
    return sum(record["files"] for record in records) * records[0]["file_size"]


def staged(root: Path, split: str, source: str) -> bool:
    """Return whether a run of `source` in `split` is in progress, or was killed."""
    directory = source_dir(root, split, source)
    return (
        any(directory.glob(".staging_*"))
        or (directory.parent / f".staging_{source}").exists()
    )


def batch_path(directory: Path, index: int) -> Path:
    """Return the path of Arrow file `index` of a source, in TempoPFN's naming."""
    return directory / f"batch_{index:08d}.arrow"


def write_record(directory: Path, record: dict[str, object]) -> Path:
    """Write the record of a run whose first Arrow file is record["first_file"]."""
    path = directory / f"record_{record['first_file']:08d}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def read_records(directory: Path) -> list[dict[str, object]]:
    """
    Read the records of the runs of a source, which must tile its Arrow files.

    The runs must cover the files 0, 1, ... without gaps or overlaps, with one
    run_id, file size and environment, and the folder must hold exactly those
    files.
    """
    paths = sorted(directory.glob("record_*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    if not records:
        msg = f"no runs recorded in {directory}"
        raise FileNotFoundError(msg)
    files = 0
    for record in records:
        if record["first_file"] != files:
            msg = f"{directory}: a run starts at file {record['first_file']} of {files}"
            raise ValueError(msg)
        files += record["files"]
    for key in ("split", "source", "run_id", "file_size", "length", "environment"):
        values = {json.dumps(record[key]) for record in records}
        if len(values) > 1:
            msg = f"{directory}: the runs differ in {key}: {sorted(values)}"
            raise ValueError(msg)
    expected = [batch_path(directory, index) for index in range(files)]
    if sorted(directory.glob("batch_*.arrow")) != expected:
        msg = f"{directory} does not hold exactly the {files} recorded Arrow files"
        raise ValueError(msg)
    return records
