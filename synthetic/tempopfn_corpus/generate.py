"""
Generate the series [start, stop) of one source of one split, deterministically.

Usage, from synthetic/, with PYTHONHASHSEED fixed (see sources.python_hash_seed):

    PYTHONHASHSEED=0 uv run python -m tempopfn_corpus.generate train kernel \
        --stop 4096 --file-size 256
    PYTHONHASHSEED=0 uv run python -m tempopfn_corpus.generate train kernel \
        --start 4096 --stop 8192 --file-size 256
    PYTHONHASHSEED=0 uv run python -m tempopfn_corpus.generate \
        train augmented_per_sample_2048 --stop 4096 --file-size 256

A base source runs TempoPFN's ContinuousGenerator with the run_id of
`sources.run_id` in place of time.time_ns() and series_in_run = start, into a
staging folder, and moves its Arrow files into place under their indices once
all of them are written and their series ids checked. SIGTERM kills it at
once, leaving the staging folder for the next run of these files to remove.
Series j gets the seed run_id + j in whichever process makes it, so
processes can share a source over disjoint ranges of files, and a later run
can extend it.

An augmented source runs TempoPFN's offline augmentation script on the base
sources of the split, which must all be complete, with --global-seed run_id;
Python's `random` gets that seed as well, since CyclicalBatchDataset
shuffles the files it reads with it. Its output depends on every base file,
so it is regenerated whole, with --replace, after the base sources grow.

Python refuses to open any file of TempoPFN's data/ folder (nan_stats.json,
from the GIFT-Eval train splits) or of the repository's datasets/ outside
--root while this runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pyarrow import feather
from src.synthetic_generation.augmentations import (
    offline_per_sample_iid_augmentations,
    offline_temp_batch_augmentations,
)
from src.synthetic_generation.continuous_generation import (
    ContinuousGenerator,
    GeneratorWrapper,
    TimeSeriesDatasetManager,
)

from src import TEMPOPFN_ROOT
from tempopfn_corpus import sources

if TYPE_CHECKING:
    from types import ModuleType

AUGMENTATIONS: dict[str, ModuleType] = {
    "augmented_per_sample_2048": offline_per_sample_iid_augmentations,
    "augmented_temp_batch_2048": offline_temp_batch_augmentations,
}

# GeneratorWrapper makes cauker_univariate from 6-channel CauKer series, series
# id base_seed * 1000 + channel, from ceil(64 / 6) seeds per chunk of 64.
CAUKER_CHANNELS = 6
CAUKER_ID_SCALE = 1000


class StoppableGenerator(ContinuousGenerator):
    """TempoPFN's ContinuousGenerator, which SIGTERM stops at once."""

    def _setup_signal_handlers(self) -> None:
        """
        Keep the default actions of SIGINT and SIGTERM.

        TempoPFN's handlers save the partial batch, which the next run of these
        files discards with its staging folder, and exit only once the C call
        in progress returns.
        """


class LoudGeneratorWrapper(GeneratorWrapper):
    """TempoPFN's GeneratorWrapper, raising where it would return fewer series."""

    def generate_batch(
        self, batch_size: int, start_seed: int
    ) -> list[dict[str, object]]:
        """Generate the `batch_size` series of the seeds from `start_seed` on."""
        batch = super().generate_batch(batch_size, start_seed)
        if len(batch) != batch_size:
            msg = (
                f"{self.generator_type} made {len(batch)} of {batch_size} series from "
                f"seed {start_seed}: TempoPFN's wrapper logged the error above"
            )
            raise RuntimeError(msg)
        return batch


def series_ids(source: str, run_id: int, first: int, count: int) -> np.ndarray:
    """Return the ids TempoPFN gives the series [first, first + count) of a run."""
    index = np.arange(first, first + count)
    if source != "cauker_univariate":
        return run_id + index
    chunk_start, position = index - index % sources.CHUNK, index % sources.CHUNK
    base_seed = run_id + chunk_start + position // CAUKER_CHANNELS
    return base_seed * CAUKER_ID_SCALE + position % CAUKER_CHANNELS


def check_batch(path: Path, file_size: int, ids: np.ndarray | None) -> None:
    """Check that `path` holds `file_size` univariate series of LENGTH, with `ids`."""
    table = feather.read_table(path, columns=["series_id", "length", "num_channels"])
    column = {name: table[name].to_numpy() for name in table.column_names}
    if (
        table.num_rows != file_size
        or (column["length"] != sources.LENGTH).any()
        or (column["num_channels"] != 1).any()
        or (ids is not None and not np.array_equal(column["series_id"], ids))
    ):
        msg = (
            f"{path}: not {file_size} univariate series of {sources.LENGTH}, ids {ids}"
        )
        raise RuntimeError(msg)


def generate_base(
    root: Path,
    split: str,
    source: str,
    series: range,
    file_size: int,
) -> dict[str, object]:
    """Write the `series` of a base source with ContinuousGenerator; describe it."""
    directory = sources.source_dir(root, split, source)
    first_file, files = series.start // file_size, len(series) // file_size
    targets = [sources.batch_path(directory, first_file + k) for k in range(files)]
    taken = [path.name for path in targets if path.exists()]
    sizes = {
        json.loads(path.read_text())["file_size"]
        for path in directory.glob("record_*.json")
    }
    if taken or sizes - {file_size}:
        msg = f"{directory}: {taken} exist, or its file size {sizes} is not {file_size}"
        raise FileExistsError(msg)
    staging = directory / f".staging_{first_file:08d}"
    if staging.exists():  # what a killed run of these files left behind
        shutil.rmtree(staging)

    run_id = sources.run_id(split, source)
    generator = StoppableGenerator(
        LoudGeneratorWrapper(source, length=sources.LENGTH, global_seed=run_id),
        TimeSeriesDatasetManager(str(staging), batch_size=file_size),
        batch_size=file_size,
        run_id=run_id,
    )
    generator.series_in_run = series.start
    began, cpu_began = time.time(), time.process_time()
    generator.run_continuous(num_batches_to_generate=files)
    seconds, cpu_seconds = time.time() - began, time.process_time() - cpu_began

    written = sorted(staging.glob("batch_*.arrow"))
    if (len(written), generator.series_in_run) != (files, series.stop):
        msg = f"{staging}: {len(written)} files up to series {generator.series_in_run}"
        raise RuntimeError(msg)
    for k, path in enumerate(written):
        ids = series_ids(source, run_id, series.start + k * file_size, file_size)
        check_batch(path, file_size, ids)
    for path, target in zip(written, targets, strict=True):
        path.rename(target)
    staging.rmdir()
    return {
        "first_file": first_file,
        "files": files,
        "seconds": round(seconds, 3),
        "cpu_seconds": round(cpu_seconds, 3),
    }


def generate_augmented(
    root: Path,
    split: str,
    source: str,
    series: int,
    file_size: int,
    *,
    replace: bool,
) -> dict[str, object]:
    """Write `series` series of an augmented source with TempoPFN; describe the run."""
    base_series = {
        base: sources.recorded_series(root, split, base)
        for base in sources.BASE_SOURCES
    }
    for base, count in base_series.items():
        if not count or sources.staged(root, split, base):
            msg = f"{split}/{base} has no runs, or one in progress, or one killed"
            raise RuntimeError(msg)
    target = sources.source_dir(root, split, source)
    if target.exists() and not replace:
        msg = f"{target} exists: pass --replace to regenerate it"
        raise FileExistsError(msg)
    staging = root / "arrow" / split / f".staging_{source}"
    if staging.exists():
        shutil.rmtree(staging)

    run_id = sources.run_id(split, source)
    module = AUGMENTATIONS[source]
    argv = [
        *("--base-data-dir", str(root / "arrow" / split), "--output-dir", str(staging)),
        *("--length", str(sources.LENGTH), "--chunk-size", str(file_size)),
        *("--num-batches", str(series // file_size), "--global-seed", str(run_id)),
    ]
    random.seed(run_id)
    sys.argv = [module.__name__, *argv]
    began, cpu_began = time.time(), time.process_time()
    try:
        module.main()
    except SystemExit as failure:
        if failure.code:
            msg = f"{module.__name__} failed: see the error it logged above"
            raise RuntimeError(msg) from failure
    seconds, cpu_seconds = time.time() - began, time.process_time() - cpu_began

    output = staging / source  # the scripts' folder: augmented_<kind>_<length>
    written = sorted(output.glob("batch_*.arrow"))
    if written != [sources.batch_path(output, k) for k in range(series // file_size)]:
        msg = f"{output} holds {len(written)} Arrow files, not {series // file_size}"
        raise RuntimeError(msg)
    for path in written:
        check_batch(path, file_size, ids=None)
    if target.exists():
        target.rename(staging / "replaced")
    output.rename(target)
    shutil.rmtree(staging)
    return {
        "first_file": 0,
        "files": len(written),
        "seconds": round(seconds, 3),
        "cpu_seconds": round(cpu_seconds, 3),
        "base_series": base_series,
        "argv": argv,
    }


def refuse_real_data(root: Path) -> None:
    """Make Python refuse to open TempoPFN's data/ or datasets/ outside `root`."""
    allowed = root.resolve()
    refused = ((TEMPOPFN_ROOT / "data").resolve(), sources.REPO_ROOT / "datasets")

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not isinstance(args[0], str | bytes | os.PathLike):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(allowed) and any(map(path.is_relative_to, refused)):
            msg = f"refusing to open {path}: the corpus reads no real data"
            raise PermissionError(msg)

    sys.addaudithook(hook)


def main() -> None:
    """Generate one range of series of one source of one split."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("split", choices=sources.SPLITS)
    parser.add_argument("source", choices=sources.SOURCES)
    parser.add_argument("--start", type=int, default=0, help="first series")
    parser.add_argument("--stop", type=int, required=True, help="last series + 1")
    parser.add_argument("--file-size", type=int, default=1024, help="series per file")
    parser.add_argument("--root", type=Path, default=sources.DEFAULT_ROOT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="augmented sources: replace the old output once the new one is made",
    )
    args = parser.parse_args()
    augmented = args.source in sources.AUGMENTED_SOURCES
    if (
        args.file_size % sources.CHUNK
        or args.start % args.file_size
        or args.stop % args.file_size
        or not 0 <= args.start < args.stop <= sources.SEEDS_PER_SOURCE
        or (augmented and args.start)
    ):
        parser.error(
            "need 0 <= --start < --stop <= 2**24 in multiples of --file-size, "
            f"itself one of {sources.CHUNK}; --start 0 for augmented sources",
        )
    environment = sources.environment()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    refuse_real_data(args.root)

    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    if augmented:
        run = generate_augmented(
            args.root,
            args.split,
            args.source,
            args.stop,
            args.file_size,
            replace=args.replace,
        )
    else:
        series = range(args.start, args.stop)
        run = generate_base(args.root, args.split, args.source, series, args.file_size)
    record = {
        "split": args.split,
        "source": args.source,
        "run_id": sources.run_id(args.split, args.source),
        "file_size": args.file_size,
        "length": sources.LENGTH,
        "series": [args.start, args.stop],
        **run,
        "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "started": started,
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "environment": environment,
    }
    directory = sources.source_dir(args.root, args.split, args.source)
    path = sources.write_record(directory, record)
    print(
        f"{args.split}/{args.source} {args.start}-{args.stop}: {run['seconds']:.0f} s"
    )
    print(f"recorded in {path}")


if __name__ == "__main__":
    main()
