"""
Build or extend the corpus: every missing range of every source, in parallel.

Usage, from synthetic/, with PYTHONHASHSEED fixed and the CPUs it may use:

    PYTHONHASHSEED=0 taskset -c 4,6,8 uv run python -m tempopfn_corpus.schedule \
        --series train=4096 --series holdout=1024 --file-size 256

Each split's base sources grow to the given number of series in `generate`
processes over ranges of files, cut so that a range takes about
--part-seconds at the speeds of SECONDS_PER_SERIES or less, longest first.
At most --jobs run at a time, by default one per CPU of the scheduler's
affinity. A job starts only if the RSS of the scheduler and its jobs, each
job counted at RESERVED_RSS at least, stays within --max-rss-gib with it, and
MemAvailable stays above --min-available-gib; once the RSS goes over
--max-rss-gib, the scheduler kills its jobs and fails. Once all base sources
of a split are complete, its augmented sources are made from them, replacing
earlier ones made from fewer base series. Each process logs to
logs/<split>/<source>/<first file>.log under --root, and the kernel kills it
when the scheduler exits.
"""

import argparse
import ctypes
import functools
import itertools
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from tempopfn_corpus import sources

# Seconds per series of one single-threaded `generate` process at 1024 series
# per file (gp mostly at 256, as fast), 56 at a time on the 56 logical CPUs
# (28 cores) of two Xeon E5-2680 v4, as `taskset -c 0-55` runs them, 39 of the
# 56 gp: from 2048 to 40448 series per source. gp takes 1.53 s when all 56 are
# gp. Alone on the machine, as the augmented sources of train run at the end
# of a corpus, gp takes 0.58 s, augmented_temp_batch_2048 0.14 s and
# augmented_per_sample_2048 0.0075 s.
SECONDS_PER_SERIES = {
    "forecast_pfn": 0.053,
    "gp": 1.5,
    "kernel": 0.27,
    "sawtooth": 0.0048,
    "sinewave": 0.0062,
    "step": 0.018,
    "anomaly": 0.0056,
    "spike": 0.0063,
    "cauker_univariate": 0.15,
    "ou_process": 0.12,
    "audio_financial_volatility": 0.049,
    "audio_multi_scale_fractal": 0.048,
    "audio_network_topology": 0.070,
    "audio_stochastic_rhythm": 0.046,
    "augmented_per_sample_2048": 0.021,
    "augmented_temp_batch_2048": 0.41,
}
POLL_SECONDS = 2.0
GIB = 2**30
# In those runs the base sources peaked at 0.60-0.81 GiB of RSS and the
# augmented ones, two per split, at 2.0-2.2 GiB; the 56 processes held at most
# 35 GiB of summed VmRSS, and MemAvailable fell from 113 to 85 GiB of 121. A
# job counts at RESERVED_RSS until it holds more, so 56 start at once within
# the default --max-rss-gib of 80, more than twice those 35 GiB.
RESERVED_RSS = GIB
PR_SET_PDEATHSIG = 1  # <linux/prctl.h>


class Job(NamedTuple):
    """The series [start, stop) of a source of a split, one `generate` process."""

    split: str
    source: str
    start: int
    stop: int

    def __str__(self) -> str:
        """Name the job in the log."""
        return f"{self.split}/{self.source} {self.start}-{self.stop}"

    @property
    def seconds(self) -> float:
        """Estimate the duration of the job."""
        return (self.stop - self.start) * SECONDS_PER_SERIES[self.source]


def base_jobs(
    root: Path,
    split: str,
    series: int,
    file_size: int,
    part_seconds: float,
) -> list[Job]:
    """Cut the files that grow the base sources of `split` to `series` into jobs."""
    jobs = []
    for source in sources.BASE_SOURCES:
        first = sources.recorded_series(root, split, source) // file_size
        files = max(0, series // file_size - first)
        cost = files * file_size * SECONDS_PER_SERIES[source]
        parts = min(files, math.ceil(cost / part_seconds))
        bounds = [first + part * files // parts for part in range(parts + 1)]
        jobs += [
            Job(split, source, start * file_size, stop * file_size)
            for start, stop in itertools.pairwise(bounds)
        ]
    return jobs


def augmented_jobs(root: Path, split: str, series: int) -> list[Job]:
    """List the augmented sources of `split` missing or made from other base series."""
    base_series = {
        source: sources.recorded_series(root, split, source)
        for source in sources.BASE_SOURCES
    }
    jobs = []
    for source in sources.AUGMENTED_SOURCES:
        directory = sources.source_dir(root, split, source)
        records = sources.read_records(directory) if directory.exists() else []
        if not records or (records[0]["base_series"], records[0]["series"]) != (
            base_series,
            [0, series],
        ):
            jobs.append(Job(split, source, 0, series))
    return jobs


def rss(pid: int) -> int:
    """Return the resident set size of process `pid` in bytes, 0 once it exited."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return 0
    lines = [line for line in status.splitlines() if line.startswith("VmRSS:")]
    return int(lines[0].split()[1]) * 1024 if lines else 0


def available_memory() -> int:
    """Return the MemAvailable of /proc/meminfo in bytes."""
    lines = Path("/proc/meminfo").read_text().splitlines()
    return next(int(line.split()[1]) * 1024 for line in lines if "MemAvailable" in line)


def memory_slots(
    processes: list[subprocess.Popen[bytes]],
    max_rss: float,
    min_available: float,
) -> int:
    """
    Return how many more jobs the memory has room for, beside `processes`.

    Each job counts at RESERVED_RSS until it holds more. Fails, which kills the
    jobs, once the scheduler and `processes` hold more than `max_rss`.
    """
    own, sizes = rss(os.getpid()), [rss(process.pid) for process in processes]
    if own + sum(sizes) > max_rss:
        msg = (
            f"the scheduler and its jobs hold {(own + sum(sizes)) / GIB:.2f} GiB, "
            f"over --max-rss-gib {max_rss / GIB:g}: killing the jobs"
        )
        raise SystemExit(msg)
    reserved = own + sum(max(size, RESERVED_RSS) for size in sizes)
    room = min(max_rss - reserved, available_memory() - min_available)
    return int(room // RESERVED_RSS)


def die_with(parent: int) -> None:
    """In a child before exec: have the kernel SIGKILL it once `parent` exits."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL):
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent:  # the scheduler exited before the prctl
        os._exit(1)


def launch(job: Job, root: Path, file_size: int) -> subprocess.Popen[bytes]:
    """Start the `generate` process of `job`, logging to its file under `root`."""
    log = root / "logs" / job.split / job.source / f"{job.start // file_size:08d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        *(sys.executable, "-m", "tempopfn_corpus.generate", job.split, job.source),
        *("--start", str(job.start), "--stop", str(job.stop)),
        *("--file-size", str(file_size), "--root", str(root)),
        *(["--replace"] if job.source in sources.AUGMENTED_SOURCES else []),
    ]
    die = functools.partial(die_with, os.getpid())
    with log.open("wb") as output:
        return subprocess.Popen(  # noqa: S603 - our own argv, no shell.
            argv,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            preexec_fn=die,  # noqa: PLW1509 - the scheduler runs one thread.
        )


def stop(signum: int, _frame: object) -> None:
    """Exit on SIGHUP, SIGINT or SIGTERM, killing the jobs on the way out."""
    raise SystemExit(128 + signum)


def run(
    ready: list[Job],
    targets: dict[str, int],
    root: Path,
    args: argparse.Namespace,
) -> list[Job]:
    """Run the `ready` jobs, then the augmented ones of `targets`; return failures."""
    waiting = set(targets)  # splits whose augmented sources wait for their base sources
    running: dict[Job, tuple[subprocess.Popen[bytes], float]] = {}
    failed: list[Job] = []
    holding = False
    try:
        while ready or running or waiting:
            for split in sorted(waiting):
                if not any(job.split == split for job in (*ready, *running, *failed)):
                    waiting.discard(split)
                    ready += augmented_jobs(root, split, targets[split])
                elif any(job.split == split for job in failed):
                    waiting.discard(split)
            ready.sort(key=lambda job: job.seconds, reverse=True)
            free = args.jobs - len(running)
            slots = memory_slots(
                [process for process, _ in running.values()],
                args.max_rss_gib * GIB,
                args.min_available_gib * GIB,
            )
            if holding != (bool(ready) and slots < free):
                holding = not holding
                state = "holding jobs for" if holding else "resuming jobs with"
                print(f"{state} memory: {slots} slots of {free}", flush=True)
            for _ in range(min(free, slots, len(ready))):
                job = ready.pop(0)
                running[job] = (launch(job, root, args.file_size), time.time())
                print(f"started {job} (~{job.seconds:.0f} s)", flush=True)
            time.sleep(POLL_SECONDS)
            for job, (process, began) in list(running.items()):
                if process.poll() is None:
                    continue
                del running[job]
                status = "done"
                if process.returncode:
                    failed.append(job)
                    status = f"failed ({process.returncode})"
                print(f"{status} {job} in {time.time() - began:.0f} s", flush=True)
    finally:
        for process, _ in running.values():
            process.kill()
            process.wait()
    return failed


def main() -> None:
    """Run the jobs that bring every split to its --series."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="SPLIT=N",
        help="series per source of a split, e.g. train=4096; repeatable",
    )
    parser.add_argument("--file-size", type=int, default=1024)
    parser.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--part-seconds", type=float, default=1200.0)
    parser.add_argument("--max-rss-gib", type=float, default=80.0)
    parser.add_argument("--min-available-gib", type=float, default=8.0)
    parser.add_argument("--root", type=Path, default=sources.DEFAULT_ROOT)
    args = parser.parse_args()
    targets = {split: int(n) for split, n in (item.split("=") for item in args.series)}
    if not set(targets) <= set(sources.SPLITS):
        parser.error(f"splits must be among {sources.SPLITS}")
    sources.environment()  # refuses an unfixed PYTHONHASHSEED or a changed clone

    root = args.root.resolve()
    ready = [
        job
        for split, series in targets.items()
        for job in base_jobs(root, split, series, args.file_size, args.part_seconds)
    ]
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    failed = run(ready, targets, root, args)
    if failed:
        print(
            f"{len(failed)} jobs failed: see their logs under {root / 'logs'}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
