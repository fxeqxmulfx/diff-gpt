"""The command line of diff_gpt.infrastructure.corpus.schedule."""

import argparse
import os
import signal
import sys
from pathlib import Path

from diff_gpt.infrastructure.corpus import schedule, sources
from diff_gpt.infrastructure.corpus.schedule import base_jobs, run, stop


def main() -> None:
    """Run the jobs that bring every split to its --series."""
    parser = argparse.ArgumentParser(description=schedule.__doc__.split("\n\n")[0])
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
