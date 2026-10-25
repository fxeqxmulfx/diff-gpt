"""The command line of diff_gpt.infrastructure.leaderboards.time.official."""

import argparse
from pathlib import Path

from diff_gpt.infrastructure.leaderboards.time import official
from diff_gpt.infrastructure.leaderboards.time.configs import DATA, SPLITS
from diff_gpt.infrastructure.leaderboards.time.official import SAMPLES, score_tasks
from diff_gpt.infrastructure.leaderboards.time.score import RESULTS


def main() -> None:
    """Score the seasonal naive on the chosen tasks."""
    parser = argparse.ArgumentParser(
        description=official.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--model",
        help="default: statsforecast_seasonal_naive[_exact][_<max length>]",
    )
    parser.add_argument("--only", nargs="*", help="keys (default: all 98)")
    parser.add_argument("--jobs", type=int, default=1, help="processes")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be 0 or more")
    score_tasks(args)


if __name__ == "__main__":
    main()
