"""The command line of diff_gpt.infrastructure.leaderboards.time.aggregate."""

import argparse
from pathlib import Path

from diff_gpt.infrastructure.leaderboards.time import aggregate
from diff_gpt.infrastructure.leaderboards.time.aggregate import print_aggregates
from diff_gpt.infrastructure.leaderboards.time.configs import SPLITS


def main() -> None:
    """Print the leaderboard's scores of each results folder."""
    parser = argparse.ArgumentParser(
        description=aggregate.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("results", type=Path, nargs="+", help="results folders")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--baseline", type=Path, help="default: by --split")
    args = parser.parse_args()
    print_aggregates(args)


if __name__ == "__main__":
    main()
