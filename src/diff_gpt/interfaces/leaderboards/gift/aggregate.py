"""The command line of diff_gpt.infrastructure.leaderboards.gift.aggregate."""

import argparse
from pathlib import Path

from diff_gpt.infrastructure.leaderboards.gift import aggregate
from diff_gpt.infrastructure.leaderboards.gift.aggregate import print_aggregates
from diff_gpt.infrastructure.leaderboards.gift.configs import SPLITS


def main() -> None:
    """Print the leaderboard's scores of each results CSV."""
    parser = argparse.ArgumentParser(
        description=aggregate.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("results", type=Path, nargs="+", help="all_results.csv")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--baseline", type=Path, help="default: by --split")
    args = parser.parse_args()
    print_aggregates(args)


if __name__ == "__main__":
    main()
