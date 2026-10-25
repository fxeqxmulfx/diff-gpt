"""The command line of diff_gpt.infrastructure.leaderboards.time.compare."""

import argparse
from pathlib import Path

from diff_gpt.infrastructure.leaderboards.time import compare
from diff_gpt.infrastructure.leaderboards.time.compare import print_differences


def main() -> None:
    """Print the relative differences of two results folders."""
    parser = argparse.ArgumentParser(
        description=compare.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--only", nargs="*", help="keys (default: all 98)")
    args = parser.parse_args()
    print_differences(args)


if __name__ == "__main__":
    main()
