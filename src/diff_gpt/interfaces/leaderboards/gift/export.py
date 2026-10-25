"""The command line of diff_gpt.infrastructure.leaderboards.gift.export."""

import argparse
from pathlib import Path

from diff_gpt.infrastructure.leaderboards.gift import export
from diff_gpt.infrastructure.leaderboards.gift.configs import (
    DATA,
    GIFT_REPO,
    RUNS,
    SPLITS,
)
from diff_gpt.infrastructure.leaderboards.gift.export import MAX_CONTEXT, export_windows


def main() -> None:
    """Export the windows of the chosen configurations and splits."""
    parser = argparse.ArgumentParser(
        description=export.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    parser.add_argument("--only", nargs="*", help="keys (default: all 97)")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--gift-repo", type=Path, default=GIFT_REPO)
    parser.add_argument("--out", type=Path, default=RUNS / "windows")
    args = parser.parse_args()
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    export_windows(args)


if __name__ == "__main__":
    main()
