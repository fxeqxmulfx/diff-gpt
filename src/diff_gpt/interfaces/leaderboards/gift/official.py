"""The command line of diff_gpt.infrastructure.leaderboards.gift.official."""

import argparse
from pathlib import Path

from diff_gpt.infrastructure.leaderboards.gift import official
from diff_gpt.infrastructure.leaderboards.gift.configs import DATA, GIFT_REPO, SPLITS
from diff_gpt.infrastructure.leaderboards.gift.evaluation import BATCH_SIZE, RESULTS
from diff_gpt.infrastructure.leaderboards.gift.official import score_configurations


def main() -> None:
    """Score the seasonal naive on the chosen configurations."""
    parser = argparse.ArgumentParser(
        description=official.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--model", help="default: statsforecast_seasonal_naive[_<max length>]"
    )
    parser.add_argument("--only", nargs="*", help="keys (default: all 97)")
    parser.add_argument("--jobs", type=int, default=1, help="processes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--gift-repo", type=Path, default=GIFT_REPO)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    score_configurations(args)


if __name__ == "__main__":
    main()
