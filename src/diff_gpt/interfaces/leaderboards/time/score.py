"""The command line of diff_gpt.infrastructure.leaderboards.time.score."""

import argparse
from pathlib import Path

from diff_gpt.application.ports import UsageError
from diff_gpt.infrastructure.leaderboards.time import score
from diff_gpt.infrastructure.leaderboards.time.configs import DATA, RUNS, SPLITS
from diff_gpt.infrastructure.leaderboards.time.score import RESULTS, score_forecasts


def main() -> None:
    """Score a model's forecasts of the chosen tasks."""
    parser = argparse.ArgumentParser(
        description=score.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--only", nargs="*", help="keys (default: all 98)")
    parser.add_argument("--jobs", type=int, default=1, help="processes")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--forecasts", type=Path, default=RUNS / "forecasts")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    try:
        score_forecasts(args)
    except UsageError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
