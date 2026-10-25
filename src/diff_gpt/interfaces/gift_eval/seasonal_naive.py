"""The command line of diff_gpt.application.gift_eval.seasonal_naive."""

import argparse
from pathlib import Path

from diff_gpt.application.gift_eval import seasonal_naive as use_case
from diff_gpt.application.gift_eval.files import RUNS
from diff_gpt.application.gift_eval.seasonal_naive import forecast_split
from diff_gpt.application.ports import UsageError


def main() -> None:
    """Forecast every exported window of a split."""
    parser = argparse.ArgumentParser(
        description=use_case.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--model", default="seasonal_naive")
    parser.add_argument("--windows", type=Path, default=RUNS / "windows")
    parser.add_argument("--forecasts", type=Path, default=RUNS / "forecasts")
    args = parser.parse_args()
    try:
        forecast_split(args)
    except UsageError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
