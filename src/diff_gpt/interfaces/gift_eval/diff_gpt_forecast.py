"""The command line of diff_gpt.application.gift_eval.diff_gpt_forecast."""

import argparse
from pathlib import Path

import torch

from diff_gpt.application.gift_eval import diff_gpt_forecast
from diff_gpt.application.gift_eval.diff_gpt_forecast import TERMS, forecast_split
from diff_gpt.application.gift_eval.files import RUNS
from diff_gpt.application.ports import UsageError
from diff_gpt.infrastructure import file_lock, provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=diff_gpt_forecast.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.bin", help="a file of --run")
    parser.add_argument("--split", choices=("test", "val"), default="val")
    parser.add_argument("--model", help="the forecasts' folder; diff_gpt-<run name>")
    parser.add_argument("--terms", nargs="+", choices=TERMS, default=list(TERMS))
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["*"],
        help="fnmatch patterns of the file keys, such as 'm4_*'",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--min-context",
        type=int,
        help="the context rows of a pass of a long horizon; block_size / 2",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="decode rows")
    parser.add_argument("--windows-per-call", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--windows", type=Path, default=RUNS / "windows")
    parser.add_argument("--forecasts", type=Path, default=RUNS / "forecasts")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Forecast the selected windows of a split; ValueError on other settings."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        forecast_split(args, git_head=provenance.git_head, lock=file_lock.locked)
    except UsageError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
