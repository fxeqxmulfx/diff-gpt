"""The command line of diff_gpt.application.synthetic_rescore."""

import argparse
import os
from pathlib import Path

import torch

from diff_gpt.application import synthetic_rescore
from diff_gpt.application.ports import UsageError
from diff_gpt.application.synthetic_rescore import rescore_run
from diff_gpt.infrastructure import provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=synthetic_rescore.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", type=Path, required=True, help="the run's --output")
    parser.add_argument("--checkpoint", default="last.bin", help="a file of --run")
    parser.add_argument("--output", type=Path, required=True, help="a new JSON file")
    parser.add_argument("--per-source", type=int, default=64)
    parser.add_argument("--window-seed", type=int, default=1)
    parser.add_argument(
        "--context",
        type=int,
        help="context rows; by default the rest of the block after --horizon",
    )
    parser.add_argument("--horizon", type=int, help="by default the run's --horizon")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--score-samples",
        type=int,
        nargs="*",
        default=[],
        help="also score forecasts of each of these numbers of samples, at most "
        "--num-samples, from disjoint groups of the ones drawn",
    )
    parser.add_argument("--temperature", type=float, help="by default the run's")
    parser.add_argument("--batch-size", type=int, default=1000, help="decode rows")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Rescore the checkpoint as the arguments say and write the JSON record."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists")
    environment = {
        "affinity": sorted(os.sched_getaffinity(0)),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        rescore_run(args, git_head=provenance.git_head, environment=environment)
    except UsageError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
