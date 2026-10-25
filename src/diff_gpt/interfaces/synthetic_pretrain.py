"""The command line of diff_gpt.application.synthetic_pretrain."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from diff_gpt.application import synthetic_pretrain
from diff_gpt.application.paths import ROOT
from diff_gpt.application.ports import UsageError
from diff_gpt.application.synthetic_pretrain import (
    RESUME_OPTIONS,
    pretrain,
    resumed,
)
from diff_gpt.domain.normalization import NORMALIZATIONS
from diff_gpt.infrastructure import provenance


def _given(parser: argparse.ArgumentParser, argv: list[str] | None) -> set[str]:
    """Return the destinations of the options that `argv` sets."""
    unset = object()
    probe = argparse.Namespace(**dict.fromkeys(vars(parser.parse_args(argv)), unset))
    # argparse sets the default of an option only where the namespace has none.
    parser.parse_args(argv, probe)
    return {key for key, value in vars(probe).items() if value is not unset}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=synthetic_pretrain.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "datasets/tempopfn-synthetic/npy",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=json.loads,
        default={},
        help="JSON map of source to weight, over TempoPFN's proportions",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--tokenizer", choices=["lgt", "mash", "mash-half"])
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATIONS,
        default="mean_abs_step",
    )
    parser.add_argument("--domain-quantile", type=float, default=0.999)
    parser.add_argument("--domain-windows", type=int, default=4096)
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument("--min-context", type=int, default=16)
    parser.add_argument("--missing-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument(
        "--train-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=["auto", "on", "off"],
        default="auto",
        help="recompute each block's activations in the backward pass instead "
        "of keeping them; auto: on a GPU, not on a CPU",
    )
    parser.add_argument("--score-every", type=int, default=1)
    parser.add_argument("--score-windows", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--score-samples",
        type=int,
        nargs="*",
        default=[],
        help="also score forecasts of each of these numbers of samples, at most "
        "--num-samples, from disjoint groups of the ones drawn",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-batch", type=int, default=256)
    parser.add_argument("--score-seed", type=int, default=0)
    parser.add_argument(
        "--score-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--keep-every",
        type=float,
        metavar="FLOPS",
        help="also keep the weights of the first evaluation at or past each "
        "multiple of FLOPS training FLOPs, as step_<updates>_flops_<FLOPs>.bin",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="go on from the state.pt of --output, with the settings of its "
        "config.json, to --max-iters (by default the saved one)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Pretrain one model as the arguments say and record its curve."""
    parser = _parser()
    args = parser.parse_args(argv)
    config: dict[str, Any] = {}
    saved: dict[str, Any] | None = None
    try:
        if args.resume:
            given = _given(parser, argv)
            other = sorted(given - RESUME_OPTIONS)
            if other:
                named = ", ".join("--" + name.replace("_", "-") for name in other)
                parser.error(f"--resume takes the settings of config.json, not {named}")
            args, config, saved = resumed(args, given)
        if any(not 1 <= size <= args.num_samples for size in args.score_samples):
            parser.error(f"--score-samples must lie in [1, {args.num_samples}]")
        if args.keep_every is not None and not args.keep_every > 0:
            parser.error("--keep-every must be above 0")
        pretrain(args, config, saved, git_head=provenance.git_head)
    except UsageError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
