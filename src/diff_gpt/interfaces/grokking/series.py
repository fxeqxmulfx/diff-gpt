"""The command line of diff_gpt.application.grokking.series."""

import argparse
import json
from pathlib import Path

import torch

from diff_gpt.application.grokking import series
from diff_gpt.application.grokking.div97 import ROOT
from diff_gpt.application.grokking.series import (
    Config,
    config_from_dict,
    run_plan,
    verify,
)


def main() -> None:
    """Run a frozen plan, prepare datasets, resume, or replay saved states."""
    parser = argparse.ArgumentParser(description=series.__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/grokking-series/experiments"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.verify:
        print(json.dumps(verify(args.verify, args.device), indent=2))
        return
    configs = (
        [config_from_dict(value) for value in json.loads(args.plan.read_text())]
        if args.plan
        else [Config()]
    )
    run_plan(
        configs,
        args.output,
        device=args.device,
        resume=args.resume,
        prepare=args.prepare,
    )


if __name__ == "__main__":
    main()
