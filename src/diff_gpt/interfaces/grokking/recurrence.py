"""The command line of diff_gpt.application.grokking.recurrence."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from diff_gpt.application.grokking import recurrence
from diff_gpt.application.grokking.div97 import ROOT
from diff_gpt.application.grokking.recurrence import Config, run_plan, verify


def main() -> None:
    """Run a fixed-optimizer plan, extend a checkpoint, or verify saved weights."""
    parser = argparse.ArgumentParser(description=recurrence.__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/grokking-div97-rows"
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--threads", type=int, default=2)
    for field, default in asdict(Config()).items():
        parser.add_argument(
            "--" + field.replace("_", "-"), type=type(default), default=default
        )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if args.verify:
        print(json.dumps(verify(args.verify, args.device, full=args.full), indent=2))
        return
    configs = (
        [Config(**item) for item in json.loads(args.plan.read_text())]
        if args.plan
        else [Config(**{key: getattr(args, key) for key in asdict(Config())})]
    )
    run_plan(
        configs, args.output, device=args.device, resume=args.resume, full=args.full
    )


if __name__ == "__main__":
    main()
