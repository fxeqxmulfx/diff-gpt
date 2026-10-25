"""The command line of diff_gpt.application.grokking.recurrence_period_probe."""

import argparse
from pathlib import Path

import torch

from diff_gpt.application.grokking import recurrence_period_probe
from diff_gpt.application.grokking.recurrence_period_probe import probe


def main() -> None:
    """Run reproducible controls without altering a checkpoint or sampler."""
    parser = argparse.ArgumentParser(description=recurrence_period_probe.__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(probe(args.directory, args.device, args.samples), flush=True)


if __name__ == "__main__":
    main()
