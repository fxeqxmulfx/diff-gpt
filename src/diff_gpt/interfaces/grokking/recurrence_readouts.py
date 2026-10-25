"""The command line of diff_gpt.application.grokking.recurrence_readouts."""

import argparse
from pathlib import Path

import torch

from diff_gpt.application.grokking import recurrence_readouts
from diff_gpt.application.grokking.recurrence_readouts import persist_readouts


def main() -> None:
    """Persist frozen-weight comparisons without interrupting training."""
    parser = argparse.ArgumentParser(description=recurrence_readouts.__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--verify-readout")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    persist_readouts(args)


if __name__ == "__main__":
    main()
