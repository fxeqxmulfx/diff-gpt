"""The command line of diff_gpt.application.grokking.recurrence_restart."""

import argparse
from pathlib import Path

from diff_gpt.application.grokking import recurrence_restart
from diff_gpt.application.grokking.recurrence_restart import restart


def main() -> None:
    """Continue a frozen parent with a deterministic second warmup, without reset."""
    parser = argparse.ArgumentParser(description=recurrence_restart.__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    restart(args)


if __name__ == "__main__":
    main()
