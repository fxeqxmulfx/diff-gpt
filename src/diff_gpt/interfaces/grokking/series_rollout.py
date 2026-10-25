"""The command line of diff_gpt.application.grokking.series_rollout."""

import argparse
from pathlib import Path

import torch

from diff_gpt.application.grokking import series_rollout
from diff_gpt.application.grokking.div97 import ROOT
from diff_gpt.application.grokking.series_rollout import rollouts


def main() -> None:
    """Score completed forecast families after GPU training has finished."""
    parser = argparse.ArgumentParser(description=series_rollout.__doc__)
    parser.add_argument(
        "output", type=Path, nargs="?", default=ROOT / "runs/grokking-series/fast-first"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--maximum", type=int, default=8)
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rollouts(args)


if __name__ == "__main__":
    main()
