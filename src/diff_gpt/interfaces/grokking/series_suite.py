"""The command line of diff_gpt.application.grokking.series_suite."""

import argparse
from pathlib import Path

import torch

from diff_gpt.application.grokking import series_suite
from diff_gpt.application.grokking.div97 import ROOT
from diff_gpt.application.grokking.series_suite import run_stages


def main() -> None:
    """Run requested stages sequentially on the single available GPU."""
    parser = argparse.ArgumentParser(description=series_suite.__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/grokking-series/fast-first-current"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "formula",
            "formula_refined",
            "noise",
            "temporal",
            "temporal_late",
            "synthetic",
            "gaussian",
            "forecasting",
        ),
        default=[
            "formula_refined",
            "synthetic",
            "gaussian",
            "noise",
            "temporal",
            "forecasting",
            "temporal_late",
        ],
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_stages(args)


if __name__ == "__main__":
    main()
