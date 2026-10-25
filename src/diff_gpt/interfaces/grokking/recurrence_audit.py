"""The command line of diff_gpt.application.grokking.recurrence_audit."""

import argparse
from pathlib import Path

import torch

from diff_gpt.application.grokking import recurrence as experiment
from diff_gpt.application.grokking import recurrence_audit
from diff_gpt.application.grokking.recurrence_audit import audit_report
from diff_gpt.infrastructure.grokking import browser_replay


def main() -> None:
    """Audit confirmed runs with bounded GPU memory and actual public forecasts."""
    parser = argparse.ArgumentParser(description=recurrence_audit.__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--evaluation-batch", type=int, default=128)
    parser.add_argument("--browser-module", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    experiment.EVALUATION_BATCH = args.evaluation_batch
    audit_report(args, node=browser_replay.run_node)


if __name__ == "__main__":
    main()
