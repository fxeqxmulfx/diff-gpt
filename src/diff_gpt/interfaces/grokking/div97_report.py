"""The command line of diff_gpt.application.grokking.div97_report."""

import argparse
from pathlib import Path

from diff_gpt.application.grokking import div97_report
from diff_gpt.application.grokking.div97_report import report
from diff_gpt.infrastructure.grokking import browser_replay


def main() -> None:
    """Write a compact JSON summary, replay check and optional plots."""
    parser = argparse.ArgumentParser(description=div97_report.__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-module", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report(args, node=browser_replay.run_node)


if __name__ == "__main__":
    main()
