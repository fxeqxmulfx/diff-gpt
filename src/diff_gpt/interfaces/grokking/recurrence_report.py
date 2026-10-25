"""The command line of diff_gpt.application.grokking.recurrence_report."""

import argparse
from pathlib import Path

from diff_gpt.application.grokking import recurrence_report
from diff_gpt.application.grokking.recurrence_report import report
from diff_gpt.infrastructure.grokking import browser_replay


def main() -> None:
    """Record current progress, optional browser replay and scientific plots."""
    parser = argparse.ArgumentParser(description=recurrence_report.__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--confirmed-only", action="store_true")
    parser.add_argument("--browser-module", type=Path)
    args = parser.parse_args()
    report(args, node=browser_replay.run_node)


if __name__ == "__main__":
    main()
