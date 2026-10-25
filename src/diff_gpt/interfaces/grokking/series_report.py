"""The command line of diff_gpt.application.grokking.series_report."""

import argparse
import json
from pathlib import Path

from diff_gpt.application.grokking import series_report
from diff_gpt.application.grokking.div97 import ROOT
from diff_gpt.application.grokking.series_report import report


def main() -> None:
    """Summarize completed stages while another process trains later stages."""
    parser = argparse.ArgumentParser(description=series_report.__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=ROOT / "runs/grokking-series/fast-first-current",
    )
    parser.add_argument("--include", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    value = report(args.output, include=args.include)
    print(
        json.dumps(
            {
                "completed_runs": len(value["runs"]),
                "three_seed_groups": sum(
                    group["confirmed_on_three_seeds"] for group in value["groups"]
                ),
                "stages_complete": value["stages_complete"],
                "summary": str(args.output / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
