"""Replay the frozen coordinate-repair cohort without training."""

import argparse
import json
from pathlib import Path

from diff_gpt.application.grokking.residual_conditioning_replay import replay


def main() -> None:
    """Export all original outcomes with independent saved-state evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
