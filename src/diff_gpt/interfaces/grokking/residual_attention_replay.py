"""Replay the frozen residual-attention probe and export portable evidence."""

import argparse
import json
from pathlib import Path

from diff_gpt.application.grokking.residual_attention_replay import replay


def main() -> None:
    """Read existing checkpoints only; do not start a new experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
