"""Run the frozen known-solution residual-attention diagnostic cohort."""

import argparse
from pathlib import Path

from diff_gpt.application.grokking.residual_attention_probe import run


def main() -> None:
    """Select only the output directory; the declared cases are fixed in code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
