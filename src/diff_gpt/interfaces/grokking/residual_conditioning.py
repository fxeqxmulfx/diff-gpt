"""Run the separately frozen residual-query conditioning repair cohort."""

import argparse
from pathlib import Path

from diff_gpt.application.grokking.residual_conditioning import run


def main() -> None:
    """Keep all learning settings in the committed protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
