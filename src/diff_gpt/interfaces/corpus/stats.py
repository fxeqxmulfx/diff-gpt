"""The command line of diff_gpt.infrastructure.corpus.stats."""

import argparse
from pathlib import Path

from diff_gpt.infrastructure.corpus import sources, stats
from diff_gpt.infrastructure.corpus.stats import write_stats


def main() -> None:
    """Print and save the statistics of the corpus under --root."""
    parser = argparse.ArgumentParser(description=stats.__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=sources.DEFAULT_ROOT)
    args = parser.parse_args()
    write_stats(args.root)


if __name__ == "__main__":
    main()
