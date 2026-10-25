"""The command line of diff_gpt.infrastructure.corpus.generate."""

import argparse
import logging
from pathlib import Path

from diff_gpt.infrastructure.corpus import generate, sources


def main() -> None:
    """Generate one range of series of one source of one split."""
    parser = argparse.ArgumentParser(description=generate.__doc__.split("\n\n")[0])
    parser.add_argument("split", choices=sources.SPLITS)
    parser.add_argument("source", choices=sources.SOURCES)
    parser.add_argument("--start", type=int, default=0, help="first series")
    parser.add_argument("--stop", type=int, required=True, help="last series + 1")
    parser.add_argument("--file-size", type=int, default=1024, help="series per file")
    parser.add_argument("--root", type=Path, default=sources.DEFAULT_ROOT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="augmented sources: replace the old output once the new one is made",
    )
    args = parser.parse_args()
    augmented = args.source in sources.AUGMENTED_SOURCES
    if (
        args.file_size % sources.CHUNK
        or args.start % args.file_size
        or args.stop % args.file_size
        or not 0 <= args.start < args.stop <= sources.SEEDS_PER_SOURCE
        or (augmented and args.start)
    ):
        parser.error(
            "need 0 <= --start < --stop <= 2**24 in multiples of --file-size, "
            f"itself one of {sources.CHUNK}; --start 0 for augmented sources",
        )
    environment = sources.environment()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    generate.refuse_real_data(args.root)
    generate.generate(args, environment)


if __name__ == "__main__":
    main()
