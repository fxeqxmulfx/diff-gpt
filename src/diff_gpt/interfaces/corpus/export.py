"""The command line of diff_gpt.infrastructure.corpus.export."""

import argparse
from pathlib import Path

from diff_gpt.infrastructure.corpus import export, sources


def main() -> None:
    """Export the corpus under --root."""
    parser = argparse.ArgumentParser(description=export.__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=sources.DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = export.export(args.root)
    for split, entries in manifest["splits"].items():
        for source, entry in entries.items():
            dropped = entry["nonfinite_float32"], entry["nonfinite_float64"]
            print(
                f"{split}/{source}: {entry['series']} series, dropped "
                f"{dropped[0]} ({dropped[1]} of them in float64)",
            )


if __name__ == "__main__":
    main()
