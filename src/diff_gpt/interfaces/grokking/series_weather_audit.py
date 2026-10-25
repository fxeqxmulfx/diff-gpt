"""The command line of diff_gpt.application.grokking.series_weather_audit."""

import argparse
from pathlib import Path

from diff_gpt.application.grokking import series_weather_audit
from diff_gpt.application.grokking.series_weather_audit import audit_runs


def main() -> None:
    """Audit every completed multivariate weather run in an output directory."""
    parser = argparse.ArgumentParser(description=series_weather_audit.__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit_runs(args.output)


if __name__ == "__main__":
    main()
