"""The command line of diff_gpt.application.grokking.series_numeric_codec."""

from diff_gpt.application.grokking.series_numeric_codec import register
from diff_gpt.interfaces.grokking import series


def main() -> None:
    """Use the unchanged shared training/resume/verifier after registering codec."""
    register()
    series.main()


if __name__ == "__main__":
    main()
