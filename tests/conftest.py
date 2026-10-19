"""
Shared pytest configuration: tests marked `slow` run only with `--runslow`.

Slow tests (long Decimal series, full training runs) are skipped by default,
so the default run stays quick enough to repeat after every change.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the `--runslow` option."""
    parser.addoption(
        "--runslow",
        action="store_true",
        help="also run the tests marked slow",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the `slow` marker."""
    config.addinivalue_line("markers", "slow: long test, runs only with --runslow")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip the tests marked `slow` unless `--runslow` is given."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if item.get_closest_marker("slow") is not None:
            item.add_marker(skip_slow)
