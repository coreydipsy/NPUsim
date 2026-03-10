"""Root conftest for neusim test suite."""

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests (regression, etc.)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    run_slow = config.getoption("runslow") or os.environ.get("NEUSIM_RUN_SLOW_TESTS")
    if not run_slow:
        skip_slow = pytest.mark.skip(
            reason="Need --runslow option or NEUSIM_RUN_SLOW_TESTS=1 env var to run"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
