"""Pytest configuration for diagnostics tests.

This conftest provides fixtures that ensure proper isolation of the global
PluginRegistry state across tests, preventing race conditions during parallel
test execution.

The PluginRegistry is a class-level singleton with global state that persists
across test runs. In pytest-xdist parallel execution, each worker has its own
Python process with its own copy of this global state. However, the order in
which tests run on each worker is non-deterministic, leading to test interdependencies
where one test's plugin registrations can affect another test's expectations.

Note: Test files in this package should use `pytestmark = pytest.mark.serial`
to ensure they run on the same xdist worker. The root conftest.py handles
the xdist_group assignment for serial markers.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="function")
def clear_plugin_registry():
    """Clear the PluginRegistry before and after each test.

    This fixture ensures test isolation for the global PluginRegistry singleton.
    Even when running on the same worker, tests might leave residual state that
    affects subsequent tests.
    """
    try:
        from nlsq.diagnostics.plugin import PluginRegistry

        PluginRegistry.clear()
        yield
        PluginRegistry.clear()
    except ImportError:
        # Module doesn't exist yet - tests will fail with appropriate import error
        yield
