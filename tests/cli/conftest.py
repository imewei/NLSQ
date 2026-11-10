"""CLI test isolation: restore the global ``nlsq.cli`` logger after every test."""

import logging

import pytest


@pytest.fixture(autouse=True)
def _restore_nlsq_cli_logger():
    lg = logging.getLogger("nlsq.cli")
    handlers, level = list(lg.handlers), lg.level
    yield
    for h in list(lg.handlers):
        if h not in handlers:
            lg.removeHandler(h)
            h.close()
    lg.handlers[:] = handlers
    lg.setLevel(level)
