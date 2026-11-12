"""Round 8: multi-start loser-start log records must not leak to NOTSET/DEBUG handlers."""

import logging
import threading

import numpy as np
import pytest

import nlsq.global_optimization.multi_start as ms
from nlsq.global_optimization.config import GlobalOptimizationConfig

_LOG = logging.getLogger("nlsq.test_round8_defer")
_MSG = "Jacobian is rank-deficient (round8)"


class _Capture(logging.Handler):
    def __init__(self, level):
        super().__init__(level)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def _fake_start(f, x, y, p0, bounds, kw, cfkw):
    _LOG.warning(_MSG)
    # Start 0 is the winner (lowest loss).
    return np.asarray(p0), float(p0[0]), None


def _run(n_starts=6):
    pts = np.array([[float(i), 1.0] for i in range(n_starts)])
    x, y = np.linspace(0, 1, 5), np.ones(5)
    bounds = (np.zeros(2), np.full(2, 10.0))
    ms.MultiStartOrchestrator(
        GlobalOptimizationConfig(n_starts=n_starts)
    ).evaluate_starting_points(lambda x, a, b: a * x + b, x, y, pts, bounds)


@pytest.mark.parametrize("handler_level", [logging.NOTSET, logging.DEBUG])
@pytest.mark.parametrize("workers", [1, 3])
def test_only_winner_warning_reaches_handler(monkeypatch, handler_level, workers):
    monkeypatch.setattr(ms, "_fit_single_start", _fake_start)
    monkeypatch.setattr(ms, "_select_worker_count", lambda n: workers)
    h = _Capture(handler_level)
    root = logging.getLogger()
    old = root.level
    root.addHandler(h)
    root.setLevel(logging.WARNING)
    try:
        _run()
    finally:
        root.removeHandler(h)
        root.setLevel(old)
    assert h.records == [("WARNING", _MSG)]


def test_concurrent_multistart_runs_each_replay_once(monkeypatch):
    monkeypatch.setattr(ms, "_fit_single_start", _fake_start)
    monkeypatch.setattr(ms, "_select_worker_count", lambda n: 2)
    h = _Capture(logging.NOTSET)
    root = logging.getLogger()
    old = root.level
    root.addHandler(h)
    root.setLevel(logging.WARNING)
    try:
        threads = [threading.Thread(target=_run) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
    finally:
        root.removeHandler(h)
        root.setLevel(old)
    assert h.records == [("WARNING", _MSG)] * 3
