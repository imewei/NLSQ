"""Thread-safety tests for NLSQ logger."""

from __future__ import annotations

import threading

import pytest


@pytest.mark.unit
def test_get_logger_concurrent_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent get_logger() for same name must return same instance."""
    import nlsq.utils.logging as mod

    monkeypatch.setattr(mod, "_loggers", {})

    results: list = [None] * 20
    barrier = threading.Barrier(20)

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = mod.get_logger("test_concurrent")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is results[0] for r in results)
