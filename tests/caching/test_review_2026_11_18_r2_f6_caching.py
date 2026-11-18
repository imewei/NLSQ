"""Regression tests for the 2026-11-18 round-2 review (fixer F6: caching/)."""

import logging
import threading

import jax.numpy as jnp
import pytest

from nlsq.caching.memory_manager import MemoryManager
from nlsq.caching.unified_cache import UnifiedCache


def _run_with_timeout(fn, timeout=10.0):
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("v", fn()), daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "deadlock: call did not finish"
    return box["v"]


def test_p07_log_handler_may_call_get_stats_on_hit():
    cache = UnifiedCache(maxsize=4)

    def f(x):
        return x + 1

    cache.get_or_compile(f, (jnp.ones(2),), {})

    class Handler(logging.Handler):
        def emit(self, record):
            cache.get_stats()

    lg = logging.getLogger("nlsq")
    handler = Handler()
    old = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        _run_with_timeout(lambda: cache.get_or_compile(f, (jnp.ones(2),), {}))
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old)


def test_p07_clear_runs_finalizers_outside_lock():
    cache = UnifiedCache(maxsize=4)

    class Holder:
        def __del__(self):
            cache.get_stats()

    holder = Holder()

    def f(x):
        return x + (holder is not None)

    cache.get_or_compile(f, (jnp.ones(2),), {})
    del holder, f
    _run_with_timeout(cache.clear)


def test_p07_clear_drops_stale_jit_trace():
    state = {"scale": 2.0}

    def f(x):
        return x * state["scale"]

    cache = UnifiedCache(maxsize=4)
    x = jnp.asarray(1.0)
    assert float(cache.get_or_compile(f, (x,), {})(x)) == 2.0
    state["scale"] = 3.0
    cache.clear()
    assert float(cache.get_or_compile(f, (x,), {})(x)) == 3.0


def test_p07_rejected_jit_options_not_counted_as_compilation():
    cache = UnifiedCache(maxsize=4)

    def f(x):
        return x

    with pytest.raises(Exception):  # noqa: B017 - jax raises ValueError/TypeError
        cache.get_or_compile(f, (jnp.ones(2),), {}, static_argnums=(2,))
    stats = cache.get_stats()
    assert stats["compilations"] == 0
    assert stats["cache_size"] == 0


def test_p07_chunk_plan_uses_fresh_memory_reading(monkeypatch):
    mm = MemoryManager()
    readings = iter([100e6, 1024.0, 1024.0, 1024.0])
    seen_refresh = []

    def fake(refresh=False):
        seen_refresh.append(refresh)
        return next(readings)

    monkeypatch.setattr(mm, "get_available_memory", fake)
    mm.estimate_chunking_strategy(1000, 10)
    assert seen_refresh[0] is True
