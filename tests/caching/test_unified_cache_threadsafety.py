"""Thread-safety tests for UnifiedCache."""

from __future__ import annotations

import threading

import jax.numpy as jnp
import pytest


@pytest.mark.unit
def test_get_global_cache_concurrent_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent get_global_cache() calls must return the same instance."""
    import nlsq.caching.unified_cache as mod

    # Reset global to force initialization race
    monkeypatch.setattr(mod, "_global_unified_cache", None)

    results: list = [None] * 20
    barrier = threading.Barrier(20)

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = mod.get_global_cache()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads must get the exact same instance
    assert all(r is results[0] for r in results)


@pytest.mark.unit
def test_unified_cache_concurrent_get_or_compile() -> None:
    """Concurrent get_or_compile() must not raise KeyError or corrupt state."""
    from nlsq.caching.unified_cache import UnifiedCache

    cache = UnifiedCache(maxsize=10, enable_stats=True)
    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def worker(idx: int) -> None:
        barrier.wait()
        try:

            def f(x):
                return x**2

            x = jnp.ones(idx + 1)
            cache.get_or_compile(f, (x,), {}, static_argnums=())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent access: {errors}"
    stats = cache.get_stats()
    assert stats["compilations"] + stats["hits"] == 20
