"""Thread-safety tests for CompilationCache."""

from __future__ import annotations

import threading

import pytest


@pytest.mark.unit
def test_get_global_compilation_cache_concurrent_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent init must return the same instance."""
    import nlsq.caching.compilation_cache as mod

    monkeypatch.setattr(mod, "_global_compilation_cache", None)

    results: list = [None] * 20
    barrier = threading.Barrier(20)

    def worker(idx: int) -> None:
        barrier.wait()
        results[idx] = mod.get_global_compilation_cache()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is results[0] for r in results)


@pytest.mark.unit
def test_compilation_cache_concurrent_compile() -> None:
    """Concurrent compile() must not corrupt cache state."""
    from nlsq.caching.compilation_cache import CompilationCache

    cache = CompilationCache(enable_stats=True, max_cache_size=10)
    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def make_func(n: int):
        def f(x):
            return x**n

        f.__name__ = f"poly_{n}"
        return f

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            func = make_func(idx)
            cache.compile(func, static_argnums=())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent compile: {errors}"
    stats = cache.get_stats()
    assert stats["compilations"] > 0
