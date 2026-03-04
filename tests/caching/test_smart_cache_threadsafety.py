"""Thread-safety tests for SmartCache and JITCompilationCache."""

from __future__ import annotations

import threading

import numpy as np
import pytest


@pytest.mark.unit
def test_smart_cache_concurrent_get_set() -> None:
    """Concurrent get/set must not raise or corrupt state."""
    from nlsq.caching.smart_cache import SmartCache

    cache = SmartCache(max_memory_items=10, disk_cache_enabled=False, enable_stats=True)
    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            key = cache.cache_key(np.ones(idx + 1))
            cache.set(key, np.array([float(idx)]))
            result = cache.get(key)
            assert result is not None
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent access: {errors}"
    stats = cache.get_stats()
    assert stats["hits"] > 0


@pytest.mark.unit
def test_smart_cache_concurrent_invalidate() -> None:
    """Concurrent invalidate must not raise or corrupt state."""
    from nlsq.caching.smart_cache import SmartCache

    cache = SmartCache(max_memory_items=50, disk_cache_enabled=False, enable_stats=True)
    errors: list[Exception] = []

    # Pre-populate the cache
    keys = []
    for i in range(30):
        key = cache.cache_key(np.ones(i + 1))
        cache.set(key, np.array([float(i)]))
        keys.append(key)

    barrier = threading.Barrier(15)

    def writer(idx: int) -> None:
        barrier.wait()
        try:
            key = cache.cache_key(np.ones(idx + 100))
            cache.set(key, np.array([float(idx)]))
        except Exception as e:
            errors.append(e)

    def invalidator(idx: int) -> None:
        barrier.wait()
        try:
            if idx < len(keys):
                cache.invalidate(keys[idx])
            else:
                cache.invalidate()
        except Exception as e:
            errors.append(e)

    threads = [
        *(threading.Thread(target=writer, args=(i,)) for i in range(10)),
        *(threading.Thread(target=invalidator, args=(i,)) for i in range(5)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent access: {errors}"


@pytest.mark.unit
def test_smart_cache_concurrent_eviction() -> None:
    """Concurrent writes exceeding max_memory_items must not corrupt state."""
    from nlsq.caching.smart_cache import SmartCache

    cache = SmartCache(max_memory_items=5, disk_cache_enabled=False, enable_stats=True)
    errors: list[Exception] = []
    barrier = threading.Barrier(20)

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            key = cache.cache_key(np.arange(idx + 1))
            cache.set(key, np.array([float(idx)]))
            # Immediately read back
            cache.get(key)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent eviction: {errors}"
    stats = cache.get_stats()
    assert stats["memory_size"] <= cache.max_memory_items


@pytest.mark.unit
def test_jit_compilation_cache_concurrent() -> None:
    """Concurrent get_or_compile must not corrupt state."""
    from nlsq.caching.smart_cache import JITCompilationCache

    cache = JITCompilationCache()
    errors: list[Exception] = []
    barrier = threading.Barrier(10)

    def make_func(n: int):
        def f(x):
            return x**n

        f.__name__ = f"poly_{n}"
        f.__module__ = "test"
        return f

    def worker(idx: int) -> None:
        barrier.wait()
        try:
            func = make_func(idx)
            compiled = cache.get_or_compile(func)
            assert callable(compiled)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent access: {errors}"


@pytest.mark.unit
def test_jit_compilation_cache_concurrent_clear() -> None:
    """Concurrent clear + get_or_compile must not raise."""
    from nlsq.caching.smart_cache import JITCompilationCache

    cache = JITCompilationCache()
    errors: list[Exception] = []
    barrier = threading.Barrier(10)

    def make_func(n: int):
        def f(x):
            return x + n

        f.__name__ = f"add_{n}"
        f.__module__ = "test"
        return f

    def compiler(idx: int) -> None:
        barrier.wait()
        try:
            func = make_func(idx)
            cache.get_or_compile(func)
        except Exception as e:
            errors.append(e)

    def clearer() -> None:
        barrier.wait()
        try:
            cache.clear()
            cache.get_stats()
        except Exception as e:
            errors.append(e)

    threads = [
        *(threading.Thread(target=compiler, args=(i,)) for i in range(8)),
        threading.Thread(target=clearer),
        threading.Thread(target=clearer),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Errors during concurrent access: {errors}"
