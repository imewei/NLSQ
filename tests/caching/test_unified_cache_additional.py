"""Additional fast tests for UnifiedCache edge paths."""

from __future__ import annotations

import types

import jax.numpy as jnp
import pytest

from nlsq.caching import unified_cache
from nlsq.caching.unified_cache import (
    UnifiedCache,
    cached_jit,
    clear_cache,
    get_global_cache,
)


@pytest.mark.cache
def test_get_stats_disabled() -> None:
    """When stats are disabled, get_stats should report disabled."""
    cache = UnifiedCache(enable_stats=False)

    stats = cache.get_stats()

    assert stats == {"enabled": False}


@pytest.mark.cache
def test_eviction_on_maxsize() -> None:
    """Adding multiple entries should evict the least-recently-used entry."""
    cache = UnifiedCache(maxsize=1, enable_stats=True)

    def f1(x):
        return x + 1

    def f2(x):
        return x + 2

    x = jnp.array([1.0])

    compiled1 = cache.get_or_compile(f1, (x,), {}, static_argnums=())
    compiled1(x)

    compiled2 = cache.get_or_compile(f2, (x,), {}, static_argnums=())
    compiled2(x)

    stats = cache.get_stats()
    assert stats["evictions"] >= 1
    assert stats["cache_size"] == 1


@pytest.mark.cache
def test_function_hash_fallback_builtin() -> None:
    """_get_function_hash should produce a persistent hash for builtins."""
    cache = UnifiedCache(enable_stats=False)
    func_hash = cache._get_function_hash(len)
    # Should be a deterministic hex string (persistent across sessions)
    assert len(func_hash) == 16
    assert all(c in "0123456789abcdef" for c in func_hash)
    # Should be deterministic
    assert cache._get_function_hash(len) == func_hash


@pytest.mark.cache
def test_array_signature_non_array() -> None:
    """Non-array args should use type name signatures."""
    cache = UnifiedCache(enable_stats=False)
    assert cache._get_array_signature(5) == "int"


@pytest.mark.cache
def test_cached_jit_uses_mocked_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """cached_jit should use the unified cache without real compilation."""
    monkeypatch.setattr(unified_cache.jax, "jit", lambda func, **_k: func)

    @cached_jit
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    stats = get_global_cache().get_stats()
    assert stats["compilations"] >= 1


@pytest.mark.cache
def test_clear_cache_resets_global() -> None:
    """clear_cache should clear the global cache instance."""
    cache = get_global_cache()
    cache._cache["x"] = lambda x: x  # type: ignore[assignment]

    clear_cache()

    assert cache._cache == {}


@pytest.mark.cache
def test_repr_with_stats_disabled() -> None:
    """__repr__ should use size-only format when stats disabled."""
    cache = UnifiedCache(enable_stats=False)
    assert "UnifiedCache(size=" in repr(cache)
