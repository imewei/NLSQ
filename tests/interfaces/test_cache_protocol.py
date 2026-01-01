"""Protocol contract tests for CacheProtocol.

This module tests that cache implementations conform to the
CacheProtocol and BoundedCacheProtocol defined in nlsq.interfaces.
"""

from typing import Any

import pytest

from nlsq.interfaces.cache_protocol import (
    BoundedCacheProtocol,
    CacheProtocol,
    DictCache,
)


class TestCacheProtocolDefinition:
    """Test that CacheProtocol is correctly defined."""

    def test_protocol_is_runtime_checkable(self):
        """CacheProtocol should be runtime_checkable."""

        class MockCache:
            def get(self, key: str, default: Any = None) -> Any:
                return default

            def set(self, key: str, value: Any) -> None:
                pass

            def clear(self) -> None:
                pass

        assert isinstance(MockCache(), CacheProtocol)

    def test_protocol_requires_all_methods(self):
        """Classes missing any method should not satisfy protocol."""

        class MissingGet:
            def set(self, key, value):
                pass

            def clear(self):
                pass

        class MissingSet:
            def get(self, key, default=None):
                return default

            def clear(self):
                pass

        class MissingClear:
            def get(self, key, default=None):
                return default

            def set(self, key, value):
                pass

        assert not isinstance(MissingGet(), CacheProtocol)
        assert not isinstance(MissingSet(), CacheProtocol)
        assert not isinstance(MissingClear(), CacheProtocol)


class TestDictCacheConformance:
    """Test that DictCache conforms to CacheProtocol."""

    def test_dict_cache_satisfies_protocol(self):
        """DictCache should satisfy CacheProtocol."""
        cache = DictCache()
        assert isinstance(cache, CacheProtocol)

    def test_dict_cache_get_default(self):
        """get() should return default for missing keys."""
        cache = DictCache()
        assert cache.get("missing") is None
        assert cache.get("missing", "default") == "default"

    def test_dict_cache_set_and_get(self):
        """set() and get() should work together."""
        cache = DictCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_dict_cache_clear(self):
        """clear() should remove all entries."""
        cache = DictCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_dict_cache_contains(self):
        """__contains__ should work correctly."""
        cache = DictCache()
        assert "key" not in cache
        cache.set("key", "value")
        assert "key" in cache

    def test_dict_cache_len(self):
        """__len__ should return item count."""
        cache = DictCache()
        assert len(cache) == 0
        cache.set("a", 1)
        assert len(cache) == 1
        cache.set("b", 2)
        assert len(cache) == 2


class TestBoundedCacheProtocol:
    """Test BoundedCacheProtocol requirements."""

    def test_protocol_is_runtime_checkable(self):
        """BoundedCacheProtocol should be runtime_checkable."""

        class MockBoundedCache:
            def get(self, key: str, default: Any = None) -> Any:
                return default

            def set(self, key: str, value: Any) -> None:
                pass

            def clear(self) -> None:
                pass

            @property
            def size_bytes(self) -> int:
                return 0

            @property
            def max_size_bytes(self) -> int:
                return 1024

            def evict(self, n_bytes: int) -> int:
                return 0

        assert isinstance(MockBoundedCache(), BoundedCacheProtocol)

    def test_protocol_requires_size_properties(self):
        """Classes missing size properties should not satisfy protocol."""

        class MissingSizeBytes:
            def get(self, key, default=None):
                return default

            def set(self, key, value):
                pass

            def clear(self):
                pass

            @property
            def max_size_bytes(self):
                return 1024

            def evict(self, n_bytes):
                return 0

        assert not isinstance(MissingSizeBytes(), BoundedCacheProtocol)


class TestConcreteImplementations:
    """Test that concrete NLSQ cache implementations satisfy protocols."""

    def test_unified_cache_has_cache_methods(self):
        """UnifiedCache should have cache-like methods."""
        from nlsq.caching.unified_cache import UnifiedCache

        cache = UnifiedCache()
        # UnifiedCache is specialized for JIT caching
        # It has get_or_compile method for compiled function caching
        assert hasattr(cache, "get_or_compile")
        assert hasattr(cache, "clear")

    def test_smart_cache_exists(self):
        """SmartCache should exist and be importable."""
        from nlsq.caching.smart_cache import SmartCache

        cache = SmartCache()
        # SmartCache provides get/set methods
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")


class TestCacheUsagePatterns:
    """Test common cache usage patterns."""

    def test_cache_as_lru(self):
        """Cache should work as simple LRU-like storage."""
        cache = DictCache()

        # Store multiple values
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")

        # Retrieve values
        assert cache.get("key_5") == "value_5"
        assert cache.get("key_nonexistent") is None

    def test_cache_with_complex_values(self):
        """Cache should handle complex values."""
        import numpy as np

        cache = DictCache()

        # Store numpy array
        arr = np.array([1.0, 2.0, 3.0])
        cache.set("array", arr)
        retrieved = cache.get("array")
        np.testing.assert_array_equal(retrieved, arr)

        # Store dict
        cache.set("dict", {"a": 1, "b": [1, 2, 3]})
        assert cache.get("dict")["a"] == 1

    def test_cache_key_overwrite(self):
        """Setting same key should overwrite value."""
        cache = DictCache()

        cache.set("key", "value1")
        assert cache.get("key") == "value1"

        cache.set("key", "value2")
        assert cache.get("key") == "value2"
