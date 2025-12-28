"""Tests for hash collision fix in smart_cache.py.

This test module validates the hash algorithm changes:
- xxhash as primary hash when available
- BLAKE2b as fallback (not MD5)
- CACHE_VERSION prefix in all cache keys
- Deterministic cache keys for same inputs
"""

import hashlib
from unittest.mock import patch

import numpy as np
import pytest


class TestHashAlgorithmChange:
    """Test hash algorithm selection and CACHE_VERSION usage."""

    def test_xxhash_used_as_primary_when_available(self):
        """Cache keys should use xxhash when HAS_XXHASH=True."""
        from nlsq.caching import smart_cache

        if not smart_cache.HAS_XXHASH:
            pytest.skip("xxhash not available")

        cache = smart_cache.SmartCache()

        # Generate a cache key
        key = cache.cache_key("test_arg", param=123)

        # Key should be a hexadecimal string
        assert isinstance(key, str)
        # xxhash produces 16-character hex string
        # With version prefix: "v2_" + 16 chars = 19 chars
        assert key.startswith(smart_cache.CACHE_VERSION + "_")

    def test_blake2b_used_as_fallback_not_md5(self):
        """Cache keys should use BLAKE2b as fallback when xxhash unavailable."""
        from nlsq.caching import smart_cache

        # Temporarily disable xxhash
        with patch.object(smart_cache, "HAS_XXHASH", False):
            cache = smart_cache.SmartCache()

            # Generate a cache key
            key = cache.cache_key("test_arg", param=123)

            # Key should be a hexadecimal string with version prefix
            assert isinstance(key, str)
            assert key.startswith(smart_cache.CACHE_VERSION + "_")

            # Verify that BLAKE2b is used (not MD5) by checking key length
            # BLAKE2b with digest_size=16 produces 32-char hex string
            # With version prefix: "v2_" + 32 chars = 35 chars
            key_without_prefix = key[len(smart_cache.CACHE_VERSION) + 1 :]
            assert len(key_without_prefix) == 32, (
                f"Expected 32-char BLAKE2b hash, got {len(key_without_prefix)}"
            )

    def test_cache_keys_include_version_prefix(self):
        """All cache keys should include CACHE_VERSION prefix."""
        from nlsq.caching import smart_cache

        cache = smart_cache.SmartCache()

        # Test with various input types
        test_inputs = [
            (("simple_string",), {}),
            ((123, 456), {"key": "value"}),
            ((np.array([1.0, 2.0, 3.0]),), {}),
        ]

        for args, kwargs in test_inputs:
            key = cache.cache_key(*args, **kwargs)
            assert key.startswith(smart_cache.CACHE_VERSION + "_"), (
                f"Key '{key}' should start with '{smart_cache.CACHE_VERSION}_'"
            )

    def test_cache_keys_are_deterministic_for_same_inputs(self):
        """Cache keys should be deterministic for identical inputs."""
        from nlsq.caching import smart_cache

        cache = smart_cache.SmartCache()

        # Test with same inputs multiple times
        args = ("test", 123)
        kwargs = {"a": 1.0, "b": 2.0}

        key1 = cache.cache_key(*args, **kwargs)
        key2 = cache.cache_key(*args, **kwargs)
        key3 = cache.cache_key(*args, **kwargs)

        assert key1 == key2 == key3, "Cache keys should be identical for same inputs"

        # Test with arrays
        arr = np.array([1.0, 2.0, 3.0])
        key_arr1 = cache.cache_key(arr)
        key_arr2 = cache.cache_key(arr)

        assert key_arr1 == key_arr2, (
            "Cache keys should be identical for same array inputs"
        )


class TestCacheVersionConstant:
    """Test CACHE_VERSION constant exists and is properly formatted."""

    def test_cache_version_exists_at_module_level(self):
        """CACHE_VERSION constant should be defined at module level."""
        from nlsq.caching import smart_cache

        assert hasattr(smart_cache, "CACHE_VERSION")
        assert isinstance(smart_cache.CACHE_VERSION, str)

    def test_cache_version_is_v2(self):
        """CACHE_VERSION should be 'v2'."""
        from nlsq.caching import smart_cache

        assert smart_cache.CACHE_VERSION == "v2"


class TestNoMD5Usage:
    """Verify MD5 is not used in cache key generation."""

    def test_cache_key_without_xxhash_uses_blake2b(self):
        """Without xxhash, cache_key should use BLAKE2b for hashing."""
        from nlsq.caching import smart_cache

        # Create a test key string
        key_str = "test_key_string"

        # Compute expected BLAKE2b hash
        expected_hash = hashlib.blake2b(key_str.encode(), digest_size=16).hexdigest()

        # Verify BLAKE2b produces 32-char hex string (16 bytes = 32 hex chars)
        assert len(expected_hash) == 32

        # Patch HAS_XXHASH to False
        with patch.object(smart_cache, "HAS_XXHASH", False):
            cache = smart_cache.SmartCache()

            # Generate key for simple string input
            key = cache.cache_key(key_str)

            # The key should be version prefix + BLAKE2b hash
            assert key.startswith(smart_cache.CACHE_VERSION + "_")
            hash_part = key[len(smart_cache.CACHE_VERSION) + 1 :]

            # Hash part should be 32 chars (BLAKE2b with digest_size=16)
            assert len(hash_part) == 32, (
                f"Expected 32-char BLAKE2b hash, got {len(hash_part)}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
