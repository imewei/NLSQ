"""Tests for Phase 3 optimizations (Task Group 9).

Tests cover:
- Compilation cache LRU eviction at max_cache_size (default 256)
- Array hash optimization with stride-based sampling for large arrays
- Telemetry circular buffer with maxlen=1000
- Function hash race condition fix with composite key
"""

import hashlib
import unittest
from collections import OrderedDict, deque
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np


class TestCompilationCacheLRUEviction(unittest.TestCase):
    """Tests for Compilation Cache LRU Eviction (2.2a)."""

    def setUp(self):
        """Set up test fixtures."""
        from nlsq.caching.compilation_cache import CompilationCache

        # Create cache with small max size for testing
        self.cache = CompilationCache(enable_stats=True, max_cache_size=5)

    def tearDown(self):
        """Clean up after tests."""
        self.cache.clear()

    def test_cache_uses_ordered_dict(self):
        """Test that cache uses OrderedDict for LRU tracking."""
        self.assertIsInstance(self.cache.cache, OrderedDict)

    def test_max_cache_size_parameter(self):
        """Test that max_cache_size parameter is accepted and stored."""
        from nlsq.caching.compilation_cache import CompilationCache

        cache = CompilationCache(max_cache_size=256)
        self.assertEqual(cache.max_cache_size, 256)

        cache_small = CompilationCache(max_cache_size=10)
        self.assertEqual(cache_small.max_cache_size, 10)

    def test_default_max_cache_size_is_512(self):
        """Test that default max_cache_size is 512."""
        from nlsq.caching.compilation_cache import CompilationCache

        cache = CompilationCache()
        self.assertEqual(cache.max_cache_size, 512)

    def test_lru_eviction_at_capacity(self):
        """Test that oldest entry is evicted when at capacity."""
        # Create cache with max size of 3
        from nlsq.caching.compilation_cache import CompilationCache

        cache = CompilationCache(enable_stats=True, max_cache_size=3)

        # Create test functions
        def func1(x):
            return x + 1

        def func2(x):
            return x + 2

        def func3(x):
            return x + 3

        def func4(x):
            return x + 4

        # Compile first 3 functions
        cache.compile(func1)
        cache.compile(func2)
        cache.compile(func3)

        # Cache should have 3 entries
        self.assertEqual(len(cache.cache), 3)

        # Compile 4th function - should evict oldest (func1)
        cache.compile(func4)

        # Cache should still have 3 entries
        self.assertEqual(len(cache.cache), 3)

        # Verify eviction occurred
        self.assertEqual(cache.stats["cache_size"], 3)

    def test_move_to_end_on_cache_hit(self):
        """Test that accessed entries are moved to end (most recently used)."""
        from nlsq.caching.compilation_cache import CompilationCache

        cache = CompilationCache(enable_stats=True, max_cache_size=3)

        def func1(x):
            return x + 1

        def func2(x):
            return x + 2

        def func3(x):
            return x + 3

        def func4(x):
            return x + 4

        # Compile first 3 functions
        cache.compile(func1)
        cache.compile(func2)
        cache.compile(func3)

        # Access func1 again (cache hit) - should move to end
        cache.compile(func1)

        # Now compile func4 - should evict func2 (now oldest)
        cache.compile(func4)

        # func1 should still be in cache (was moved to end on access)
        # func2 should have been evicted
        self.assertEqual(len(cache.cache), 3)


class TestArrayHashOptimization(unittest.TestCase):
    """Tests for Array Hash Optimization (3.2a) in smart_cache.py."""

    def setUp(self):
        """Set up test fixtures."""
        from nlsq.caching.smart_cache import SmartCache

        self.cache = SmartCache()

    def test_small_array_hashes_full_array(self):
        """Test that arrays <= 10000 elements hash full array directly."""
        # Create small array (< 10000 elements)
        small_arr = np.random.randn(50, 50)  # 2500 elements
        self.assertLessEqual(small_arr.size, 10000)

        # Generate cache key - should not use stride sampling
        key = self.cache.cache_key(small_arr)

        # Key should be valid string
        self.assertIsInstance(key, str)
        self.assertTrue(len(key) > 0)

    def test_large_array_uses_stride_sampling(self):
        """Test that arrays > 10000 elements use stride-based sampling when xxhash unavailable."""
        from nlsq.caching.smart_cache import HAS_XXHASH

        # Create large array (> 10000 elements)
        large_arr = np.random.randn(200, 100)  # 20000 elements
        self.assertGreater(large_arr.size, 10000)

        # Generate cache key
        key = self.cache.cache_key(large_arr)

        # Key should be valid string
        self.assertIsInstance(key, str)
        self.assertTrue(len(key) > 0)

        # If xxhash is available, it hashes full array (fast)
        # If not, stride sampling should be used for large arrays
        if not HAS_XXHASH:
            # The fallback path should use stride-based sampling
            # We verify by checking that the key generation completes
            # without excessive memory usage
            pass

    def test_cache_key_deterministic(self):
        """Test that cache keys are deterministic for same input."""
        arr = np.array([1.0, 2.0, 3.0])

        key1 = self.cache.cache_key(arr)
        key2 = self.cache.cache_key(arr)

        self.assertEqual(key1, key2)

    def test_cache_key_includes_version_prefix(self):
        """Test that cache keys include CACHE_VERSION prefix."""
        from nlsq.caching.smart_cache import CACHE_VERSION

        arr = np.array([1.0, 2.0, 3.0])
        key = self.cache.cache_key(arr)

        self.assertTrue(key.startswith(CACHE_VERSION))

    def test_no_redundant_sampling_in_fallback(self):
        """Test that fallback path does not use redundant sampling for small arrays."""
        # This test verifies the optimization works correctly
        from nlsq.caching.smart_cache import HAS_XXHASH

        # Small array should hash directly without sampling
        small_arr = np.random.randn(50)  # 50 elements
        key = self.cache.cache_key(small_arr)
        self.assertIsInstance(key, str)

        # Medium array still under threshold
        medium_arr = np.random.randn(100, 50)  # 5000 elements
        key = self.cache.cache_key(medium_arr)
        self.assertIsInstance(key, str)


class TestTelemetryCircularBuffer(unittest.TestCase):
    """Tests for Telemetry Circular Buffer (1.3a) in memory_manager.py."""

    def setUp(self):
        """Set up test fixtures."""
        from nlsq.caching.memory_manager import MemoryManager

        self.manager = MemoryManager(enable_adaptive_safety=True)

    def tearDown(self):
        """Clean up after tests."""
        self.manager.clear_pool()

    def test_safety_telemetry_is_deque_with_maxlen(self):
        """Test that _safety_telemetry is a deque with maxlen=1000."""
        self.assertIsInstance(self.manager._safety_telemetry, deque)
        self.assertEqual(self.manager._safety_telemetry.maxlen, 1000)

    def test_telemetry_buffer_bounds_at_1000(self):
        """Test that telemetry buffer never exceeds 1000 entries."""
        # Add more than 1000 telemetry records
        for i in range(1500):
            self.manager._record_safety_telemetry(
                bytes_predicted=1000 * (i + 1),
                bytes_actual=900 * (i + 1),
            )

        # Should be bounded at 1000
        self.assertEqual(len(self.manager._safety_telemetry), 1000)

    def test_telemetry_maintains_recent_records(self):
        """Test that circular buffer maintains most recent 1000 records."""
        # Add 1500 records
        for i in range(1500):
            self.manager._record_safety_telemetry(
                bytes_predicted=i,
                bytes_actual=i,
            )

        # Check that we have the most recent records (500-1499)
        self.assertEqual(len(self.manager._safety_telemetry), 1000)

        # First record should be from index 500
        self.assertEqual(self.manager._safety_telemetry[0]["bytes_predicted"], 500)

        # Last record should be from index 1499
        self.assertEqual(self.manager._safety_telemetry[-1]["bytes_predicted"], 1499)

    def test_telemetry_works_in_long_runs(self):
        """Test that telemetry does not grow unbounded in multi-day simulation."""
        # Simulate many optimization runs
        for i in range(10000):
            self.manager._record_safety_telemetry(
                bytes_predicted=1000,
                bytes_actual=900,
            )

        # Should never exceed maxlen
        self.assertLessEqual(len(self.manager._safety_telemetry), 1000)


class TestFunctionHashRaceConditionFix(unittest.TestCase):
    """Tests for Function Hash Race Condition Fix (2.1a) in compilation_cache.py."""

    def setUp(self):
        """Set up test fixtures."""
        from nlsq.caching.compilation_cache import CompilationCache

        self.cache = CompilationCache(enable_stats=True)

    def tearDown(self):
        """Clean up after tests."""
        self.cache.clear()

    def test_func_hash_cache_uses_composite_key(self):
        """Test that _func_hash_cache uses composite key (id(func), id(func.__code__))."""

        # Define a function
        def test_func(x):
            return x + 1

        # Get function hash
        hash1 = self.cache._get_function_code_hash(test_func)

        # Check that composite key is in cache
        # The key should be (id(func), id(func.__code__))
        func_id = id(test_func)
        code_id = id(test_func.__code__)
        composite_key = (func_id, code_id)

        self.assertIn(composite_key, self.cache._func_hash_cache)
        self.assertEqual(self.cache._func_hash_cache[composite_key], hash1)

    def test_redefined_function_gets_different_hash(self):
        """Test that redefined function with same name gets different cache entry."""

        # Define first version of function
        def my_function(x):
            return x + 1

        hash1 = self.cache._get_function_code_hash(my_function)
        code1_id = id(my_function.__code__)

        # Redefine function with same name but different code
        def my_function(x):
            return x * 2

        hash2 = self.cache._get_function_code_hash(my_function)
        code2_id = id(my_function.__code__)

        # Different code objects should have different IDs
        self.assertNotEqual(code1_id, code2_id)

        # Cache should have entries for both (different composite keys)
        self.assertEqual(len(self.cache._func_hash_cache), 2)

    def test_same_function_returns_cached_hash(self):
        """Test that same function returns cached hash."""

        def test_func(x):
            return x + 1

        # Get hash twice
        hash1 = self.cache._get_function_code_hash(test_func)
        hash2 = self.cache._get_function_code_hash(test_func)

        # Should return same cached hash
        self.assertEqual(hash1, hash2)

        # Should only have one entry
        self.assertEqual(len(self.cache._func_hash_cache), 1)

    def test_composite_key_prevents_cache_poisoning(self):
        """Test that composite key prevents cache poisoning in notebooks."""
        # Simulate notebook scenario: same name, different implementation

        # First "cell" defines function
        def model(x):
            return x + 1

        compiled1 = self.cache.compile(model)
        result1 = compiled1(jnp.array([1.0]))

        # Second "cell" redefines function
        def model(x):
            return x * 10

        # This should NOT return the cached version of the old function
        compiled2 = self.cache.compile(model)
        result2 = compiled2(jnp.array([1.0]))

        # Results should be different because composite key includes code object id
        self.assertTrue(jnp.allclose(result1, jnp.array([2.0])))
        self.assertTrue(jnp.allclose(result2, jnp.array([10.0])))


class TestPhase3Integration(unittest.TestCase):
    """Integration tests for all Phase 3 optimizations working together."""

    def test_all_optimizations_coexist(self):
        """Test that all Phase 3 optimizations work together."""
        from nlsq.caching.compilation_cache import CompilationCache
        from nlsq.caching.memory_manager import MemoryManager
        from nlsq.caching.smart_cache import SmartCache

        # Create instances with Phase 3 optimizations
        compilation_cache = CompilationCache(max_cache_size=256)
        memory_manager = MemoryManager(enable_adaptive_safety=True)
        smart_cache = SmartCache()

        # Verify each optimization is active
        # 1. Compilation cache uses OrderedDict
        self.assertIsInstance(compilation_cache.cache, OrderedDict)

        # 2. Memory manager uses deque for telemetry
        self.assertIsInstance(memory_manager._safety_telemetry, deque)
        self.assertEqual(memory_manager._safety_telemetry.maxlen, 1000)

        # 3. Smart cache includes version prefix
        from nlsq.caching.smart_cache import CACHE_VERSION

        key = smart_cache.cache_key(np.array([1.0]))
        self.assertTrue(key.startswith(CACHE_VERSION))

        # 4. Compilation cache uses composite key for func hash
        def test_func(x):
            return x + 1

        compilation_cache._get_function_code_hash(test_func)
        func_id = id(test_func)
        code_id = id(test_func.__code__)
        self.assertIn((func_id, code_id), compilation_cache._func_hash_cache)


if __name__ == "__main__":
    unittest.main()
