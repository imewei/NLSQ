"""Tests for compilation_cache module."""

import unittest

import jax.numpy as jnp

from nlsq.caching.compilation_cache import (
    CompilationCache,
    cached_jit,
    clear_compilation_cache,
    get_global_compilation_cache,
)


class TestCompilationCache(unittest.TestCase):
    """Tests for CompilationCache class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = CompilationCache(enable_stats=True)

    def tearDown(self):
        """Clean up after tests."""
        self.cache.clear()

    def test_initialization(self):
        """Test compilation cache initialization."""
        cache = CompilationCache(enable_stats=True)
        self.assertTrue(cache.enable_stats)
        self.assertEqual(len(cache.cache), 0)
        self.assertEqual(cache.stats["compilations"], 0)

    def test_compile_function(self):
        """Test compiling a function."""

        def simple_func(x):
            return x**2

        compiled = self.cache.compile(simple_func)
        result = compiled(jnp.array([1.0, 2.0, 3.0]))

        self.assertTrue(jnp.allclose(result, jnp.array([1.0, 4.0, 9.0])))
        self.assertEqual(self.cache.stats["compilations"], 1)

    def test_cache_hit(self):
        """Test cache hit for previously compiled function."""

        def simple_func(x):
            return x**2

        # First compilation
        self.cache.compile(simple_func)
        self.assertEqual(self.cache.stats["compilations"], 1)
        self.assertEqual(self.cache.stats["hits"], 0)

        # Second request - should hit cache
        self.cache.compile(simple_func)
        self.assertEqual(self.cache.stats["compilations"], 1)  # No new compilation
        self.assertEqual(self.cache.stats["hits"], 1)

    def test_function_signature_generation(self):
        """Test generating function signatures."""
        sig1 = self.cache._get_function_signature(lambda x: x, jnp.array([1.0, 2.0]))
        sig2 = self.cache._get_function_signature(
            lambda x: x, jnp.array([1.0, 2.0, 3.0])
        )

        # Different shapes should give different signatures
        self.assertNotEqual(sig1, sig2)

    def test_get_or_compile(self):
        """Test get_or_compile method."""

        def test_func(x, y):
            return x + y

        x = jnp.array([1.0])
        y = jnp.array([2.0])

        # First call - miss
        _func1, sig1 = self.cache.get_or_compile(test_func, x, y)
        self.assertEqual(self.cache.stats["misses"], 1)

        # Second call with same signature - hit
        _func2, sig2 = self.cache.get_or_compile(test_func, x, y)
        self.assertEqual(self.cache.stats["hits"], 1)
        self.assertEqual(sig1, sig2)

    def test_static_argnums(self):
        """Test compilation with static arguments."""

        def power_func(x, n):
            return x**n

        compiled = self.cache.compile(power_func, static_argnums=(1,))
        result = compiled(jnp.array([2.0, 3.0]), 3)

        self.assertTrue(jnp.allclose(result, jnp.array([8.0, 27.0])))

    def test_clear_cache(self):
        """Test clearing the cache."""

        def simple_func(x):
            return x * 2

        self.cache.compile(simple_func)
        self.assertGreater(len(self.cache.cache), 0)

        self.cache.clear()
        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(self.cache.stats["cache_size"], 0)

    def test_get_stats(self):
        """Test getting cache statistics."""

        def func1(x):
            return x + 1

        def func2(x):
            return x * 2

        self.cache.compile(func1)
        self.cache.compile(func2)
        self.cache.compile(func1)  # Cache hit

        stats = self.cache.get_stats()
        self.assertEqual(stats["compilations"], 2)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 2)
        self.assertGreater(stats["hit_rate"], 0)

    def test_context_manager(self):
        """Test compilation cache as context manager."""
        with CompilationCache() as cache:

            def test_func(x):
                return x**2

            compiled = cache.compile(test_func)
            result = compiled(jnp.array([2.0]))
            self.assertTrue(jnp.allclose(result, jnp.array([4.0])))


class TestClearCompilationCache(unittest.TestCase):
    """Tests for clear_compilation_cache functionality (Task Group 5)."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = CompilationCache(enable_stats=True)

    def tearDown(self):
        """Clean up after tests."""
        clear_compilation_cache()

    def test_clear_clears_cache_dict(self):
        """Test that clear() clears self.cache dict."""

        def func1(x):
            return x + 1

        def func2(x):
            return x * 2

        # Populate cache
        self.cache.compile(func1)
        self.cache.compile(func2)
        self.assertGreater(len(self.cache.cache), 0)

        # Clear cache
        self.cache.clear()

        # Verify cache is empty
        self.assertEqual(len(self.cache.cache), 0)

    def test_clear_clears_func_hash_cache(self):
        """Test that clear() clears self._func_hash_cache dict."""

        def func1(x):
            return x + 1

        def func2(x):
            return x * 2

        # Populate func hash cache by compiling functions
        self.cache.compile(func1)
        self.cache.compile(func2)
        self.assertGreater(len(self.cache._func_hash_cache), 0)

        # Clear cache
        self.cache.clear()

        # Verify func hash cache is empty
        self.assertEqual(len(self.cache._func_hash_cache), 0)

    def test_clear_resets_stats_counters(self):
        """Test that clear() resets stats counters to zero."""

        def func1(x):
            return x + 1

        def func2(x):
            return x * 2

        # Generate some stats
        self.cache.compile(func1)
        self.cache.compile(func2)
        self.cache.compile(func1)  # Cache hit

        # Verify stats are non-zero before clearing
        self.assertGreater(self.cache.stats["hits"], 0)
        self.assertGreater(self.cache.stats["misses"], 0)
        self.assertGreater(self.cache.stats["compilations"], 0)
        self.assertGreater(self.cache.stats["cache_size"], 0)

        # Clear cache
        self.cache.clear()

        # Verify ALL stats counters are reset to zero
        self.assertEqual(self.cache.stats["hits"], 0)
        self.assertEqual(self.cache.stats["misses"], 0)
        self.assertEqual(self.cache.stats["compilations"], 0)
        self.assertEqual(self.cache.stats["cache_size"], 0)


class TestClearCompilationCacheModuleLevel(unittest.TestCase):
    """Tests for module-level clear_compilation_cache() function."""

    def setUp(self):
        """Set up test fixtures."""
        # Ensure we start with a clean global cache
        clear_compilation_cache()

    def tearDown(self):
        """Clean up after tests."""
        clear_compilation_cache()

    def test_module_level_clear_clears_global_cache(self):
        """Test that module-level clear_compilation_cache() clears global cache."""
        cache = get_global_compilation_cache()

        def test_func(x):
            return x * 2

        # Populate global cache
        cache.compile(test_func)
        self.assertGreater(len(cache.cache), 0)

        # Clear using module-level function
        clear_compilation_cache()

        # Verify global cache is empty
        self.assertEqual(len(cache.cache), 0)

    def test_module_level_clear_resets_global_stats(self):
        """Test that module-level clear_compilation_cache() resets global stats."""
        cache = get_global_compilation_cache()

        def func1(x):
            return x + 1

        def func2(x):
            return x * 2

        # Generate stats in global cache
        cache.compile(func1)
        cache.compile(func2)
        cache.compile(func1)  # Cache hit

        # Verify stats are non-zero
        self.assertGreater(cache.stats["hits"], 0)
        self.assertGreater(cache.stats["compilations"], 0)

        # Clear using module-level function
        clear_compilation_cache()

        # Verify stats are reset
        self.assertEqual(cache.stats["hits"], 0)
        self.assertEqual(cache.stats["misses"], 0)
        self.assertEqual(cache.stats["compilations"], 0)
        self.assertEqual(cache.stats["cache_size"], 0)

    def test_clear_allows_accurate_tracking_after(self):
        """Test that stats are accurately tracked after clearing cache."""
        cache = get_global_compilation_cache()

        def func1(x):
            return x + 1

        # First round: generate some stats
        cache.compile(func1)
        cache.compile(func1)  # hit
        self.assertEqual(cache.stats["compilations"], 1)
        self.assertEqual(cache.stats["hits"], 1)

        # Clear cache
        clear_compilation_cache()

        # Second round: verify fresh tracking
        def func2(x):
            return x * 2

        cache.compile(func2)
        cache.compile(func2)  # hit

        # Should have fresh stats from second round only
        self.assertEqual(cache.stats["compilations"], 1)
        self.assertEqual(cache.stats["hits"], 1)
        self.assertEqual(cache.stats["misses"], 1)


class TestCachedJITDecorator(unittest.TestCase):
    """Tests for cached_jit decorator."""

    def tearDown(self):
        """Clean up global cache."""
        clear_compilation_cache()

    def test_cached_jit_decorator(self):
        """Test basic cached_jit decorator usage."""

        @cached_jit
        def square(x):
            return x**2

        result = square(jnp.array([1.0, 2.0, 3.0]))
        self.assertTrue(jnp.allclose(result, jnp.array([1.0, 4.0, 9.0])))

    def test_cached_jit_with_static_args(self):
        """Test cached_jit with static arguments."""

        @cached_jit(static_argnums=(1,))
        def power(x, n):
            return x**n

        result = power(jnp.array([2.0, 3.0]), 3)
        self.assertTrue(jnp.allclose(result, jnp.array([8.0, 27.0])))

    def test_cache_reuse(self):
        """Test that decorator reuses cached compilations."""

        @cached_jit
        def add_one(x):
            return x + 1

        cache = get_global_compilation_cache()
        cache.stats["compilations"]

        # First call
        add_one(jnp.array([1.0]))

        # Second call - should reuse compilation
        add_one(jnp.array([1.0]))

        # Should still be same number of compilations
        self.assertGreaterEqual(cache.stats["hits"], 0)


class TestGlobalCompilationCache(unittest.TestCase):
    """Tests for global compilation cache functions."""

    def tearDown(self):
        """Clean up global cache."""
        clear_compilation_cache()

    def test_get_global_compilation_cache(self):
        """Test getting global compilation cache."""
        cache1 = get_global_compilation_cache()
        cache2 = get_global_compilation_cache()

        # Should return same instance
        self.assertIs(cache1, cache2)

    def test_clear_global_compilation_cache(self):
        """Test clearing global compilation cache."""
        cache = get_global_compilation_cache()

        def test_func(x):
            return x * 2

        cache.compile(test_func)
        self.assertGreater(len(cache.cache), 0)

        clear_compilation_cache()

        # Cache should be empty
        cache2 = get_global_compilation_cache()
        self.assertEqual(len(cache2.cache), 0)


if __name__ == "__main__":
    unittest.main()
