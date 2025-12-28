"""Tests for memory_pool module."""

import unittest

import jax.numpy as jnp

from nlsq.caching.memory_pool import (
    MemoryPool,
    TRFMemoryPool,
    clear_global_pool,
    get_global_pool,
)


class TestMemoryPool(unittest.TestCase):
    """Tests for MemoryPool class."""

    def setUp(self):
        """Set up test fixtures."""
        self.pool = MemoryPool(max_pool_size=5, enable_stats=True)

    def tearDown(self):
        """Clean up after tests."""
        self.pool.clear()

    def test_initialization(self):
        """Test memory pool initialization."""
        pool = MemoryPool(max_pool_size=10, enable_stats=True)
        self.assertEqual(pool.max_pool_size, 10)
        self.assertTrue(pool.enable_stats)
        self.assertEqual(len(pool.pools), 0)

    def test_allocate_new_array(self):
        """Test allocating a new array."""
        arr = self.pool.allocate((10,), dtype=jnp.float64)
        self.assertEqual(arr.shape, (10,))
        self.assertEqual(arr.dtype, jnp.float64)
        self.assertEqual(self.pool.stats["allocations"], 1)
        self.assertEqual(self.pool.stats["reuses"], 0)

    def test_reuse_array(self):
        """Test reusing array from pool."""
        # Allocate and release
        arr1 = self.pool.allocate((10,), dtype=jnp.float64)
        self.pool.release(arr1)

        # Allocate again - should reuse
        arr2 = self.pool.allocate((10,), dtype=jnp.float64)
        self.assertEqual(arr2.shape, (10,))
        self.assertEqual(self.pool.stats["allocations"], 1)
        self.assertEqual(self.pool.stats["reuses"], 1)

    def test_release_and_pool_size_limit(self):
        """Test pool size limit enforcement."""
        arrays = []
        for _ in range(10):
            arr = self.pool.allocate((5,), dtype=jnp.float32)
            arrays.append(arr)

        # Release all
        for arr in arrays:
            self.pool.release(arr)

        # Pool should only keep max_pool_size arrays
        key = ((5,), jnp.float32)
        self.assertLessEqual(len(self.pool.pools.get(key, [])), self.pool.max_pool_size)

    def test_different_shapes(self):
        """Test allocating arrays of different shapes."""
        arr1 = self.pool.allocate((10, 5), dtype=jnp.float64)
        arr2 = self.pool.allocate((20,), dtype=jnp.float64)
        arr3 = self.pool.allocate((10, 5), dtype=jnp.float32)

        self.assertEqual(arr1.shape, (10, 5))
        self.assertEqual(arr2.shape, (20,))
        self.assertEqual(arr3.shape, (10, 5))
        self.assertEqual(arr3.dtype, jnp.float32)

    def test_context_manager(self):
        """Test memory pool as context manager."""
        with MemoryPool(enable_stats=True) as pool:
            arr = pool.allocate((10,))
            self.assertEqual(arr.shape, (10,))

        # Pool should be cleared after context
        self.assertEqual(len(pool.pools), 0)

    def test_get_stats(self):
        """Test statistics tracking."""
        arr1 = self.pool.allocate((10,))
        self.pool.release(arr1)
        self.pool.allocate((10,))

        stats = self.pool.get_stats()
        self.assertEqual(stats["allocations"], 1)
        self.assertEqual(stats["reuses"], 1)
        self.assertEqual(stats["releases"], 1)
        self.assertGreater(stats["reuse_rate"], 0)

    def test_clear(self):
        """Test clearing the pool."""
        arr = self.pool.allocate((10,))
        self.pool.release(arr)

        self.pool.clear()

        self.assertEqual(len(self.pool.pools), 0)
        self.assertEqual(len(self.pool.allocated), 0)
        self.assertEqual(self.pool.stats["allocations"], 0)


class TestTRFMemoryPool(unittest.TestCase):
    """Tests for TRFMemoryPool class."""

    def test_initialization(self):
        """Test TRF memory pool initialization."""
        pool = TRFMemoryPool(m=100, n=10)
        self.assertEqual(pool.m, 100)
        self.assertEqual(pool.n, 10)
        self.assertEqual(pool.jacobian_buffer.shape, (100, 10))
        self.assertEqual(pool.residual_buffer.shape, (100,))
        self.assertEqual(pool.gradient_buffer.shape, (10,))

    def test_get_buffers(self):
        """Test getting buffers from TRF pool."""
        pool = TRFMemoryPool(m=50, n=5)

        jac = pool.get_jacobian_buffer()
        res = pool.get_residual_buffer()
        grad = pool.get_gradient_buffer()
        step = pool.get_step_buffer()

        self.assertEqual(jac.shape, (50, 5))
        self.assertEqual(res.shape, (50,))
        self.assertEqual(grad.shape, (5,))
        self.assertEqual(step.shape, (5,))

    def test_reset(self):
        """Test resetting TRF pool buffers."""
        pool = TRFMemoryPool(m=10, n=3)

        # Modify a buffer
        pool.gradient_buffer = jnp.ones(3)

        # Reset
        pool.reset()

        # Should be zeros again
        self.assertTrue(jnp.allclose(pool.gradient_buffer, jnp.zeros(3)))


class TestGlobalPool(unittest.TestCase):
    """Tests for global pool functions."""

    def tearDown(self):
        """Clean up global pool."""
        clear_global_pool()

    def test_get_global_pool(self):
        """Test getting global pool."""
        pool1 = get_global_pool()
        pool2 = get_global_pool()

        # Should return same instance
        self.assertIs(pool1, pool2)

    def test_clear_global_pool(self):
        """Test clearing global pool."""
        pool = get_global_pool()
        arr = pool.allocate((10,))
        pool.release(arr)

        clear_global_pool()

        # Get new pool instance
        new_pool = get_global_pool()
        self.assertEqual(len(new_pool.pools), 0)


if __name__ == "__main__":
    unittest.main()
