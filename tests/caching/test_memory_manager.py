"""
Comprehensive test suite for memory_manager module.

Tests intelligent memory management including:
- Memory usage monitoring and prediction
- Array pooling and reuse
- Memory guards and availability checking
- Context managers for temporary allocations
- Chunking strategy estimation
"""

import unittest
from unittest.mock import patch

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nlsq.caching.memory_manager import (
    MemoryManager,
    clear_memory_pool,
    get_memory_manager,
    get_memory_stats,
)


class TestMemoryManagerBasic(unittest.TestCase):
    """Basic tests for MemoryManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after each test."""
        self.manager.clear_pool()

    def test_initialization(self):
        """Test MemoryManager initialization."""
        manager = MemoryManager(gc_threshold=0.7, safety_factor=1.5)

        self.assertEqual(manager.gc_threshold, 0.7)
        self.assertEqual(manager.safety_factor, 1.5)
        self.assertEqual(len(manager.memory_pool), 0)
        self.assertEqual(len(manager.allocation_history), 0)

    def test_initialization_defaults(self):
        """Test MemoryManager with default parameters."""
        manager = MemoryManager()

        self.assertEqual(manager.gc_threshold, 0.8)
        self.assertEqual(manager.safety_factor, 1.2)

    def test_get_available_memory(self):
        """Test getting available memory."""
        available = self.manager.get_available_memory()

        # Should return a positive number
        self.assertGreater(available, 0)
        self.assertIsInstance(available, (int, float))

    def test_get_memory_usage_bytes(self):
        """Test getting current memory usage."""
        usage = self.manager.get_memory_usage_bytes()

        # Should return a positive number
        self.assertGreater(usage, 0)
        self.assertIsInstance(usage, (int, float))

    def test_get_memory_usage_fraction(self):
        """Test getting memory usage fraction."""
        fraction = self.manager.get_memory_usage_fraction()

        # Should be between 0 and 1
        self.assertGreaterEqual(fraction, 0)
        self.assertLessEqual(fraction, 1)


class TestMemoryPrediction(unittest.TestCase):
    """Tests for memory requirement prediction."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def test_predict_memory_trf(self):
        """Test memory prediction for TRF algorithm."""
        n_points = 1000
        n_params = 10

        memory = self.manager.predict_memory_requirement(n_points, n_params, "trf")

        # Should return positive memory requirement
        self.assertGreater(memory, 0)

        # Should scale roughly with n_points * n_params
        expected_min = 8 * n_points * n_params  # At least Jacobian
        self.assertGreater(memory, expected_min)

    def test_predict_memory_lm(self):
        """Test memory prediction for LM algorithm."""
        n_points = 1000
        n_params = 10

        memory = self.manager.predict_memory_requirement(n_points, n_params, "lm")

        # Should return positive memory requirement
        self.assertGreater(memory, 0)

    def test_predict_memory_dogbox(self):
        """Test memory prediction for dogbox algorithm."""
        n_points = 1000
        n_params = 10

        memory = self.manager.predict_memory_requirement(n_points, n_params, "dogbox")

        # Should return positive memory requirement
        self.assertGreater(memory, 0)

    def test_predict_memory_unknown_algorithm(self):
        """Test memory prediction for unknown algorithm."""
        n_points = 1000
        n_params = 10

        memory = self.manager.predict_memory_requirement(n_points, n_params, "unknown")

        # Should still return a conservative estimate
        self.assertGreater(memory, 0)

    def test_predict_memory_scaling(self):
        """Test that memory prediction scales correctly."""
        n_params = 10

        memory_1k = self.manager.predict_memory_requirement(1000, n_params)
        memory_10k = self.manager.predict_memory_requirement(10000, n_params)

        # Should scale approximately linearly with n_points
        self.assertGreater(memory_10k, memory_1k * 5)
        self.assertLess(memory_10k, memory_1k * 15)

    def test_safety_factor_applied(self):
        """Test that safety factor is applied to predictions."""
        manager1 = MemoryManager(safety_factor=1.0)
        manager2 = MemoryManager(safety_factor=2.0)

        memory1 = manager1.predict_memory_requirement(1000, 10)
        memory2 = manager2.predict_memory_requirement(1000, 10)

        # Should be approximately 2x with 2x safety factor
        self.assertAlmostEqual(memory2, memory1 * 2.0, delta=memory1 * 0.1)


class TestMemoryAvailability(unittest.TestCase):
    """Tests for memory availability checking."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def test_check_memory_availability_sufficient(self):
        """Test checking availability with sufficient memory."""
        bytes_needed = 1024  # 1 KB (very small)

        available, message = self.manager.check_memory_availability(bytes_needed)

        self.assertTrue(available)
        self.assertIn("available", message.lower())

    def test_check_memory_availability_insufficient(self):
        """Test checking availability with insufficient memory."""
        bytes_needed = 1e15  # 1 PB (way too much)

        available, message = self.manager.check_memory_availability(bytes_needed)

        self.assertFalse(available)
        self.assertIn("insufficient", message.lower())


class TestMemoryPooling(unittest.TestCase):
    """Tests for array pooling functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after each test."""
        self.manager.clear_pool()

    def test_allocate_array_basic(self):
        """Test basic array allocation."""
        shape = (100, 10)
        arr = self.manager.allocate_array(shape)

        # Check array properties
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.dtype, np.float64)

        # Should be zeroed by default
        np.testing.assert_array_equal(arr, np.zeros(shape))

    def test_allocate_array_custom_dtype(self):
        """Test allocation with custom dtype."""
        shape = (50, 20)
        arr = self.manager.allocate_array(shape, dtype=np.float32)

        self.assertEqual(arr.dtype, np.float32)

    def test_allocate_array_no_zero(self):
        """Test allocation without zero initialization."""
        shape = (100, 10)
        arr = self.manager.allocate_array(shape, zero=False)

        # Array should exist but not necessarily be zeros
        self.assertEqual(arr.shape, shape)

    def test_array_pooling_reuse(self):
        """Test that arrays are reused from pool."""
        shape = (100, 10)

        # Allocate first time
        arr1 = self.manager.allocate_array(shape)
        arr1_id = id(arr1)

        # Return to pool
        self.manager.free_array(arr1)

        # Allocate again with same shape
        arr2 = self.manager.allocate_array(shape)
        arr2_id = id(arr2)

        # Should be the same array object
        self.assertEqual(arr1_id, arr2_id)

    def test_array_pooling_different_shapes(self):
        """Test pooling with different shapes."""
        self.manager.allocate_array((100, 10))
        self.manager.allocate_array((50, 20))

        # Pool should contain both arrays
        self.assertEqual(len(self.manager.memory_pool), 2)

    def test_clear_pool(self):
        """Test clearing memory pool."""
        # Allocate some arrays
        self.manager.allocate_array((100, 10))
        self.manager.allocate_array((50, 20))

        self.assertGreater(len(self.manager.memory_pool), 0)

        # Clear pool
        self.manager.clear_pool()

        self.assertEqual(len(self.manager.memory_pool), 0)

    def test_free_array(self):
        """Test returning array to pool."""
        arr = np.random.randn(100, 10)
        shape = arr.shape
        dtype = arr.dtype

        # Free the array
        self.manager.free_array(arr)

        # Pool should contain it
        key = (shape, dtype)
        self.assertIn(key, self.manager.memory_pool)

    def test_optimize_memory_pool(self):
        """Test memory pool optimization."""
        # Allocate many arrays of different sizes
        for i in range(10):
            size = (i + 1) * 10
            self.manager.allocate_array((size, size))

        # Pool should have 10 arrays
        self.assertEqual(len(self.manager.memory_pool), 10)

        # Optimize to keep only 5
        self.manager.optimize_memory_pool(max_arrays=5)

        # Should keep only 5 largest arrays
        self.assertEqual(len(self.manager.memory_pool), 5)

    def test_optimize_memory_pool_no_op(self):
        """Test pool optimization when under limit."""
        # Clear pool first to start fresh
        self.manager.clear_pool()

        # Allocate only 3 arrays with different shapes
        self.manager.allocate_array((10, 10))
        self.manager.allocate_array((20, 20))
        self.manager.allocate_array((30, 30))

        pool_size_before = len(self.manager.memory_pool)

        # Optimize to keep 5 (more than we have)
        self.manager.optimize_memory_pool(max_arrays=5)

        # Should keep all arrays when under limit
        self.assertGreaterEqual(len(self.manager.memory_pool), min(3, pool_size_before))


class TestMemoryGuard(unittest.TestCase):
    """Tests for memory_guard context manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after each test."""
        self.manager.clear_pool()

    def test_memory_guard_success(self):
        """Test memory guard with available memory."""
        bytes_needed = 1024  # 1 KB

        with self.manager.memory_guard(bytes_needed):
            # Should work without raising
            pass

        # Should track allocation
        self.assertEqual(len(self.manager.allocation_history), 1)

    def test_memory_guard_failure(self):
        """Test memory guard with insufficient memory."""
        bytes_needed = 1e15  # 1 PB

        with (
            self.assertRaises(MemoryError),
            self.manager.memory_guard(bytes_needed),
        ):
            pass

    def test_memory_guard_tracks_peak(self):
        """Test that memory guard tracks peak memory."""
        initial_peak = self.manager._peak_memory

        bytes_needed = 1024
        with self.manager.memory_guard(bytes_needed):
            # Allocate something
            _ = np.zeros((100, 100))

        # Peak should be updated
        self.assertGreaterEqual(self.manager._peak_memory, initial_peak)

    @patch("nlsq.caching.memory_manager.gc.collect")
    def test_memory_guard_triggers_gc(self, mock_gc):
        """Test that memory guard triggers GC when threshold exceeded."""
        # Set low threshold
        manager = MemoryManager(gc_threshold=0.0)  # Always trigger GC

        bytes_needed = 1024
        with manager.memory_guard(bytes_needed):
            pass

        # GC should have been called
        mock_gc.assert_called()


class TestTemporaryAllocation(unittest.TestCase):
    """Tests for temporary_allocation context manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after each test."""
        self.manager.clear_pool()

    def test_temporary_allocation_basic(self):
        """Test basic temporary allocation."""
        shape = (100, 10)

        with self.manager.temporary_allocation(shape) as arr:
            # Array should be allocated
            self.assertEqual(arr.shape, shape)
            arr_id = id(arr)

        # After exit, array should be in pool
        key = (shape, arr.dtype)
        self.assertIn(key, self.manager.memory_pool)
        self.assertEqual(id(self.manager.memory_pool[key]), arr_id)

    def test_temporary_allocation_reuse(self):
        """Test that temporary allocations are reused."""
        shape = (100, 10)

        with self.manager.temporary_allocation(shape) as arr1:
            arr1_id = id(arr1)

        with self.manager.temporary_allocation(shape) as arr2:
            arr2_id = id(arr2)

        # Should reuse the same array
        self.assertEqual(arr1_id, arr2_id)

    def test_temporary_allocation_exception(self):
        """Test temporary allocation with exception."""
        shape = (100, 10)

        try:
            with self.manager.temporary_allocation(shape) as arr:
                arr_id = id(arr)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Array should still be returned to pool despite exception
        key = (shape, arr.dtype)
        self.assertIn(key, self.manager.memory_pool)
        self.assertEqual(id(self.manager.memory_pool[key]), arr_id)


class TestMemoryStats(unittest.TestCase):
    """Tests for memory statistics."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after each test."""
        self.manager.clear_pool()

    def test_get_memory_stats_basic(self):
        """Test getting basic memory statistics."""
        stats = self.manager.get_memory_stats()

        # Check required fields
        required_fields = [
            "current_usage_gb",
            "available_gb",
            "peak_usage_gb",
            "usage_fraction",
            "pool_memory_gb",
            "pool_arrays",
            "allocations",
        ]

        for field in required_fields:
            self.assertIn(field, stats)

    def test_get_memory_stats_with_allocations(self):
        """Test statistics after allocations."""
        # Make some allocations
        with self.manager.memory_guard(1024):
            pass

        with self.manager.memory_guard(2048):
            pass

        stats = self.manager.get_memory_stats()

        # Should have tracked allocations
        self.assertEqual(stats["allocations"], 2)
        self.assertIn("total_requested_gb", stats)
        self.assertIn("total_used_gb", stats)
        self.assertIn("efficiency", stats)

    def test_get_memory_stats_pool_tracking(self):
        """Test that pool memory is tracked."""
        # Allocate arrays
        self.manager.allocate_array((100, 100))
        self.manager.allocate_array((50, 50))

        stats = self.manager.get_memory_stats()

        # Should track pool size
        self.assertEqual(stats["pool_arrays"], 2)
        self.assertGreater(stats["pool_memory_gb"], 0)


class TestChunkingStrategy(unittest.TestCase):
    """Tests for chunking strategy estimation."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def test_chunking_not_needed(self):
        """Test when chunking is not needed."""
        n_points = 100
        n_params = 5
        memory_limit_gb = 10.0  # Plenty of memory

        strategy = self.manager.estimate_chunking_strategy(
            n_points, n_params, memory_limit_gb=memory_limit_gb
        )

        # Should not need chunking
        self.assertFalse(strategy["needs_chunking"])
        self.assertEqual(strategy["chunk_size"], n_points)
        self.assertEqual(strategy["n_chunks"], 1)

    def test_chunking_needed(self):
        """Test when chunking is needed."""
        n_points = 1_000_000
        n_params = 100
        memory_limit_gb = 0.01  # Very limited memory

        strategy = self.manager.estimate_chunking_strategy(
            n_points, n_params, memory_limit_gb=memory_limit_gb
        )

        # Should need chunking
        self.assertTrue(strategy["needs_chunking"])
        self.assertLess(strategy["chunk_size"], n_points)
        self.assertGreater(strategy["n_chunks"], 1)
        self.assertEqual(strategy["total_points"], n_points)

    def test_chunking_strategy_fields(self):
        """Test that chunking strategy has all required fields."""
        strategy = self.manager.estimate_chunking_strategy(1000, 10)

        required_fields = [
            "needs_chunking",
            "chunk_size",
            "n_chunks",
            "memory_per_chunk_gb",
        ]

        for field in required_fields:
            self.assertIn(field, strategy)

    def test_chunking_minimum_chunk_size(self):
        """Test that chunk size has some reasonable minimum."""
        n_points = 1000
        n_params = 5
        memory_limit_gb = 0.00001  # Extremely limited

        strategy = self.manager.estimate_chunking_strategy(
            n_points, n_params, memory_limit_gb=memory_limit_gb
        )

        # Should have some reasonable minimum chunk size (algorithm may vary)
        # The actual minimum depends on implementation
        self.assertGreater(strategy["chunk_size"], 0)
        self.assertTrue(strategy["needs_chunking"])


class TestGlobalFunctions(unittest.TestCase):
    """Tests for global convenience functions."""

    def test_get_memory_manager(self):
        """Test getting global memory manager."""
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()

        # Should return the same instance
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, MemoryManager)

    def test_clear_memory_pool_global(self):
        """Test clearing global memory pool."""
        manager = get_memory_manager()

        # Allocate something
        manager.allocate_array((100, 10))
        self.assertGreater(len(manager.memory_pool), 0)

        # Clear via global function
        clear_memory_pool()

        self.assertEqual(len(manager.memory_pool), 0)

    def test_get_memory_stats_global(self):
        """Test getting global memory stats."""
        stats = get_memory_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn("current_usage_gb", stats)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after each test."""
        self.manager.clear_pool()

    def test_allocate_zero_size_array(self):
        """Test allocating zero-size array."""
        shape = (0, 0)
        arr = self.manager.allocate_array(shape)

        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.size, 0)

    def test_predict_memory_zero_params(self):
        """Test memory prediction with zero parameters."""
        memory = self.manager.predict_memory_requirement(1000, 0)

        # Should handle gracefully
        self.assertGreater(memory, 0)

    def test_predict_memory_zero_points(self):
        """Test memory prediction with zero points."""
        memory = self.manager.predict_memory_requirement(0, 10)

        # Should handle gracefully
        self.assertGreater(memory, 0)


class TestPropertyBasedMemoryManager(unittest.TestCase):
    """Property-based tests for memory manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after each test."""
        self.manager.clear_pool()

    @given(
        n_points=st.integers(min_value=1, max_value=100000),
        n_params=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_memory_prediction_always_positive(self, n_points, n_params):
        """Test that memory predictions are always positive."""
        memory = self.manager.predict_memory_requirement(n_points, n_params)

        self.assertGreater(memory, 0)
        self.assertIsInstance(memory, int)

    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),
            st.integers(min_value=1, max_value=100),
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_array_allocation_correct_shape(self, shape):
        """Test that allocated arrays have correct shape."""
        arr = self.manager.allocate_array(shape)

        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.dtype, np.float64)

    @given(
        n_points=st.integers(min_value=100, max_value=10000),
        n_params=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30, deadline=None)
    def test_chunking_strategy_valid(self, n_points, n_params):
        """Test that chunking strategy is always valid."""
        strategy = self.manager.estimate_chunking_strategy(
            n_points, n_params, memory_limit_gb=0.1
        )

        # Check validity
        self.assertIn("needs_chunking", strategy)
        self.assertIn("chunk_size", strategy)
        self.assertIn("n_chunks", strategy)

        # Check that chunks cover all points
        chunk_size = strategy["chunk_size"]
        n_chunks = strategy["n_chunks"]
        self.assertGreaterEqual(chunk_size * n_chunks, n_points)


class TestMixedPrecisionIntegration(unittest.TestCase):
    """Tests for mixed precision integration with memory manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after each test."""
        self.manager.clear_pool()

    def test_predict_memory_float32(self):
        """Test memory prediction with float32 dtype."""
        import jax.numpy as jnp

        n_points = 1000
        n_params = 10

        memory_float32 = self.manager.predict_memory_requirement(
            n_points, n_params, "trf", dtype=jnp.float32
        )

        # Should return positive memory requirement
        self.assertGreater(memory_float32, 0)

        # Should scale with float32 size (4 bytes)
        expected_min = 4 * n_points * n_params  # At least Jacobian
        self.assertGreater(memory_float32, expected_min)

    def test_predict_memory_float64(self):
        """Test memory prediction with float64 dtype (default)."""
        import jax.numpy as jnp

        n_points = 1000
        n_params = 10

        memory_float64 = self.manager.predict_memory_requirement(
            n_points, n_params, "trf", dtype=jnp.float64
        )

        # Should return positive memory requirement
        self.assertGreater(memory_float64, 0)

        # Should scale with float64 size (8 bytes)
        expected_min = 8 * n_points * n_params  # At least Jacobian
        self.assertGreater(memory_float64, expected_min)

    def test_memory_difference_float32_vs_float64(self):
        """Test that float32 uses ~50% memory compared to float64."""
        import jax.numpy as jnp

        n_points = 10000
        n_params = 50

        memory_float32 = self.manager.predict_memory_requirement(
            n_points, n_params, "trf", dtype=jnp.float32
        )
        memory_float64 = self.manager.predict_memory_requirement(
            n_points, n_params, "trf", dtype=jnp.float64
        )

        # float32 should use approximately 50% of float64 memory
        ratio = memory_float32 / memory_float64
        self.assertAlmostEqual(ratio, 0.5, delta=0.01)

    def test_get_current_precision_memory_multiplier_no_manager(self):
        """Test memory multiplier without mixed precision manager."""
        multiplier = self.manager.get_current_precision_memory_multiplier(None)

        # Should default to 1.0 (float64)
        self.assertEqual(multiplier, 1.0)

    def test_get_current_precision_memory_multiplier_with_manager_float32(self):
        """Test memory multiplier with mixed precision manager in float32."""

        from nlsq.precision.mixed_precision import (
            MixedPrecisionConfig,
            MixedPrecisionManager,
        )

        # Create manager starting in float32
        mp_manager = MixedPrecisionManager(MixedPrecisionConfig())

        # Initial state should be FLOAT32_ACTIVE, multiplier should be 0.5
        multiplier = self.manager.get_current_precision_memory_multiplier(mp_manager)
        self.assertEqual(multiplier, 0.5)

    def test_get_current_precision_memory_multiplier_with_manager_float64(self):
        """Test memory multiplier with mixed precision manager in float64."""
        import jax.numpy as jnp

        from nlsq.precision.mixed_precision import (
            MixedPrecisionConfig,
            MixedPrecisionManager,
        )

        # Create manager and manually set it to float64
        # (simulating after upgrade)
        mp_manager = MixedPrecisionManager(MixedPrecisionConfig())
        mp_manager.current_dtype = jnp.float64

        # Should return 1.0 for float64
        multiplier = self.manager.get_current_precision_memory_multiplier(mp_manager)
        self.assertEqual(multiplier, 1.0)

    def test_predict_memory_default_dtype(self):
        """Test that default dtype is float64."""
        import jax.numpy as jnp

        n_points = 1000
        n_params = 10

        # Call without dtype parameter
        memory_default = self.manager.predict_memory_requirement(
            n_points, n_params, "trf"
        )

        # Call with explicit float64
        memory_float64 = self.manager.predict_memory_requirement(
            n_points, n_params, "trf", dtype=jnp.float64
        )

        # Should be identical
        self.assertEqual(memory_default, memory_float64)


if __name__ == "__main__":
    unittest.main()
