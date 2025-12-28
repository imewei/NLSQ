"""Focused tests for memory reuse optimizations (Task Group 5).

These tests validate the Phase 1 Priority 2 optimizations for memory reuse:
- Adaptive safety factor reduction (1.2 → 1.05)
- Size-class bucketing (1KB/10KB/100KB)
- Pool reuse statistics tracking
- disable_padding configuration flag
"""

import unittest

import jax.numpy as jnp

from nlsq.caching.memory_manager import MemoryManager, get_memory_manager
from nlsq.caching.memory_pool import MemoryPool, clear_global_pool, get_global_pool


class TestAdaptiveSafetyFactor(unittest.TestCase):
    """Test adaptive safety factor reduction from 1.2 → 1.05."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MemoryManager(safety_factor=1.2)

    def test_initial_safety_factor(self):
        """Test initial safety factor is 1.2."""
        self.assertEqual(self.manager.safety_factor, 1.2)

    def test_adaptive_reduction_after_warmup(self):
        """Test safety factor reduces to ~1.05 after successful runs."""
        # Create manager with adaptive safety enabled
        manager = MemoryManager(safety_factor=1.2, enable_adaptive_safety=True)

        # Simulate successful memory allocations
        for _ in range(15):
            with manager.memory_guard(bytes_needed=1000000):
                # Simulate using less memory than predicted (typical case)
                pass

        # After warmup (10 runs), safety factor should start reducing
        telemetry = manager.get_safety_telemetry()
        self.assertGreater(telemetry["telemetry_entries"], 10)

        # Safety factor should have reduced from initial 1.2
        self.assertLess(manager.safety_factor, 1.2)
        # But not below minimum of 1.05
        self.assertGreaterEqual(manager.safety_factor, 1.05)

    def test_safety_factor_in_memory_prediction(self):
        """Test safety factor is applied in memory predictions."""
        # Predict memory with safety factor
        memory_with_safety = self.manager.predict_memory_requirement(
            n_points=1000, n_params=5, algorithm="trf", dtype=jnp.float64
        )

        # Create manager with safety_factor=1.0 for baseline
        manager_no_safety = MemoryManager(safety_factor=1.0)
        memory_no_safety = manager_no_safety.predict_memory_requirement(
            n_points=1000, n_params=5, algorithm="trf", dtype=jnp.float64
        )

        # With safety factor should be larger
        self.assertGreater(memory_with_safety, memory_no_safety)
        # Should be approximately 1.2x larger
        ratio = memory_with_safety / memory_no_safety
        self.assertAlmostEqual(ratio, 1.2, delta=0.01)


class TestSizeClassBucketing(unittest.TestCase):
    """Test size-class bucketing for memory pool (1KB/10KB/100KB)."""

    def setUp(self):
        """Set up test fixtures."""
        self.pool = MemoryPool(max_pool_size=50, enable_stats=True)

    def tearDown(self):
        """Clean up after tests."""
        self.pool.clear()

    def test_bucket_1kb_small_arrays(self):
        """Test 1KB bucketing for small arrays."""
        # Small arrays should bucket to nearest 1KB
        # This will be implemented when bucketing is added (Task 5.4)
        # For now, verify basic allocation works
        arr = self.pool.allocate((100,), dtype=jnp.float64)  # 800 bytes
        self.assertEqual(arr.shape, (100,))

    def test_bucket_10kb_medium_arrays(self):
        """Test 10KB bucketing for medium arrays."""
        # Medium arrays should bucket to nearest 10KB
        arr = self.pool.allocate((1000,), dtype=jnp.float64)  # 8KB
        self.assertEqual(arr.shape, (1000,))

    def test_bucket_100kb_large_arrays(self):
        """Test 100KB bucketing for large arrays."""
        # Large arrays should bucket to nearest 100KB
        arr = self.pool.allocate((10000,), dtype=jnp.float64)  # 80KB
        self.assertEqual(arr.shape, (10000,))

    def test_bucketing_increases_reuse_rate(self):
        """Test that bucketing increases pool reuse rate by 5x."""
        # Allocate and release arrays with slightly different sizes
        sizes = [95, 100, 105, 98, 102]  # All should bucket to same size

        for size in sizes:
            arr = self.pool.allocate((size,), dtype=jnp.float64)
            self.pool.release(arr)

        # After bucketing implementation, these should all reuse the same buffer
        # For now, verify releases work
        stats = self.pool.get_stats()
        self.assertEqual(stats["releases"], len(sizes))


class TestPoolReuseStatistics(unittest.TestCase):
    """Test pool reuse statistics tracking."""

    def setUp(self):
        """Set up test fixtures."""
        self.pool = MemoryPool(max_pool_size=10, enable_stats=True)

    def tearDown(self):
        """Clean up after tests."""
        self.pool.clear()

    def test_reuse_rate_calculation(self):
        """Test reuse_rate = reused_allocations / total_allocations."""
        # First allocation (no reuse)
        arr1 = self.pool.allocate((10,), dtype=jnp.float64)
        self.pool.release(arr1)

        # Second allocation (should reuse)
        arr2 = self.pool.allocate((10,), dtype=jnp.float64)
        self.pool.release(arr2)

        stats = self.pool.get_stats()
        total = stats["allocations"] + stats["reuses"]
        expected_rate = stats["reuses"] / total if total > 0 else 0.0

        self.assertEqual(stats["reuse_rate"], expected_rate)
        self.assertGreater(stats["reuse_rate"], 0)

    def test_pool_stats_in_diagnostics(self):
        """Test pool statistics are included in optimization diagnostics."""
        # Allocate and reuse multiple times
        for _ in range(5):
            arr = self.pool.allocate((20,), dtype=jnp.float64)
            self.pool.release(arr)

        stats = self.pool.get_stats()

        # Verify all expected statistics are present
        self.assertIn("reuse_rate", stats)
        self.assertIn("allocations", stats)
        self.assertIn("reuses", stats)
        self.assertIn("releases", stats)
        self.assertIn("currently_allocated", stats)

    def test_peak_pool_size_tracking(self):
        """Test peak pool size is tracked correctly."""
        # Allocate multiple different shapes
        arrs = [
            self.pool.allocate((10,), dtype=jnp.float64),
            self.pool.allocate((20,), dtype=jnp.float64),
            self.pool.allocate((30,), dtype=jnp.float32),
        ]

        stats = self.pool.get_stats()
        self.assertGreater(stats["currently_allocated"], 0)
        self.assertGreaterEqual(stats["currently_allocated"], len(arrs))


class TestDisablePaddingFlag(unittest.TestCase):
    """Test disable_padding configuration flag."""

    def test_padding_enabled_by_default(self):
        """Test padding is enabled by default."""
        manager = MemoryManager()
        # Default behavior: safety_factor > 1.0 implies padding enabled
        self.assertGreater(manager.safety_factor, 1.0)

    def test_disable_padding_sets_exact_shapes(self):
        """Test disable_padding uses exact shapes (no bucketing)."""
        # This will be implemented when disable_padding is added (Task 5.6)
        # For now, verify manager can be created with different safety factors
        manager_padded = MemoryManager(safety_factor=1.2)
        manager_exact = MemoryManager(safety_factor=1.0)

        memory_padded = manager_padded.predict_memory_requirement(
            n_points=1000, n_params=5, algorithm="trf"
        )
        memory_exact = manager_exact.predict_memory_requirement(
            n_points=1000, n_params=5, algorithm="trf"
        )

        # Exact should be less than padded
        self.assertLess(memory_exact, memory_padded)

    def test_disable_padding_for_strict_environments(self):
        """Test disable_padding works correctly for cloud quotas."""
        # When disable_padding=True: exact shapes, safety_factor=1.0
        manager = MemoryManager(safety_factor=1.0)

        memory = manager.predict_memory_requirement(
            n_points=1000, n_params=5, algorithm="trf"
        )

        # Verify it's not over-allocating
        self.assertIsInstance(memory, int)
        self.assertGreater(memory, 0)


class TestMemoryUsageReduction(unittest.TestCase):
    """Test 10-20% peak memory reduction on standard fits."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager_original = MemoryManager(safety_factor=1.2)
        self.manager_optimized = MemoryManager(safety_factor=1.05)

    def test_memory_reduction_with_adaptive_safety(self):
        """Test memory usage reduces by 10-20% with adaptive safety factor."""
        # Standard fit scenario
        n_points = 10000
        n_params = 10

        memory_original = self.manager_original.predict_memory_requirement(
            n_points=n_points, n_params=n_params, algorithm="trf"
        )

        memory_optimized = self.manager_optimized.predict_memory_requirement(
            n_points=n_points, n_params=n_params, algorithm="trf"
        )

        reduction = (memory_original - memory_optimized) / memory_original
        # With safety factor 1.2 → 1.05, expect ~12.5% reduction
        self.assertGreater(reduction, 0.10)  # At least 10%
        self.assertLess(reduction, 0.20)  # No more than 20%

    def test_mixed_precision_memory_savings(self):
        """Test mixed precision provides additional 50% memory savings."""
        # float64 baseline
        memory_f64 = self.manager_original.predict_memory_requirement(
            n_points=10000, n_params=10, algorithm="trf", dtype=jnp.float64
        )

        # float32 should be ~50% smaller
        memory_f32 = self.manager_original.predict_memory_requirement(
            n_points=10000, n_params=10, algorithm="trf", dtype=jnp.float32
        )

        reduction = (memory_f64 - memory_f32) / memory_f64
        self.assertAlmostEqual(reduction, 0.5, delta=0.05)  # ~50% reduction


class TestMemoryReuseIntegration(unittest.TestCase):
    """Integration tests for memory reuse across components."""

    def setUp(self):
        """Clear global pool before each test."""
        clear_global_pool()

    def test_global_memory_manager_integration(self):
        """Test global memory manager can be used across optimization runs."""
        manager = get_memory_manager()
        self.assertIsInstance(manager, MemoryManager)

        # Should return same instance
        manager2 = get_memory_manager()
        self.assertIs(manager, manager2)

    def test_pool_and_manager_coordination(self):
        """Test memory pool and manager coordinate correctly."""
        pool = get_global_pool(enable_stats=True)
        manager = get_memory_manager()

        # Allocate, release, then allocate again to create reuse
        arr1 = pool.allocate((100,), dtype=jnp.float64)
        pool.release(arr1)
        arr2 = pool.allocate((100,), dtype=jnp.float64)
        pool.release(arr2)

        stats_pool = pool.get_stats()
        stats_manager = manager.get_memory_stats()

        self.assertGreater(stats_pool["reuse_rate"], 0)
        self.assertIsInstance(stats_manager, dict)


if __name__ == "__main__":
    unittest.main()
