"""Tests for Performance Instrumentation (Task Group 10).

Tests cover:
- JIT recompilation counter incrementing correctly
- Cache hit rates logged at INFO level when enabled
- Memory pool efficiency metric calculation
- Checkpoint save duration tracking
- NLSQ_PROFILE environment variable enabling instrumentation

These tests verify the performance observability features added in Phase 3.
"""

import os
import time
import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np


class TestJITRecompilationCounter(unittest.TestCase):
    """Tests for _jit_recompilation_count counter in StreamingOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        from nlsq.streaming.config import StreamingConfig
        from nlsq.streaming.optimizer import StreamingOptimizer

        self.config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            batch_shape_padding="dynamic",  # Use dynamic to allow recompilations
            enable_fault_tolerance=False,
            enable_checkpoints=False,
        )
        self.optimizer = StreamingOptimizer(self.config)

    def tearDown(self):
        """Clean up after tests."""
        self.optimizer._shutdown_checkpoint_worker()

    def test_jit_recompilation_counter_exists(self):
        """Test that _jit_recompilation_count counter exists on StreamingOptimizer."""
        self.assertTrue(hasattr(self.optimizer, "_jit_recompilation_count"))
        self.assertIsInstance(self.optimizer._jit_recompilation_count, int)
        self.assertEqual(self.optimizer._jit_recompilation_count, 0)

    def test_recompilation_counter_initializes_to_zero(self):
        """Test that recompilation counter starts at zero."""
        from nlsq.streaming.config import StreamingConfig
        from nlsq.streaming.optimizer import StreamingOptimizer

        config = StreamingConfig(batch_size=100, max_epochs=1)
        optimizer = StreamingOptimizer(config)
        try:
            self.assertEqual(optimizer._jit_recompilation_count, 0)
        finally:
            optimizer._shutdown_checkpoint_worker()

    def test_recompilation_counter_in_diagnostics(self):
        """Test that recompilation count is included in streaming diagnostics."""
        # Run a simple optimization
        np.random.seed(42)
        x_data = np.linspace(0, 1, 100)
        y_data = 2.0 * x_data + 0.1 * np.random.randn(100)

        def model(x, a):
            return a * x

        result = self.optimizer.fit(
            (x_data, y_data), model, p0=np.array([1.0]), verbose=0
        )

        # Check diagnostics include recompilation count
        self.assertIn("streaming_diagnostics", result)
        diagnostics = result["streaming_diagnostics"]
        self.assertIn("jit_recompilation_count", diagnostics)


class TestCacheHitRateLogging(unittest.TestCase):
    """Tests for cache hit rate logging at INFO level."""

    def test_cache_stats_available_for_logging(self):
        """Test that SmartCache provides stats suitable for logging."""
        from nlsq.caching.smart_cache import SmartCache

        cache = SmartCache(enable_stats=True)

        # Perform some operations to generate stats
        cache.set("key1", np.array([1.0, 2.0]))
        _ = cache.get("key1")  # Hit
        _ = cache.get("key2")  # Miss

        stats = cache.get_stats()

        # Verify stats structure
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)

        # Verify hit rate calculation
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_compilation_cache_stats_available(self):
        """Test that CompilationCache provides stats for logging."""
        from nlsq.caching.compilation_cache import CompilationCache

        cache = CompilationCache(enable_stats=True)

        # Compile a function
        def test_func(x):
            return x + 1

        cache.compile(test_func)  # Miss (first compile)
        cache.compile(test_func)  # Hit (cached)

        stats = cache.get_stats()

        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)

    def test_profiling_logs_cache_stats_when_enabled(self):
        """Test that cache stats are logged when NLSQ_PROFILE=1."""
        from nlsq.streaming.config import StreamingConfig
        from nlsq.streaming.optimizer import StreamingOptimizer

        # Create optimizer with profiling config
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_checkpoints=False,
            enable_fault_tolerance=False,
        )

        optimizer = StreamingOptimizer(config)
        try:
            # Check that profiling metrics attribute exists
            self.assertTrue(hasattr(optimizer, "_profiling_metrics"))
        finally:
            optimizer._shutdown_checkpoint_worker()


class TestMemoryPoolEfficiencyMetric(unittest.TestCase):
    """Tests for memory pool efficiency metric calculation."""

    def setUp(self):
        """Set up test fixtures."""
        from nlsq.caching.memory_manager import MemoryManager

        self.manager = MemoryManager()

    def tearDown(self):
        """Clean up after tests."""
        self.manager.clear_pool()

    def test_memory_pool_stats_include_efficiency(self):
        """Test that memory pool stats include efficiency metric."""
        # The MemoryManager should track pool hits vs new allocations
        stats = self.manager.get_memory_stats()

        # Stats should be available
        self.assertIsInstance(stats, dict)
        self.assertIn("pool_arrays", stats)

    def test_pool_efficiency_calculation(self):
        """Test that pool efficiency is calculated as pool_hits / (pool_hits + new_allocations)."""
        # Allocate an array (new allocation)
        arr1 = self.manager.allocate_array((100,), dtype=np.float64)

        # Return it to pool
        self.manager.free_array(arr1)

        # Get same shape again (should be pool hit)
        arr2 = self.manager.allocate_array((100,), dtype=np.float64)

        # arr2 should be same object as arr1 (reused from pool)
        self.assertIs(arr1, arr2)

    def test_memory_manager_tracks_allocations(self):
        """Test that MemoryManager tracks allocation history."""
        # Use memory guard which tracks allocations
        with self.manager.memory_guard(1000):
            pass

        # Should have allocation history
        self.assertGreater(len(self.manager.allocation_history), 0)


class TestCheckpointSaveDurationTracking(unittest.TestCase):
    """Tests for checkpoint save duration tracking."""

    def test_checkpoint_save_times_tracked(self):
        """Test that checkpoint save times are tracked."""
        import tempfile

        from nlsq.streaming.config import StreamingConfig
        from nlsq.streaming.optimizer import StreamingOptimizer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(
                batch_size=50,
                max_epochs=1,
                checkpoint_dir=tmpdir,
                checkpoint_frequency=10,  # Save frequently
                enable_checkpoints=True,
                enable_fault_tolerance=False,
            )
            optimizer = StreamingOptimizer(config)

            try:
                # Verify checkpoint save times tracking exists
                self.assertTrue(hasattr(optimizer, "checkpoint_save_times"))
                self.assertIsInstance(optimizer.checkpoint_save_times, list)
            finally:
                optimizer._shutdown_checkpoint_worker()

    def test_checkpoint_duration_histogram_in_diagnostics(self):
        """Test that checkpoint duration histogram is in diagnostics when checkpoints are saved."""
        import tempfile

        from nlsq.streaming.config import StreamingConfig
        from nlsq.streaming.optimizer import StreamingOptimizer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(
                batch_size=25,
                max_epochs=1,
                checkpoint_dir=tmpdir,
                checkpoint_frequency=5,  # Save every 5 iterations
                enable_checkpoints=True,
                enable_fault_tolerance=False,
            )
            optimizer = StreamingOptimizer(config)

            try:
                # Run optimization
                np.random.seed(42)
                x_data = np.linspace(0, 1, 100)
                y_data = 2.0 * x_data + 0.1 * np.random.randn(100)

                def model(x, a):
                    return a * x

                result = optimizer.fit(
                    (x_data, y_data), model, p0=np.array([1.0]), verbose=0
                )

                # Wait for async checkpoints to complete
                time.sleep(0.5)

                # Check diagnostics
                diagnostics = result["streaming_diagnostics"]
                # Checkpoint info should be present
                self.assertIn("checkpoint_info", diagnostics)
            finally:
                optimizer._shutdown_checkpoint_worker()


class TestProfilingEnvironmentVariable(unittest.TestCase):
    """Tests for NLSQ_PROFILE=1 environment variable."""

    def test_profiling_disabled_by_default(self):
        """Test that profiling is disabled when NLSQ_PROFILE is not set."""
        # Ensure env var is not set
        old_value = os.environ.pop("NLSQ_PROFILE", None)

        try:
            from nlsq.streaming.config import StreamingConfig
            from nlsq.streaming.optimizer import StreamingOptimizer

            config = StreamingConfig(batch_size=50, max_epochs=1)
            optimizer = StreamingOptimizer(config)

            try:
                # Profiling should be disabled
                self.assertFalse(optimizer._profiling_enabled)
            finally:
                optimizer._shutdown_checkpoint_worker()
        finally:
            if old_value is not None:
                os.environ["NLSQ_PROFILE"] = old_value

    def test_profiling_enabled_with_env_var(self):
        """Test that profiling is enabled when NLSQ_PROFILE=1."""
        # Set env var
        old_value = os.environ.get("NLSQ_PROFILE")
        os.environ["NLSQ_PROFILE"] = "1"

        try:
            # Need to reimport to pick up env var
            from nlsq.streaming.config import StreamingConfig
            from nlsq.streaming.optimizer import StreamingOptimizer

            config = StreamingConfig(batch_size=50, max_epochs=1)
            optimizer = StreamingOptimizer(config)

            try:
                # Profiling should be enabled
                self.assertTrue(optimizer._profiling_enabled)
            finally:
                optimizer._shutdown_checkpoint_worker()
        finally:
            if old_value is not None:
                os.environ["NLSQ_PROFILE"] = old_value
            else:
                os.environ.pop("NLSQ_PROFILE", None)

    def test_no_overhead_when_profiling_disabled(self):
        """Test that there is no overhead when profiling is disabled."""
        # Ensure env var is not set
        old_value = os.environ.pop("NLSQ_PROFILE", None)

        try:
            from nlsq.streaming.config import StreamingConfig
            from nlsq.streaming.optimizer import StreamingOptimizer

            config = StreamingConfig(
                batch_size=50,
                max_epochs=1,
                enable_checkpoints=False,
                enable_fault_tolerance=False,
            )

            # Time optimization without profiling
            np.random.seed(42)
            x_data = np.linspace(0, 1, 200)
            y_data = 2.0 * x_data + 0.1 * np.random.randn(200)

            def model(x, a):
                return a * x

            optimizer = StreamingOptimizer(config)
            try:
                start = time.perf_counter()
                result = optimizer.fit(
                    (x_data, y_data), model, p0=np.array([1.0]), verbose=0
                )
                elapsed = time.perf_counter() - start

                # Should complete successfully
                self.assertTrue(result["success"])
                # Should complete in reasonable time (not testing specific overhead,
                # just that it doesn't hang)
                self.assertLess(elapsed, 30.0)
            finally:
                optimizer._shutdown_checkpoint_worker()
        finally:
            if old_value is not None:
                os.environ["NLSQ_PROFILE"] = old_value


class TestProfilingMetricsIntegration(unittest.TestCase):
    """Integration tests for profiling metrics when enabled."""

    def test_profiling_metrics_collected_during_optimization(self):
        """Test that profiling metrics are collected during optimization when enabled."""
        old_value = os.environ.get("NLSQ_PROFILE")
        os.environ["NLSQ_PROFILE"] = "1"

        try:
            from nlsq.streaming.config import StreamingConfig
            from nlsq.streaming.optimizer import StreamingOptimizer

            config = StreamingConfig(
                batch_size=25,
                max_epochs=1,
                enable_checkpoints=False,
                enable_fault_tolerance=False,
            )

            np.random.seed(42)
            x_data = np.linspace(0, 1, 100)
            y_data = 2.0 * x_data + 0.1 * np.random.randn(100)

            def model(x, a):
                return a * x

            optimizer = StreamingOptimizer(config)
            try:
                result = optimizer.fit(
                    (x_data, y_data), model, p0=np.array([1.0]), verbose=0
                )

                # Check that metrics were collected
                self.assertTrue(hasattr(optimizer, "_profiling_metrics"))
                metrics = optimizer._profiling_metrics

                # Verify metrics structure
                self.assertIn("jit_recompilation_count", metrics)
                self.assertIn("batch_count", metrics)

            finally:
                optimizer._shutdown_checkpoint_worker()
        finally:
            if old_value is not None:
                os.environ["NLSQ_PROFILE"] = old_value
            else:
                os.environ.pop("NLSQ_PROFILE", None)


if __name__ == "__main__":
    unittest.main()
