"""Benchmark tests for cache performance with LRU eviction.

Task 8.1: Measure cache hit rates and memory pool efficiency.

This module benchmarks:
- Cache hit rates before/after LRU eviction
- Memory pool efficiency: hits / (hits + new_allocations)
- Compare FIFO vs LRU eviction patterns

Expected results:
- LRU eviction should maintain higher hit rates for hot arrays
- Memory pool efficiency should be 5-10% better with LRU
- Frequently accessed array shapes should remain in pool

Uses pytest-benchmark for consistent measurement methodology.
"""

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nlsq.caching.memory_manager import MemoryManager


def create_psutil_mock():
    """Create a properly configured psutil mock."""
    mock_psutil = MagicMock()

    # Mock virtual_memory
    mock_mem = MagicMock()
    mock_mem.available = 8 * 1024**3  # 8 GB
    mock_mem.percent = 50.0  # 50% usage
    mock_psutil.virtual_memory.return_value = mock_mem

    # Mock Process for memory_info
    mock_process = MagicMock()
    mock_process.memory_info.return_value.rss = 1 * 1024**3  # 1 GB
    mock_psutil.Process.return_value = mock_process

    return mock_psutil


class TestLRUMemoryPoolBenchmark:
    """Benchmark LRU memory pool performance vs FIFO eviction."""

    @pytest.mark.benchmark(group="lru_memory_pool")
    def test_lru_pool_allocation_benchmark(self, benchmark):
        """Benchmark memory pool allocation with LRU tracking.

        Tests allocation of arrays with LRU tracking enabled.
        The OrderedDict-based pool should efficiently track access patterns.
        """
        manager = MemoryManager(adaptive_ttl=True)

        def allocate_arrays():
            # Simulate typical optimization pattern: frequent reuse of certain shapes
            shapes = [
                (100, 10),
                (100, 10),  # Repeated shape (should hit cache)
                (200, 5),
                (100, 10),  # Repeated again
                (50, 20),
                (100, 10),  # Most frequently used shape
            ]
            for shape in shapes:
                manager.allocate_array(shape, dtype=np.float64)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            benchmark(allocate_arrays)

    @pytest.mark.benchmark(group="lru_memory_pool")
    def test_lru_eviction_benchmark(self, benchmark):
        """Benchmark LRU eviction when pool is at capacity.

        Tests that LRU eviction efficiently removes least-recently-used
        arrays while keeping hot arrays in the pool.
        """
        manager = MemoryManager(adaptive_ttl=True)

        def eviction_test():
            # Fill pool with various shapes
            for i in range(50):
                manager.allocate_array((100 + i, 10), dtype=np.float64)

            # Access some shapes frequently (should stay in pool after eviction)
            hot_shapes = [(100, 10), (110, 10), (120, 10)]
            for _ in range(10):
                for shape in hot_shapes:
                    manager.allocate_array(shape, dtype=np.float64)

            # Trigger eviction
            manager.optimize_memory_pool(max_arrays=20)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            benchmark(eviction_test)


class TestMemoryPoolEfficiency:
    """Test memory pool efficiency metrics."""

    def test_lru_pool_hit_rate(self):
        """Measure cache hit rate for memory pool.

        Tests that frequently accessed shapes have high hit rate due to
        LRU tracking moving them to end of OrderedDict.
        """
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            # Track hits and misses
            hits = 0
            misses = 0

            # Simulate workload with repeated shapes
            shapes = [
                (100, 10),  # Shape A
                (200, 5),  # Shape B
                (100, 10),  # Shape A again (hit)
                (300, 3),  # Shape C
                (100, 10),  # Shape A again (hit)
                (200, 5),  # Shape B again (hit)
                (100, 10),  # Shape A again (hit)
                (400, 2),  # Shape D
                (100, 10),  # Shape A again (hit)
            ]

            for shape in shapes:
                key = (shape, np.float64)
                if key in manager.memory_pool:
                    hits += 1
                else:
                    misses += 1
                manager.allocate_array(shape, dtype=np.float64)

            # Calculate hit rate
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0

            print(f"\n[LRU Pool Hit Rate] Hits: {hits}, Misses: {misses}")
            print(f"[LRU Pool Hit Rate] Hit rate: {hit_rate * 100:.1f}%")

            # With repeated shapes, we should have some hits
            assert hits > 0, "Should have cache hits for repeated shapes"
            assert hit_rate > 0.3, f"Expected >30% hit rate, got {hit_rate * 100:.1f}%"

    def test_lru_vs_fifo_simulation(self):
        """Compare LRU vs FIFO eviction behavior.

        Simulates both eviction strategies and compares their
        effectiveness at keeping hot arrays in the pool.
        """
        # LRU simulation using OrderedDict
        lru_pool: OrderedDict = OrderedDict()

        # FIFO simulation using regular dict with list for order
        fifo_pool: dict = {}
        fifo_order: list = []

        max_size = 5

        # Workload: some shapes are accessed more frequently
        workload = [
            (100, 10),  # Hot shape
            (200, 5),
            (300, 3),
            (400, 2),
            (500, 1),
            (100, 10),  # Hot shape accessed again
            (600, 1),  # New shape (triggers eviction)
            (100, 10),  # Hot shape accessed again
            (700, 1),  # New shape (triggers eviction)
            (100, 10),  # Hot shape accessed again
        ]

        lru_hits = 0
        lru_misses = 0
        fifo_hits = 0
        fifo_misses = 0

        for shape in workload:
            key = (shape, np.float64)

            # LRU: Check hit and update
            if key in lru_pool:
                lru_hits += 1
                lru_pool.move_to_end(key)
            else:
                lru_misses += 1
                if len(lru_pool) >= max_size:
                    lru_pool.popitem(last=False)  # Evict LRU
                lru_pool[key] = True

            # FIFO: Check hit and update
            if key in fifo_pool:
                fifo_hits += 1
                # FIFO doesn't update order on access
            else:
                fifo_misses += 1
                if len(fifo_pool) >= max_size:
                    # Evict oldest (first inserted)
                    oldest = fifo_order.pop(0)
                    del fifo_pool[oldest]
                fifo_pool[key] = True
                fifo_order.append(key)

        lru_hit_rate = lru_hits / (lru_hits + lru_misses)
        fifo_hit_rate = fifo_hits / (fifo_hits + fifo_misses)

        print(
            f"\n[LRU vs FIFO] LRU hits: {lru_hits}, misses: {lru_misses}, rate: {lru_hit_rate * 100:.1f}%"
        )
        print(
            f"[LRU vs FIFO] FIFO hits: {fifo_hits}, misses: {fifo_misses}, rate: {fifo_hit_rate * 100:.1f}%"
        )

        # LRU should be at least as good as FIFO for this workload
        # (and typically better for workloads with hot items)
        assert lru_hit_rate >= fifo_hit_rate, (
            f"LRU hit rate ({lru_hit_rate * 100:.1f}%) should be >= FIFO ({fifo_hit_rate * 100:.1f}%)"
        )

    def test_pool_efficiency_metric(self):
        """Calculate memory pool efficiency: hits / (hits + new_allocations).

        This metric shows how well the pool is being utilized to avoid
        redundant memory allocations.
        """
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            hits = 0
            new_allocations = 0

            # Simulate optimization workload
            # Common pattern: Jacobian and gradient arrays are frequently reused
            common_shapes = [(1000, 5), (5, 5), (1000,), (5,)]

            for _ in range(10):
                for shape in common_shapes:
                    key = (shape, np.float64)
                    if key in manager.memory_pool:
                        hits += 1
                    else:
                        new_allocations += 1
                    manager.allocate_array(shape, dtype=np.float64)

            total = hits + new_allocations
            efficiency = hits / total if total > 0 else 0

            print(
                f"\n[Pool Efficiency] Hits: {hits}, New allocations: {new_allocations}"
            )
            print(f"[Pool Efficiency] Efficiency: {efficiency * 100:.1f}%")

            # First iteration should all be misses (4 shapes),
            # subsequent 9 iterations should all be hits (36 hits)
            expected_new = len(common_shapes)  # 4
            expected_hits = len(common_shapes) * 9  # 36
            expected_efficiency = expected_hits / (expected_hits + expected_new)

            assert new_allocations == expected_new, (
                f"Expected {expected_new} new allocations, got {new_allocations}"
            )
            assert hits == expected_hits, f"Expected {expected_hits} hits, got {hits}"
            assert efficiency > 0.8, (
                f"Expected >80% efficiency, got {efficiency * 100:.1f}%"
            )


class TestLRUEvictionPatterns:
    """Test LRU eviction patterns and behavior."""

    def test_lru_preserves_hot_arrays(self):
        """Verify that LRU eviction preserves frequently accessed arrays.

        Hot arrays (frequently accessed) should remain in pool even
        after eviction, while cold arrays (rarely accessed) are evicted.
        """
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            hot_shapes = [(100, 10), (200, 5)]
            cold_shapes = [(300, 3), (400, 2), (500, 1)]

            # Add all arrays to pool
            for shape in hot_shapes + cold_shapes:
                manager.allocate_array(shape, dtype=np.float64)

            # Access hot shapes multiple times (marks as recently used)
            for _ in range(5):
                for shape in hot_shapes:
                    manager.allocate_array(shape, dtype=np.float64)

            # Evict to keep only 3 arrays
            manager.optimize_memory_pool(max_arrays=3)

            # Check that hot shapes are still in pool
            for shape in hot_shapes:
                key = (shape, np.float64)
                assert key in manager.memory_pool, (
                    f"Hot shape {shape} should be preserved after LRU eviction"
                )

            # Pool should have exactly max_arrays entries
            assert len(manager.memory_pool) == 3, (
                f"Pool should have 3 arrays after eviction, has {len(manager.memory_pool)}"
            )

            print(
                f"\n[LRU Preservation] Pool contains: {list(manager.memory_pool.keys())}"
            )

    def test_lru_order_tracking(self):
        """Verify that LRU order is correctly tracked with move_to_end.

        The most recently accessed array should be at the end of the
        OrderedDict, and the least recently used at the beginning.
        """
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            # Add arrays in order
            shapes = [(100, 10), (200, 5), (300, 3)]
            for shape in shapes:
                manager.allocate_array(shape, dtype=np.float64)

            # Access first shape again (should move to end)
            manager.allocate_array((100, 10), dtype=np.float64)

            # Get order of keys
            pool_order = list(manager.memory_pool.keys())

            # First shape should now be at the end
            assert pool_order[-1] == ((100, 10), np.float64), (
                "Most recently accessed shape should be at end"
            )

            # Second shape should now be first (oldest/LRU)
            assert pool_order[0] == ((200, 5), np.float64), (
                "Shape (200, 5) should be LRU (first in OrderedDict)"
            )

            print(f"\n[LRU Order] Pool order: {pool_order}")


class TestPoolBenchmarkSuite:
    """Comprehensive benchmark suite for memory pool operations."""

    @pytest.mark.benchmark(group="pool_operations")
    def test_allocate_performance(self, benchmark):
        """Benchmark raw allocation performance with pooling."""
        manager = MemoryManager(adaptive_ttl=True)

        shape = (1000, 10)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            # Warm up - first allocation
            manager.allocate_array(shape, dtype=np.float64)

            # Benchmark cached allocation
            benchmark(lambda: manager.allocate_array(shape, dtype=np.float64))

    @pytest.mark.benchmark(group="pool_operations")
    def test_free_array_performance(self, benchmark):
        """Benchmark free_array performance (returning to pool)."""
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            arr = np.zeros((1000, 10), dtype=np.float64)

            benchmark(lambda: manager.free_array(arr))

    @pytest.mark.benchmark(group="pool_operations")
    def test_optimize_pool_performance(self, benchmark):
        """Benchmark optimize_memory_pool (LRU eviction) performance."""
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):

            def setup_and_evict():
                # Add many arrays
                for i in range(100):
                    manager.memory_pool[((i, 10), np.float64)] = np.zeros((i + 1, 10))
                # Evict to target size
                manager.optimize_memory_pool(max_arrays=50)

            benchmark(setup_and_evict)
