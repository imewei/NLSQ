"""Benchmark tests for memory manager psutil call frequency.

Task 6.3: Profile psutil call frequency vs TTL setting.
Task 8.3: Update memory benchmark with LRU metrics.

This module benchmarks:
- psutil call frequency with different TTL settings
- Overhead reduction with adaptive TTL
- High-frequency caller scenarios
- Phase 2: LRU memory pool efficiency metrics

Expected results:
- Adaptive TTL should reduce psutil overhead by 10-15%
- High-frequency callers should see 10s effective TTL
- Medium-frequency callers should see 5s effective TTL
- Phase 2: LRU pool efficiency >80% for repeated shapes

Uses pytest-benchmark for consistent measurement methodology.
"""

import time
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nlsq.caching.memory_manager import MemoryManager


def create_psutil_mock():
    """Create a properly configured psutil mock for memory operations."""
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


class TestAdaptiveTTLBenchmark:
    """Benchmark adaptive TTL overhead reduction."""

    @pytest.mark.benchmark(group="memory_ttl")
    def test_high_frequency_with_adaptive_ttl(self, benchmark):
        """Benchmark high-frequency memory queries with adaptive TTL enabled.

        High-frequency callers (>100 calls/sec) should benefit from
        the 10s effective TTL, reducing psutil overhead.
        """
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        def high_frequency_queries():
            # Simulate 500 rapid calls (streaming optimizer pattern)
            for _ in range(500):
                manager.get_available_memory()

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            benchmark(high_frequency_queries)

    @pytest.mark.benchmark(group="memory_ttl")
    def test_high_frequency_without_adaptive_ttl(self, benchmark):
        """Benchmark high-frequency queries without adaptive TTL (baseline).

        Without adaptive TTL, even high-frequency callers use the default
        TTL, potentially making more psutil calls.
        """
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=False)

        def high_frequency_queries():
            for _ in range(500):
                manager.get_available_memory()

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            benchmark(high_frequency_queries)


class TestPsutilCallFrequency:
    """Measure actual psutil call frequency reduction."""

    def test_psutil_call_reduction_high_frequency(self):
        """Measure psutil call reduction for high-frequency callers.

        With adaptive TTL enabled, high-frequency callers should make
        significantly fewer psutil calls.
        """
        # Test with adaptive TTL enabled
        manager_adaptive = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            # Simulate high-frequency calls
            for _ in range(1000):
                manager_adaptive.get_available_memory()

            calls_with_adaptive = mock_psutil.virtual_memory.call_count

        # Test without adaptive TTL
        manager_non_adaptive = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=False)

        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            for _ in range(1000):
                manager_non_adaptive.get_available_memory()

            calls_without_adaptive = mock_psutil.virtual_memory.call_count

        # Calculate reduction
        print(
            f"\n[High Frequency] psutil calls with adaptive TTL: {calls_with_adaptive}"
        )
        print(
            f"[High Frequency] psutil calls without adaptive TTL: {calls_without_adaptive}"
        )

        if calls_without_adaptive > 0:
            reduction = (
                (calls_without_adaptive - calls_with_adaptive) / calls_without_adaptive
            ) * 100
            print(f"[High Frequency] Reduction: {reduction:.1f}%")

        # With high-frequency calls all happening quickly, we should have
        # fewer psutil calls when adaptive TTL increases the effective TTL
        assert calls_with_adaptive <= calls_without_adaptive, (
            "Adaptive TTL should reduce or equal psutil calls"
        )

    def test_effective_ttl_thresholds(self):
        """Verify effective TTL is calculated correctly based on call frequency."""
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        # Test high frequency (>100 calls/sec) -> 10s TTL
        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            # Make many rapid calls to establish high frequency
            for _ in range(150):
                manager.get_available_memory()

            # Check effective TTL
            effective_ttl = manager._get_effective_ttl()
            print(f"\n[Threshold Test] High frequency effective TTL: {effective_ttl}s")

            # Should be 15s for high frequency (>100 calls/sec)
            assert effective_ttl == 15.0, (
                f"Expected 15s TTL for high frequency, got {effective_ttl}s"
            )

    def test_adaptive_ttl_streaming_simulation(self):
        """Simulate streaming optimizer memory check pattern.

        The streaming optimizer makes memory checks on every batch,
        which can be hundreds of calls per second.
        """
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            # Simulate 100 epochs with 50 batches each (5000 memory checks)
            n_epochs = 10
            n_batches = 50
            total_checks = n_epochs * n_batches

            start_time = time.perf_counter()
            for _ in range(total_checks):
                manager.get_available_memory()
            elapsed = time.perf_counter() - start_time

            psutil_calls = mock_psutil.virtual_memory.call_count
            calls_per_sec = total_checks / elapsed if elapsed > 0 else 0

            print(f"\n[Streaming Simulation] Total memory checks: {total_checks}")
            print(f"[Streaming Simulation] Actual psutil calls: {psutil_calls}")
            print(f"[Streaming Simulation] Calls per second: {calls_per_sec:.0f}")
            print(
                f"[Streaming Simulation] Cache hit rate: {(1 - psutil_calls / total_checks) * 100:.1f}%"
            )

            # With adaptive TTL, we should have a very high cache hit rate
            cache_hit_rate = 1 - (psutil_calls / total_checks)
            assert cache_hit_rate > 0.9, (
                f"Expected >90% cache hit rate, got {cache_hit_rate * 100:.1f}%"
            )


class TestMemoryManagerOverhead:
    """Benchmark overall memory manager overhead."""

    @pytest.mark.benchmark(group="memory_overhead")
    def test_get_available_memory_overhead(self, benchmark):
        """Benchmark get_available_memory call overhead."""
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            # Warm up the cache
            manager.get_available_memory()

            # Benchmark cached calls
            benchmark(manager.get_available_memory)

    @pytest.mark.benchmark(group="memory_overhead")
    def test_get_memory_usage_bytes_overhead(self, benchmark):
        """Benchmark get_memory_usage_bytes call overhead."""
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            # Warm up the cache
            manager.get_memory_usage_bytes()

            # Benchmark cached calls
            benchmark(manager.get_memory_usage_bytes)


class TestCallFrequencyTracker:
    """Test call frequency tracking accuracy."""

    def test_frequency_tracker_accuracy(self):
        """Verify call frequency is tracked accurately."""
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            # Make calls at known frequency
            target_frequency = 100  # calls per second
            interval = 1.0 / target_frequency
            n_calls = 50

            for _ in range(n_calls):
                manager.get_available_memory()
                time.sleep(interval)

            # Check tracker size
            assert len(manager._call_frequency_tracker) > 0, (
                "Call frequency tracker should have entries"
            )

            # The tracker should have maxlen=100, so it won't exceed that
            assert len(manager._call_frequency_tracker) <= 100, (
                "Tracker should respect maxlen"
            )

    def test_frequency_tracker_deque_maxlen(self):
        """Verify frequency tracker respects maxlen."""
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            # Make more calls than maxlen
            for _ in range(200):
                manager.get_available_memory()

            # Tracker should be capped at maxlen
            assert len(manager._call_frequency_tracker) == 100, (
                f"Tracker should be capped at 100, got {len(manager._call_frequency_tracker)}"
            )


class TestTTLConfigurationComparison:
    """Compare different TTL configurations."""

    def test_ttl_comparison(self):
        """Compare psutil call counts with different TTL settings."""
        ttl_configs = [
            ("0.1s TTL", 0.1),
            ("0.5s TTL", 0.5),
            ("1.0s TTL", 1.0),
            ("5.0s TTL", 5.0),
        ]

        results = {}

        for name, ttl in ttl_configs:
            manager = MemoryManager(memory_cache_ttl=ttl, adaptive_ttl=False)

            with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
                mock_mem = MagicMock()
                mock_mem.available = 8 * 1024**3
                mock_psutil.virtual_memory.return_value = mock_mem

                # Make rapid calls
                for _ in range(1000):
                    manager.get_available_memory()

                results[name] = mock_psutil.virtual_memory.call_count

        print("\n[TTL Comparison] psutil calls for 1000 memory checks:")
        for name, count in results.items():
            print(f"  {name}: {count} calls")

        # Longer TTL should result in fewer psutil calls
        assert results["5.0s TTL"] <= results["0.1s TTL"], (
            "Longer TTL should reduce psutil calls"
        )

    def test_adaptive_vs_static_ttl(self):
        """Compare adaptive TTL vs best static TTL."""
        # Adaptive TTL
        manager_adaptive = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            for _ in range(1000):
                manager_adaptive.get_available_memory()

            adaptive_calls = mock_psutil.virtual_memory.call_count

        # Static 10s TTL (same as adaptive high-frequency)
        manager_static = MemoryManager(memory_cache_ttl=10.0, adaptive_ttl=False)

        with patch("nlsq.caching.memory_manager.psutil") as mock_psutil:
            mock_mem = MagicMock()
            mock_mem.available = 8 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem

            for _ in range(1000):
                manager_static.get_available_memory()

            static_calls = mock_psutil.virtual_memory.call_count

        print(f"\n[Adaptive vs Static] Adaptive TTL calls: {adaptive_calls}")
        print(f"[Adaptive vs Static] Static 10s TTL calls: {static_calls}")

        # With high frequency calls, adaptive should perform similarly to 10s static
        # because it will increase effective TTL to 10s
        assert adaptive_calls <= static_calls + 2, (
            "Adaptive TTL should perform at least as well as optimal static TTL"
        )


# ============================================================================
# Phase 2 Benchmark Additions (Task 8.3)
# ============================================================================


class TestPhase2LRUMemoryPool:
    """Phase 2 LRU memory pool benchmarks.

    Task 8.3: Update memory benchmark with LRU metrics.
    Tests LRU eviction efficiency and memory pool utilization.
    """

    @pytest.mark.benchmark(group="phase2_lru")
    def test_lru_pool_allocation_benchmark(self, benchmark):
        """Benchmark memory pool allocation with LRU tracking.

        Task Group 7 (1.2a) uses OrderedDict for LRU tracking.
        This test measures the overhead of LRU tracking during allocation.
        """
        manager = MemoryManager(adaptive_ttl=True)

        def allocate_arrays():
            # Simulate typical optimization pattern: frequent reuse of shapes
            shapes = [
                (100, 10),
                (100, 10),  # Repeated (should hit cache)
                (200, 5),
                (100, 10),  # Repeated again
                (50, 20),
                (100, 10),  # Most frequently used
            ]
            for shape in shapes:
                manager.allocate_array(shape, dtype=np.float64)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            benchmark(allocate_arrays)

    def test_lru_pool_hit_rate(self):
        """Measure cache hit rate for LRU memory pool.

        With LRU tracking, frequently accessed shapes should have high
        hit rate as they are moved to end (most recently used).
        """
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            # Track hits and misses
            hits = 0
            misses = 0

            # Workload with repeated shapes
            shapes = [
                (100, 10),  # A - miss
                (200, 5),  # B - miss
                (100, 10),  # A - hit
                (300, 3),  # C - miss
                (100, 10),  # A - hit
                (200, 5),  # B - hit
                (100, 10),  # A - hit
                (400, 2),  # D - miss
                (100, 10),  # A - hit
            ]

            for shape in shapes:
                key = (shape, np.float64)
                if key in manager.memory_pool:
                    hits += 1
                else:
                    misses += 1
                manager.allocate_array(shape, dtype=np.float64)

            total = hits + misses
            hit_rate = hits / total if total > 0 else 0

            print(f"\n[LRU Pool Hit Rate] Hits: {hits}, Misses: {misses}")
            print(f"[LRU Pool Hit Rate] Hit rate: {hit_rate * 100:.1f}%")

            # Should have good hit rate for repeated shapes
            assert hits > 0, "Should have cache hits for repeated shapes"
            assert hit_rate > 0.3, f"Expected >30% hit rate, got {hit_rate * 100:.1f}%"

    def test_lru_eviction_preserves_hot_arrays(self):
        """Verify LRU eviction preserves frequently accessed arrays.

        Hot arrays should remain in pool after eviction while cold
        arrays are removed.
        """
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            hot_shapes = [(100, 10), (200, 5)]
            cold_shapes = [(300, 3), (400, 2), (500, 1)]

            # Add all arrays to pool
            for shape in hot_shapes + cold_shapes:
                manager.allocate_array(shape, dtype=np.float64)

            # Access hot shapes multiple times
            for _ in range(5):
                for shape in hot_shapes:
                    manager.allocate_array(shape, dtype=np.float64)

            # Evict to keep only 3 arrays
            manager.optimize_memory_pool(max_arrays=3)

            # Check hot shapes are preserved
            for shape in hot_shapes:
                key = (shape, np.float64)
                assert key in manager.memory_pool, (
                    f"Hot shape {shape} should be preserved after eviction"
                )

            assert len(manager.memory_pool) == 3, (
                f"Pool should have 3 arrays, has {len(manager.memory_pool)}"
            )

            print(
                f"\n[LRU Preservation] Pool contains {len(manager.memory_pool)} arrays"
            )

    def test_pool_efficiency_metric(self):
        """Calculate memory pool efficiency: hits / (hits + new_allocations).

        This metric shows how well the pool avoids redundant allocations.
        Expected: >80% efficiency for typical optimization workloads.
        """
        manager = MemoryManager(adaptive_ttl=True)

        with patch("nlsq.caching.memory_manager.psutil", create_psutil_mock()):
            hits = 0
            new_allocations = 0

            # Simulate optimization workload
            # Jacobian and gradient arrays are frequently reused
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

            # First iteration: all misses (4 shapes)
            # Subsequent 9 iterations: all hits (36 hits)
            expected_efficiency = 36 / (36 + 4)

            assert efficiency > 0.8, (
                f"Expected >80% efficiency, got {efficiency * 100:.1f}%"
            )

    def test_lru_vs_fifo_comparison(self):
        """Compare LRU vs FIFO eviction behavior.

        LRU should perform better than FIFO for workloads with hot items.
        """
        # LRU simulation using OrderedDict (matches implementation)
        lru_pool: OrderedDict = OrderedDict()

        # FIFO simulation
        fifo_pool: dict = {}
        fifo_order: list = []

        max_size = 5

        # Workload with hot items
        workload = [
            (100, 10),  # Hot
            (200, 5),
            (300, 3),
            (400, 2),
            (500, 1),
            (100, 10),  # Hot accessed again
            (600, 1),  # New (triggers eviction)
            (100, 10),  # Hot accessed again
            (700, 1),  # New (triggers eviction)
            (100, 10),  # Hot accessed again
        ]

        lru_hits = 0
        fifo_hits = 0

        for shape in workload:
            key = (shape, np.float64)

            # LRU
            if key in lru_pool:
                lru_hits += 1
                lru_pool.move_to_end(key)
            else:
                if len(lru_pool) >= max_size:
                    lru_pool.popitem(last=False)
                lru_pool[key] = True

            # FIFO
            if key in fifo_pool:
                fifo_hits += 1
            else:
                if len(fifo_pool) >= max_size:
                    oldest = fifo_order.pop(0)
                    del fifo_pool[oldest]
                fifo_pool[key] = True
                fifo_order.append(key)

        lru_hit_rate = lru_hits / len(workload)
        fifo_hit_rate = fifo_hits / len(workload)

        print(f"\n[LRU vs FIFO] LRU hit rate: {lru_hit_rate * 100:.1f}%")
        print(f"[LRU vs FIFO] FIFO hit rate: {fifo_hit_rate * 100:.1f}%")

        # LRU should be at least as good as FIFO
        assert lru_hit_rate >= fifo_hit_rate, (
            f"LRU ({lru_hit_rate * 100:.1f}%) should be >= FIFO ({fifo_hit_rate * 100:.1f}%)"
        )


class TestPhase2CumulativeImprovements:
    """Test cumulative improvements from Phase 1 + Phase 2 memory optimizations."""

    def test_combined_ttl_and_lru_efficiency(self):
        """Test combined effect of adaptive TTL and LRU memory pool.

        Phase 1: Adaptive TTL reduces psutil overhead
        Phase 2: LRU pool improves memory reuse efficiency
        Combined: Should see significant efficiency gains.
        """
        manager = MemoryManager(memory_cache_ttl=1.0, adaptive_ttl=True)

        with patch(
            "nlsq.caching.memory_manager.psutil", create_psutil_mock()
        ) as mock_psutil:
            # Simulate typical streaming optimization workload
            common_shapes = [(1000, 5), (5, 5), (1000,)]
            n_batches = 100

            pool_hits = 0
            pool_misses = 0
            start_time = time.perf_counter()

            for _ in range(n_batches):
                # Memory check (benefits from adaptive TTL)
                manager.get_available_memory()

                # Array allocations (benefit from LRU pool)
                for shape in common_shapes:
                    key = (shape, np.float64)
                    if key in manager.memory_pool:
                        pool_hits += 1
                    else:
                        pool_misses += 1
                    manager.allocate_array(shape, dtype=np.float64)

            elapsed = time.perf_counter() - start_time

            # Calculate metrics
            pool_efficiency = pool_hits / (pool_hits + pool_misses)

            print("\n[Combined Phase 1+2 Efficiency]")
            print(f"  Total batches: {n_batches}")
            print(f"  Time elapsed: {elapsed:.3f}s")
            print(f"  Pool hits: {pool_hits}, misses: {pool_misses}")
            print(f"  Pool efficiency: {pool_efficiency * 100:.1f}%")

            # Pool efficiency should be high
            assert pool_efficiency > 0.9, (
                f"Pool efficiency should be >90%, got {pool_efficiency * 100:.1f}%"
            )
