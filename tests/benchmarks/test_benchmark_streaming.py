"""Benchmark tests for streaming optimizer throughput.

Task 6.1: Benchmark streaming throughput with different padding strategies.
Task 8.3: Add Phase 2 measurements to streaming benchmark.

This module benchmarks the streaming optimizer with:
- Static padding mode (early activation)
- Auto padding mode (early activation)
- Dynamic padding mode (baseline/no padding)
- Phase 2: JIT-compiled validation impact on throughput

Expected results:
- Static/auto modes should show 30-50% throughput improvement on GPU
- Eliminates JIT recompilation from varying batch sizes
- Phase 2: JIT validation adds minimal overhead (<5%)

Uses pytest-benchmark for consistent measurement methodology.
"""

import time

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.config import StreamingConfig
from nlsq.streaming.optimizer import StreamingOptimizer


# Test data fixture
@pytest.fixture
def streaming_test_data():
    """Generate test data for streaming benchmarks."""
    np.random.seed(42)
    n_samples = 5000  # Moderate size for benchmark
    x_data = np.linspace(0, 10, n_samples)
    true_params = [2.5, 0.3]
    y_data = true_params[0] * np.exp(-true_params[1] * x_data) + 0.1 * np.random.randn(
        n_samples
    )
    p0 = np.array([1.0, 0.1])
    return x_data, y_data, p0


def model_func(x, a, b):
    """Simple exponential decay model for benchmarking.

    Uses jax.numpy for JIT compatibility with the streaming optimizer.
    """
    return a * jnp.exp(-b * x)


class TestStreamingPaddingThroughput:
    """Benchmark streaming throughput with different padding strategies."""

    @pytest.mark.benchmark(group="streaming_padding")
    def test_static_padding_throughput(self, benchmark, streaming_test_data):
        """Benchmark streaming optimizer with static padding mode.

        Static mode sets _max_batch_shape = batch_size at initialization,
        eliminating warmup overhead and enabling padding from first batch.
        Expected: Best throughput due to consistent batch shapes.
        """
        x_data, y_data, p0 = streaming_test_data

        def run_optimization():
            config = StreamingConfig(
                batch_size=100,
                max_epochs=2,
                batch_shape_padding="static",
                enable_fault_tolerance=False,  # Fast mode
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
            return result

        result = benchmark(run_optimization)
        assert result["success"], "Optimization should succeed"

    @pytest.mark.benchmark(group="streaming_padding")
    def test_auto_padding_throughput(self, benchmark, streaming_test_data):
        """Benchmark streaming optimizer with auto padding mode.

        Auto mode also sets _max_batch_shape = batch_size at initialization
        (with early activation optimization), similar to static mode.
        Expected: Similar throughput to static mode.
        """
        x_data, y_data, p0 = streaming_test_data

        def run_optimization():
            config = StreamingConfig(
                batch_size=100,
                max_epochs=2,
                batch_shape_padding="auto",
                enable_fault_tolerance=False,
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
            return result

        result = benchmark(run_optimization)
        assert result["success"], "Optimization should succeed"

    @pytest.mark.benchmark(group="streaming_padding")
    def test_dynamic_padding_throughput(self, benchmark, streaming_test_data):
        """Benchmark streaming optimizer with dynamic padding mode (baseline).

        Dynamic mode retains the original behavior: _max_batch_shape starts
        as None and requires warmup phase to detect batch shapes.
        Expected: Slower than static/auto due to potential JIT recompilation.
        """
        x_data, y_data, p0 = streaming_test_data

        def run_optimization():
            config = StreamingConfig(
                batch_size=100,
                max_epochs=2,
                batch_shape_padding="dynamic",
                enable_fault_tolerance=False,
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
            return result

        result = benchmark(run_optimization)
        assert result["success"], "Optimization should succeed"


class TestStreamingBatchSizeVariation:
    """Benchmark throughput with varying batch sizes to measure JIT impact."""

    @pytest.mark.benchmark(group="batch_size_variation")
    def test_static_padding_with_last_batch_variation(
        self, benchmark, streaming_test_data
    ):
        """Benchmark static mode with intentional last-batch size variation.

        Uses a batch size that does not evenly divide the data, creating
        a smaller last batch. Static padding should handle this without
        JIT recompilation.
        """
        x_data, y_data, p0 = streaming_test_data

        def run_optimization():
            # Use batch size that doesn't evenly divide 5000 samples
            config = StreamingConfig(
                batch_size=128,  # 5000 / 128 = 39.0625 batches
                max_epochs=2,
                batch_shape_padding="static",
                enable_fault_tolerance=False,
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
            return result

        result = benchmark(run_optimization)
        assert result["success"], "Optimization should succeed"

    @pytest.mark.benchmark(group="batch_size_variation")
    def test_dynamic_mode_with_last_batch_variation(
        self, benchmark, streaming_test_data
    ):
        """Benchmark dynamic mode with last-batch size variation (baseline).

        Without padding, the smaller last batch may cause JIT recompilation
        on each epoch, reducing throughput.
        """
        x_data, y_data, p0 = streaming_test_data

        def run_optimization():
            config = StreamingConfig(
                batch_size=128,
                max_epochs=2,
                batch_shape_padding="dynamic",
                enable_fault_tolerance=False,
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
            return result

        result = benchmark(run_optimization)
        assert result["success"], "Optimization should succeed"


class TestStreamingThroughputMetrics:
    """Measure throughput in batches per second."""

    def test_measure_batches_per_second_static(self, streaming_test_data):
        """Measure batches/sec for static padding mode.

        This test provides a concrete throughput measurement rather than
        just relative timing.
        """
        x_data, y_data, p0 = streaming_test_data

        config = StreamingConfig(
            batch_size=100,
            max_epochs=3,
            batch_shape_padding="static",
            enable_fault_tolerance=False,
            enable_checkpoints=False,
        )
        optimizer = StreamingOptimizer(config)

        start_time = time.perf_counter()
        result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
        elapsed = time.perf_counter() - start_time

        # Calculate batches/sec
        n_samples = len(x_data)
        batch_size = config.batch_size
        n_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
        total_batches = n_batches_per_epoch * config.max_epochs
        batches_per_sec = total_batches / elapsed

        print(f"\n[Static Mode] Throughput: {batches_per_sec:.1f} batches/sec")
        print(f"[Static Mode] Total time: {elapsed:.3f}s for {total_batches} batches")

        assert result["success"], "Optimization should succeed"
        # Record metric for comparison
        assert batches_per_sec > 0, "Should have positive throughput"

    def test_measure_batches_per_second_dynamic(self, streaming_test_data):
        """Measure batches/sec for dynamic mode (baseline).

        Compare with static mode to quantify improvement.
        """
        x_data, y_data, p0 = streaming_test_data

        config = StreamingConfig(
            batch_size=100,
            max_epochs=3,
            batch_shape_padding="dynamic",
            enable_fault_tolerance=False,
            enable_checkpoints=False,
        )
        optimizer = StreamingOptimizer(config)

        start_time = time.perf_counter()
        result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
        elapsed = time.perf_counter() - start_time

        # Calculate batches/sec
        n_samples = len(x_data)
        batch_size = config.batch_size
        n_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
        total_batches = n_batches_per_epoch * config.max_epochs
        batches_per_sec = total_batches / elapsed

        print(f"\n[Dynamic Mode] Throughput: {batches_per_sec:.1f} batches/sec")
        print(f"[Dynamic Mode] Total time: {elapsed:.3f}s for {total_batches} batches")

        assert result["success"], "Optimization should succeed"
        assert batches_per_sec > 0, "Should have positive throughput"


class TestStreamingJITRecompilationTracking:
    """Verify JIT recompilation count is reduced with padding."""

    def test_static_mode_no_post_warmup_recompiles(self, streaming_test_data):
        """Verify static mode has no post-warmup JIT recompilations.

        With early activation, static mode should never enter warmup phase
        and should have zero post-warmup recompilations.
        """
        x_data, y_data, p0 = streaming_test_data

        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            batch_shape_padding="static",
            enable_fault_tolerance=False,
            enable_checkpoints=False,
        )
        optimizer = StreamingOptimizer(config)

        # Verify warmup is disabled at initialization
        assert optimizer._warmup_phase is False, "Static mode should skip warmup"
        assert optimizer._max_batch_shape == config.batch_size, (
            "Static mode should set max batch shape immediately"
        )

        # Run optimization
        result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)

        # Check post-warmup recompilations
        assert optimizer._post_warmup_recompiles == 0, (
            f"Expected 0 post-warmup recompiles, got {optimizer._post_warmup_recompiles}"
        )
        assert result["success"], "Optimization should succeed"

    def test_auto_mode_no_post_warmup_recompiles(self, streaming_test_data):
        """Verify auto mode has no post-warmup JIT recompilations.

        With early activation, auto mode should also skip warmup phase.
        """
        x_data, y_data, p0 = streaming_test_data

        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            batch_shape_padding="auto",
            enable_fault_tolerance=False,
            enable_checkpoints=False,
        )
        optimizer = StreamingOptimizer(config)

        # Verify warmup is disabled at initialization
        assert optimizer._warmup_phase is False, "Auto mode should skip warmup"
        assert optimizer._max_batch_shape == config.batch_size, (
            "Auto mode should set max batch shape immediately"
        )

        # Run optimization
        result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)

        # Check post-warmup recompilations
        assert optimizer._post_warmup_recompiles == 0, (
            f"Expected 0 post-warmup recompiles, got {optimizer._post_warmup_recompiles}"
        )
        assert result["success"], "Optimization should succeed"


# ============================================================================
# Phase 2 Benchmark Additions (Task 8.3)
# ============================================================================


class TestPhase2StreamingOptimizations:
    """Phase 2 streaming optimization benchmarks.

    Task 8.3: Add Phase 2 measurements including JIT-compiled validation
    and cumulative throughput improvements.
    """

    @pytest.mark.benchmark(group="phase2_streaming")
    def test_streaming_with_jit_validation(self, benchmark, streaming_test_data):
        """Benchmark streaming with JIT-compiled NaN/Inf validation.

        Phase 2 optimization: Task Group 7 (4.3a) moves validation into
        JIT-compiled gradient function for GPU-accelerated checking.
        Expected overhead: <5% due to efficient jnp.isfinite in JIT context.
        """
        x_data, y_data, p0 = streaming_test_data

        def run_with_validation():
            config = StreamingConfig(
                batch_size=100,
                max_epochs=2,
                batch_shape_padding="static",
                enable_fault_tolerance=True,  # Enables validation
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
            return result

        result = benchmark(run_with_validation)
        assert result["success"], "Optimization should succeed"

    @pytest.mark.benchmark(group="phase2_streaming")
    def test_streaming_without_validation(self, benchmark, streaming_test_data):
        """Benchmark streaming without validation (baseline for comparison).

        Compare with test_streaming_with_jit_validation to measure
        the overhead of JIT-compiled validation.
        """
        x_data, y_data, p0 = streaming_test_data

        def run_without_validation():
            config = StreamingConfig(
                batch_size=100,
                max_epochs=2,
                batch_shape_padding="static",
                enable_fault_tolerance=False,  # Disables validation
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
            return result

        result = benchmark(run_without_validation)
        assert result["success"], "Optimization should succeed"

    def test_phase2_cumulative_throughput_improvement(self, streaming_test_data):
        """Measure cumulative throughput improvement from Phase 1 + Phase 2.

        Compares:
        - Baseline: dynamic padding, no optimizations
        - Phase 1: static padding with early activation
        - Phase 1+2: static padding + JIT validation

        Expected cumulative improvement: 35-50% over baseline.
        """
        x_data, y_data, p0 = streaming_test_data
        n_runs = 3  # Average over multiple runs

        def measure_throughput(config):
            times = []
            for _ in range(n_runs):
                optimizer = StreamingOptimizer(config)
                start = time.perf_counter()
                result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
                elapsed = time.perf_counter() - start
                assert result["success"], "Optimization should succeed"
                times.append(elapsed)
            return sum(times) / len(times)

        # Baseline: dynamic padding, no optimizations
        baseline_config = StreamingConfig(
            batch_size=100,
            max_epochs=3,
            batch_shape_padding="dynamic",
            enable_fault_tolerance=False,
            enable_checkpoints=False,
        )
        baseline_time = measure_throughput(baseline_config)

        # Phase 1: static padding with early activation
        phase1_config = StreamingConfig(
            batch_size=100,
            max_epochs=3,
            batch_shape_padding="static",
            enable_fault_tolerance=False,
            enable_checkpoints=False,
        )
        phase1_time = measure_throughput(phase1_config)

        # Phase 1+2: static padding + fault tolerance (JIT validation)
        phase1_2_config = StreamingConfig(
            batch_size=100,
            max_epochs=3,
            batch_shape_padding="static",
            enable_fault_tolerance=True,
            enable_checkpoints=False,
        )
        phase1_2_time = measure_throughput(phase1_2_config)

        # Calculate improvements
        phase1_improvement = (baseline_time - phase1_time) / baseline_time * 100
        phase1_2_vs_baseline = (baseline_time - phase1_2_time) / baseline_time * 100
        validation_overhead = (phase1_2_time - phase1_time) / phase1_time * 100

        n_samples = len(x_data)
        batch_size = baseline_config.batch_size
        n_batches = (
            (n_samples + batch_size - 1) // batch_size * baseline_config.max_epochs
        )

        print("\n[Phase 2 Cumulative Throughput]")
        print(
            f"  Baseline (dynamic):      {baseline_time:.3f}s ({n_batches / baseline_time:.1f} batches/sec)"
        )
        print(
            f"  Phase 1 (static):        {phase1_time:.3f}s ({n_batches / phase1_time:.1f} batches/sec)"
        )
        print(
            f"  Phase 1+2 (w/ validation): {phase1_2_time:.3f}s ({n_batches / phase1_2_time:.1f} batches/sec)"
        )
        print(f"  Phase 1 improvement:     {phase1_improvement:.1f}%")
        print(f"  Phase 1+2 vs baseline:   {phase1_2_vs_baseline:.1f}%")
        print(f"  Validation overhead:     {validation_overhead:.1f}%")

        # Validation overhead should be reasonable (<50%)
        # Note: Overhead varies significantly based on system load, CPU frequency
        # scaling, and JIT compilation state. In ideal conditions it's <10%,
        # but in CI/parallel execution environments it can be higher.
        assert validation_overhead < 50, (
            f"Validation overhead ({validation_overhead:.1f}%) should be <50%"
        )

    def test_larger_batch_performance(self, streaming_test_data):
        """Test performance with larger batch sizes.

        Larger batches should reduce overhead and improve throughput.
        Phase 2 optimizations should scale well with batch size.
        """
        x_data, y_data, p0 = streaming_test_data

        batch_sizes = [50, 100, 200, 500]
        results = {}

        for batch_size in batch_sizes:
            config = StreamingConfig(
                batch_size=batch_size,
                max_epochs=2,
                batch_shape_padding="static",
                enable_fault_tolerance=True,
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)

            start = time.perf_counter()
            result = optimizer.fit((x_data, y_data), model_func, p0=p0, verbose=0)
            elapsed = time.perf_counter() - start

            assert result["success"], (
                f"Optimization should succeed for batch_size={batch_size}"
            )

            n_batches = (len(x_data) + batch_size - 1) // batch_size * config.max_epochs
            throughput = n_batches / elapsed

            results[batch_size] = {
                "time": elapsed,
                "batches": n_batches,
                "throughput": throughput,
            }

        print("\n[Batch Size Performance Scaling]")
        for batch_size, metrics in results.items():
            print(
                f"  batch_size={batch_size}: {metrics['time']:.3f}s, {metrics['throughput']:.1f} batches/sec"
            )

        # Larger batch sizes should generally be more efficient (fewer batches to process)
        assert len(results) == len(batch_sizes), "All batch sizes should complete"
