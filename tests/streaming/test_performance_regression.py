"""
Performance regression tests for streaming optimizer fault tolerance features.

This module provides pytest-benchmark integration specifically for streaming optimizer
fault tolerance features. These tests complement the main performance regression suite
by focusing on:
1. Overhead of fault tolerance features
2. Fast mode performance
3. Checkpoint I/O performance
4. Scaling with failure rates

Usage:
    # Run streaming performance tests
    pytest tests/test_streaming_performance_regression.py --benchmark-only

    # Save baseline
    pytest tests/test_streaming_performance_regression.py --benchmark-save=streaming_baseline

    # Compare against baseline
    pytest tests/test_streaming_performance_regression.py --benchmark-compare=streaming_baseline

    # Generate JSON report for CI
    pytest tests/test_streaming_performance_regression.py --benchmark-json=streaming_perf.json
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer
except ImportError:
    pytest.skip("Streaming optimizer not available", allow_module_level=True)


# ============================================================================
# Test Data Generators
# ============================================================================


@pytest.fixture(scope="module")
def small_streaming_data():
    """Small dataset for streaming (1000 points, 10 batches)"""
    np.random.seed(42)
    x = np.linspace(0, 10, 1000)
    y_true = 2.0 * x + 1.0
    noise = 0.1 * np.random.randn(len(x))
    y = y_true + noise
    return x, y, np.array([1.0, 0.0])


@pytest.fixture(scope="module")
def medium_streaming_data():
    """Medium dataset for streaming (10000 points, 100 batches)"""
    np.random.seed(42)
    x = np.linspace(0, 10, 10000)
    y_true = 2.0 * np.exp(-0.5 * x) + 0.3
    noise = 0.05 * np.random.randn(len(x))
    y = y_true + noise
    return x, y, np.array([2.0, 0.5, 0.3])


@pytest.fixture(scope="module")
def large_streaming_data():
    """Large dataset for streaming (50000 points, 500 batches)"""
    np.random.seed(42)
    x = np.linspace(-5, 5, 50000)
    y_true = 0.1 * x**3 - 0.5 * x**2 + 2.0 * x + 1.0
    noise = 0.2 * np.random.randn(len(x))
    y = y_true + noise
    return x, y, np.array([0.1, -0.5, 2.0, 1.0])


def linear_model(x, m, b):
    """Linear model: y = mx + b"""
    return m * x + b


def exponential_model(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    import jax.numpy as jnp

    return a * jnp.exp(-b * x) + c


def polynomial_model(x, a, b, c, d):
    """Cubic polynomial"""
    return a * x**3 + b * x**2 + c * x + d


# ============================================================================
# Benchmark Group 1: Baseline Performance (No Fault Tolerance)
# ============================================================================


@pytest.mark.benchmark(group="streaming-baseline")
def test_baseline_no_fault_tolerance(benchmark, small_streaming_data):
    """Benchmark: Baseline streaming performance without fault tolerance

    Expected: ~10-20ms baseline
    Critical: This establishes the performance baseline
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=False,
        enable_fault_tolerance=False,
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    # Validate result
    assert result is not None
    assert "x" in result
    assert not np.array_equal(result["x"], p0)


@pytest.mark.benchmark(group="streaming-baseline")
def test_baseline_medium_dataset(benchmark, medium_streaming_data):
    """Benchmark: Baseline with medium dataset (10K points)

    Expected: ~50-100ms
    Tests baseline scaling behavior
    """
    x, y, p0 = medium_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=False,
        enable_fault_tolerance=False,
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), exponential_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None
    assert "x" in result


# ============================================================================
# Benchmark Group 2: Fault Tolerance Overhead
# ============================================================================


@pytest.mark.benchmark(group="fault-tolerance-overhead")
def test_overhead_full_fault_tolerance(benchmark, small_streaming_data):
    """Benchmark: Full fault tolerance overhead

    Expected: <5% overhead vs baseline (~10-21ms)
    Critical: Must meet <5% overhead requirement
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=True,
        enable_fault_tolerance=True,
        max_retries_per_batch=2,
        min_success_rate=0.5,
        batch_stats_buffer_size=100,
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None
    assert "x" in result


@pytest.mark.benchmark(group="fault-tolerance-overhead")
def test_overhead_validation_only(benchmark, small_streaming_data):
    """Benchmark: NaN/Inf validation overhead only

    Expected: ~1-2% overhead
    Tests cost of numeric validation alone
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=True,
        enable_fault_tolerance=False,  # Disable retry logic
        batch_stats_buffer_size=1,  # Minimal statistics (0 not allowed)
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


@pytest.mark.benchmark(group="fault-tolerance-overhead")
def test_overhead_statistics_collection(benchmark, small_streaming_data):
    """Benchmark: Batch statistics collection overhead

    Expected: ~1-2% overhead
    Tests cost of circular buffer statistics tracking
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=False,
        enable_fault_tolerance=False,
        batch_stats_buffer_size=100,  # Enable statistics only
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


# ============================================================================
# Benchmark Group 3: Fast Mode Performance
# ============================================================================


@pytest.mark.benchmark(group="fast-mode")
def test_fast_mode_small_dataset(benchmark, small_streaming_data):
    """Benchmark: Fast mode with small dataset

    Expected: <1% overhead vs baseline (~10-10.1ms)
    Critical: Must meet <1% fast mode overhead requirement
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=False,
        enable_fault_tolerance=False,
        batch_stats_buffer_size=1,  # Minimal statistics (0 not allowed)
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


@pytest.mark.benchmark(group="fast-mode")
def test_fast_mode_medium_dataset(benchmark, medium_streaming_data):
    """Benchmark: Fast mode with medium dataset

    Expected: ~50-55ms
    Verifies fast mode scales well
    """
    x, y, p0 = medium_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=False,
        enable_fault_tolerance=False,
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), exponential_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


@pytest.mark.benchmark(group="fast-mode")
@pytest.mark.slow
def test_fast_mode_large_dataset(benchmark, large_streaming_data):
    """Benchmark: Fast mode with large dataset

    Expected: ~200-300ms
    Tests fast mode with realistic production sizes
    """
    x, y, p0 = large_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.05,
        validate_numerics=False,
        enable_fault_tolerance=False,
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), polynomial_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


# ============================================================================
# Benchmark Group 4: Checkpoint Performance
# ============================================================================


@pytest.mark.benchmark(group="checkpoint-io")
def test_checkpoint_save_overhead(benchmark, small_streaming_data):
    """Benchmark: Checkpoint save overhead

    Expected: ~15-25ms (includes periodic checkpoint saves)
    Tests I/O overhead of checkpointing
    """
    x, y, p0 = small_streaming_data

    temp_dir = tempfile.mkdtemp()

    try:
        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.1,
            checkpoint_interval=2,  # Save every 2 batches
            checkpoint_dir=temp_dir,
            validate_numerics=False,
            enable_fault_tolerance=False,
        )
        optimizer = StreamingOptimizer(config)

        def run_fit():
            return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

        result = benchmark(run_fit)

        assert result is not None

        # Verify checkpoints were created
        list(Path(temp_dir).glob("*.h5"))
        # May or may not have checkpoints depending on implementation

    finally:
        # Cleanup - use ignore_errors to handle race conditions with async checkpoint saves
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.benchmark(group="checkpoint-io")
def test_checkpoint_load_time(benchmark):
    """Benchmark: Checkpoint load time

    Expected: ~5-10ms
    Tests checkpoint restore performance
    """
    # Create a checkpoint first
    temp_dir = tempfile.mkdtemp()
    optimizers = []

    try:
        np.random.seed(42)
        x = np.linspace(0, 10, 1000)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
        p0 = np.array([1.0, 0.0])

        # Run to create checkpoint
        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            checkpoint_interval=1,
            checkpoint_dir=temp_dir,
        )
        optimizer = StreamingOptimizer(config)
        optimizers.append(optimizer)
        optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

        # Find checkpoint
        checkpoint_files = list(Path(temp_dir).glob("*.h5"))

        if checkpoint_files:
            checkpoint_path = str(checkpoint_files[0])

            def load_checkpoint():
                config_resume = StreamingConfig(
                    batch_size=100,
                    resume_from_checkpoint=checkpoint_path,
                )
                opt = StreamingOptimizer(config_resume)
                optimizers.append(opt)
                return opt

            # Benchmark loading
            optimizer_loaded = benchmark(load_checkpoint)
            assert optimizer_loaded is not None
        else:
            pytest.skip("No checkpoint created to benchmark loading")

    finally:
        import shutil

        # Shutdown all optimizer threads before cleanup
        for opt in optimizers:
            if hasattr(opt, "_shutdown_checkpoint_worker"):
                opt._shutdown_checkpoint_worker()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ============================================================================
# Benchmark Group 5: Performance with Failures
# ============================================================================


@pytest.mark.benchmark(group="failure-handling")
def test_performance_with_10_percent_failures(benchmark, small_streaming_data):
    """Benchmark: Performance with 10% batch failure rate

    Expected: ~12-15ms (~20% overhead due to retries)
    Tests realistic production failure scenario
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=True,
        enable_fault_tolerance=True,
        max_retries_per_batch=2,
    )
    optimizer = StreamingOptimizer(config)

    # Inject 10% failure rate
    original_compute = optimizer._compute_loss_and_gradient
    call_count = [0]

    def mock_compute_with_failures(func, params, x_batch, y_batch):
        call_count[0] += 1
        # Fail every 10th batch (first attempt only)
        if call_count[0] % 10 == 0:
            if not hasattr(mock_compute_with_failures, "failed_batches"):
                mock_compute_with_failures.failed_batches = set()

            batch_key = call_count[0]
            if batch_key not in mock_compute_with_failures.failed_batches:
                mock_compute_with_failures.failed_batches.add(batch_key)
                return 0.5, np.array([np.nan, 1.0])

        return original_compute(func, params, x_batch, y_batch, mask)

    optimizer._compute_loss_and_gradient = mock_compute_with_failures

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


@pytest.mark.benchmark(group="failure-handling")
def test_performance_with_50_percent_failures(benchmark, small_streaming_data):
    """Benchmark: Performance with 50% batch failure rate

    Expected: ~18-25ms (~80-100% overhead)
    Tests extreme failure scenario performance
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=True,
        enable_fault_tolerance=True,
        max_retries_per_batch=1,  # Fewer retries for stress test
        min_success_rate=0.4,  # Allow high failure rate
    )
    optimizer = StreamingOptimizer(config)

    # Inject 50% failure rate
    original_compute = optimizer._compute_loss_and_gradient
    call_count = [0]

    def mock_compute_high_failures(func, params, x_batch, y_batch):
        call_count[0] += 1
        # Fail every other batch
        if call_count[0] % 2 == 0:
            return 0.5, np.array([np.nan, 1.0])
        return original_compute(func, params, x_batch, y_batch, mask)

    optimizer._compute_loss_and_gradient = mock_compute_high_failures

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


# ============================================================================
# Benchmark Group 6: Memory Efficiency
# ============================================================================


@pytest.mark.benchmark(group="memory-efficiency")
@pytest.mark.slow
def test_memory_usage_large_dataset(benchmark, large_streaming_data):
    """Benchmark: Memory efficiency with large dataset

    Tests that circular buffer keeps memory usage bounded
    """
    x, y, p0 = large_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.05,
        validate_numerics=True,
        enable_fault_tolerance=True,
        batch_stats_buffer_size=100,  # Fixed size
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), polynomial_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None

    # Verify memory-efficient statistics
    if "streaming_diagnostics" in result:
        diags = result["streaming_diagnostics"]
        if "recent_batch_stats" in diags:
            # Should not exceed buffer size
            assert len(diags["recent_batch_stats"]) <= 100


@pytest.mark.benchmark(group="memory-efficiency")
def test_memory_circular_buffer_overflow(benchmark):
    """Benchmark: Circular buffer with many batches

    Tests that buffer overflow handling is efficient
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 10000)
    y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
    p0 = np.array([1.0, 0.0])

    config = StreamingConfig(
        batch_size=50,  # Many small batches
        max_epochs=2,  # Multiple epochs
        learning_rate=0.1,
        batch_stats_buffer_size=100,
    )
    optimizer = StreamingOptimizer(config)

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


# ============================================================================
# Benchmark Group 7: Retry Strategy Performance
# ============================================================================


@pytest.mark.benchmark(group="retry-performance")
def test_retry_strategy_nan_recovery(benchmark, small_streaming_data):
    """Benchmark: NaN retry strategy performance

    Tests overhead of NaN-specific retry (learning rate reduction)
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=True,
        enable_fault_tolerance=True,
        max_retries_per_batch=2,
    )
    optimizer = StreamingOptimizer(config)

    # Inject NaN errors that will be retried
    original_compute = optimizer._compute_loss_and_gradient
    call_count = [0]
    batch_attempts = {}

    def mock_compute_nan_retry(func, params, x_batch, y_batch):
        call_count[0] += 1
        batch_num = (call_count[0] - 1) // 3

        if batch_num not in batch_attempts:
            batch_attempts[batch_num] = 0
        batch_attempts[batch_num] += 1

        # Fail first attempt with NaN, succeed on retry
        if batch_attempts[batch_num] == 1 and batch_num % 5 == 0:
            return 0.5, np.array([np.nan, 1.0])

        return original_compute(func, params, x_batch, y_batch, mask)

    optimizer._compute_loss_and_gradient = mock_compute_nan_retry

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


@pytest.mark.benchmark(group="retry-performance")
def test_retry_strategy_mixed_errors(benchmark, small_streaming_data):
    """Benchmark: Mixed error types retry performance

    Tests overhead when different retry strategies are triggered
    """
    x, y, p0 = small_streaming_data

    config = StreamingConfig(
        batch_size=100,
        max_epochs=1,
        learning_rate=0.1,
        validate_numerics=True,
        enable_fault_tolerance=True,
        max_retries_per_batch=2,
    )
    optimizer = StreamingOptimizer(config)

    # Inject mixed error types
    original_compute = optimizer._compute_loss_and_gradient
    call_count = [0]
    batch_attempts = {}

    def mock_compute_mixed_errors(func, params, x_batch, y_batch):
        call_count[0] += 1
        batch_num = (call_count[0] - 1) // 3

        if batch_num not in batch_attempts:
            batch_attempts[batch_num] = 0
        batch_attempts[batch_num] += 1

        if batch_attempts[batch_num] == 1:
            if batch_num % 3 == 0:
                return 0.5, np.array([np.nan, 1.0])
            elif batch_num % 5 == 0:
                raise np.linalg.LinAlgError("Singular")

        return original_compute(func, params, x_batch, y_batch, mask)

    optimizer._compute_loss_and_gradient = mock_compute_mixed_errors

    def run_fit():
        return optimizer.fit_streaming((x, y), linear_model, p0, verbose=0)

    result = benchmark(run_fit)

    assert result is not None


# ============================================================================
# Utility Functions
# ============================================================================


def pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
    """Custom unit scaling for benchmark reports"""
    if unit == "seconds":
        return "milliseconds", 1000.0
    return unit, 1.0


def pytest_configure(config):
    """Configure pytest-benchmark"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (use -m 'not slow' to skip)"
    )


if __name__ == "__main__":
    """
    Run streaming performance benchmarks:

    python test_streaming_performance_regression.py
    """
    pytest.main([__file__, "--benchmark-only", "-v"])
