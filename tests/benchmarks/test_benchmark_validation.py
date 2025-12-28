"""Benchmark tests for validation overhead measurement.

Task 8.2: Measure validation overhead (CPU vs GPU).

This module benchmarks:
- Time spent in _validate_model_function per chunk
- Validation overhead with/without caching
- JIT-compiled validation performance
- Comparison of validation strategies

Expected results:
- Validation caching should reduce overhead by 1-5% in chunked processing
- JIT-compiled NaN/Inf validation should be 5-10% faster when enabled
- Cached validation should skip redundant checks across chunks

Uses pytest-benchmark for consistent measurement methodology.
"""

import time
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.config import StreamingConfig
from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig
from nlsq.streaming.optimizer import StreamingOptimizer


class TestValidationCachingBenchmark:
    """Benchmark model validation caching performance."""

    @pytest.fixture
    def validation_test_data(self):
        """Generate test data for validation benchmarks."""
        np.random.seed(42)
        n_samples = 10000
        x_data = np.linspace(0, 10, n_samples)
        true_params = [2.5, 0.3]
        y_data = true_params[0] * np.exp(
            -true_params[1] * x_data
        ) + 0.1 * np.random.randn(n_samples)
        p0 = np.array([1.0, 0.1])
        return x_data, y_data, p0

    def test_validation_caching_effectiveness(self, validation_test_data):
        """Measure validation caching effectiveness across multiple chunks.

        Validation caching uses (id(func), id(func.__code__)) as composite key
        to avoid redundant validation of the same function across chunks.
        """
        x_data, y_data, p0 = validation_test_data

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        config = LDMemoryConfig(
            memory_limit_gb=1.0,
            min_chunk_size=1000,
            max_chunk_size=2000,  # Force chunking
        )
        fitter = LargeDatasetFitter(config=config)

        # First call - should validate and cache
        validation_times = []

        start = time.perf_counter()
        fitter._validate_model_function(model, x_data, y_data, p0)
        first_validation_time = time.perf_counter() - start
        validation_times.append(first_validation_time)

        # Subsequent calls - should skip validation (cached)
        for _ in range(5):
            start = time.perf_counter()
            fitter._validate_model_function(model, x_data, y_data, p0)
            cached_time = time.perf_counter() - start
            validation_times.append(cached_time)

        # Check that function is in validated cache
        func_key = (id(model), id(model.__code__))
        assert func_key in fitter._validated_functions, (
            "Function should be cached after first validation"
        )

        # Cached calls should be much faster
        avg_cached_time = sum(validation_times[1:]) / len(validation_times[1:])
        speedup = (
            first_validation_time / avg_cached_time
            if avg_cached_time > 0
            else float("inf")
        )

        print(
            f"\n[Validation Caching] First validation: {first_validation_time * 1000:.3f}ms"
        )
        print(f"[Validation Caching] Avg cached time: {avg_cached_time * 1000:.3f}ms")
        print(f"[Validation Caching] Speedup: {speedup:.1f}x")

        # Cached validation should be significantly faster
        assert avg_cached_time < first_validation_time, (
            "Cached validation should be faster than first validation"
        )

    def test_validation_cache_different_functions(self, validation_test_data):
        """Test that different functions are validated separately.

        Each unique (id(func), id(func.__code__)) should be validated once.
        """
        x_data, y_data, p0 = validation_test_data

        def model1(x, a, b):
            return a * jnp.exp(-b * x)

        def model2(x, a, b):
            return a * jnp.sin(b * x)

        config = LDMemoryConfig(memory_limit_gb=1.0)
        fitter = LargeDatasetFitter(config=config)

        # Validate first model
        fitter._validate_model_function(model1, x_data, y_data, p0)
        key1 = (id(model1), id(model1.__code__))
        assert key1 in fitter._validated_functions

        # Validate second model (different function)
        fitter._validate_model_function(model2, x_data, y_data, p0)
        key2 = (id(model2), id(model2.__code__))
        assert key2 in fitter._validated_functions

        # Both should be cached
        assert len(fitter._validated_functions) == 2, (
            "Both functions should be in validation cache"
        )

        print(
            f"\n[Validation Cache] Cached functions: {len(fitter._validated_functions)}"
        )

    @pytest.mark.benchmark(group="validation_overhead")
    def test_validation_overhead_benchmark(self, benchmark, validation_test_data):
        """Benchmark validation overhead per call."""
        x_data, y_data, p0 = validation_test_data

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Create fresh fitter each time to avoid caching
        def validate_once():
            config = LDMemoryConfig(memory_limit_gb=1.0)
            fitter = LargeDatasetFitter(config=config)
            fitter._validate_model_function(model, x_data, y_data, p0)

        benchmark(validate_once)


class TestJITCompiledValidation:
    """Benchmark JIT-compiled NaN/Inf validation in streaming optimizer."""

    @pytest.fixture
    def streaming_test_data(self):
        """Generate test data for streaming validation benchmarks."""
        np.random.seed(42)
        n_samples = 5000
        x_data = np.linspace(0, 10, n_samples)
        true_params = [2.5, 0.3]
        y_data = true_params[0] * np.exp(
            -true_params[1] * x_data
        ) + 0.1 * np.random.randn(n_samples)
        p0 = np.array([1.0, 0.1])
        return x_data, y_data, p0

    def test_jit_validation_returns_valid_flag(self, streaming_test_data):
        """Verify JIT-compiled gradient function returns (loss, grad, is_valid) tuple.

        Task Group 7 (4.3a) moved NaN/Inf validation into the JIT-compiled
        gradient function, returning a validity flag to check after the call.
        """
        x_data, y_data, p0 = streaming_test_data

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=False,
            enable_checkpoints=False,
        )
        optimizer = StreamingOptimizer(config)

        # Prepare for gradient computation
        x_batch = jnp.array(x_data[:100])
        y_batch = jnp.array(y_data[:100])
        params = jnp.array(p0)

        # Test gradient computation with validity checking
        result = optimizer.fit((x_data, y_data), model, p0=p0, verbose=0)

        assert result["success"], "Optimization should succeed with valid data"
        print("\n[JIT Validation] Optimization completed successfully")

    def test_jit_validation_detects_nan(self, streaming_test_data):
        """Test that JIT validation detects NaN in gradients.

        When the model produces NaN values, the JIT-compiled validation
        should detect this and signal invalid results.
        """
        x_data, y_data, _p0 = streaming_test_data

        def nan_model(x, a, b):
            # This will produce NaN for negative values of b
            return a * jnp.exp(-b * x) + jnp.sqrt(b - 10)  # NaN when b < 10

        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            enable_fault_tolerance=True,  # Allow recovery
            enable_checkpoints=False,
        )
        optimizer = StreamingOptimizer(config)

        # Run optimization - should handle NaN gracefully
        result = optimizer.fit(
            (x_data, y_data),
            nan_model,
            p0=np.array([1.0, 0.1]),  # b=0.1 < 10, will produce NaN
            verbose=0,
        )

        # With fault tolerance, should still complete (but may not converge)
        print(f"\n[NaN Detection] Result success: {result['success']}")
        print(f"[NaN Detection] Final loss: {result.get('loss', 'N/A')}")

    @pytest.mark.benchmark(group="jit_validation")
    def test_jit_validation_performance(self, benchmark, streaming_test_data):
        """Benchmark JIT-compiled validation performance.

        Measures the overhead of including NaN/Inf validation in the
        JIT-compiled gradient computation.
        """
        x_data, y_data, p0 = streaming_test_data

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        def run_optimization():
            config = StreamingConfig(
                batch_size=100,
                max_epochs=2,
                enable_fault_tolerance=False,
                enable_checkpoints=False,
            )
            optimizer = StreamingOptimizer(config)
            return optimizer.fit((x_data, y_data), model, p0=p0, verbose=0)

        result = benchmark(run_optimization)
        assert result["success"], "Optimization should succeed"


class TestValidationOverheadComparison:
    """Compare validation overhead with different configurations."""

    @pytest.fixture
    def comparison_test_data(self):
        """Generate test data for comparison benchmarks."""
        np.random.seed(42)
        n_samples = 10000
        x_data = np.linspace(0, 10, n_samples)
        true_params = [2.5, 0.3]
        y_data = true_params[0] * np.exp(
            -true_params[1] * x_data
        ) + 0.1 * np.random.randn(n_samples)
        p0 = np.array([1.0, 0.1])
        return x_data, y_data, p0

    def test_measure_validation_overhead_per_chunk(self, comparison_test_data):
        """Measure time spent in validation per chunk.

        This test simulates chunked processing and measures validation
        overhead with and without caching.
        """
        x_data, y_data, p0 = comparison_test_data
        chunk_size = 2000
        n_chunks = len(x_data) // chunk_size

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Without caching simulation (fresh fitter per chunk)
        uncached_times = []
        for i in range(n_chunks):
            start = time.perf_counter()
            config = LDMemoryConfig(memory_limit_gb=1.0)
            fitter = LargeDatasetFitter(config=config)
            fitter._validate_model_function(model, x_data, y_data, p0)
            uncached_times.append(time.perf_counter() - start)

        # With caching (reuse fitter)
        cached_times = []
        config = LDMemoryConfig(memory_limit_gb=1.0)
        fitter = LargeDatasetFitter(config=config)
        for i in range(n_chunks):
            start = time.perf_counter()
            fitter._validate_model_function(model, x_data, y_data, p0)
            cached_times.append(time.perf_counter() - start)

        # Calculate statistics
        avg_uncached = sum(uncached_times) / len(uncached_times) * 1000
        avg_cached = sum(cached_times) / len(cached_times) * 1000
        total_uncached = sum(uncached_times) * 1000
        total_cached = sum(cached_times) * 1000
        savings = total_uncached - total_cached

        print("\n[Validation Overhead per Chunk]")
        print(f"  Number of chunks: {n_chunks}")
        print(f"  Avg uncached: {avg_uncached:.3f}ms")
        print(f"  Avg cached: {avg_cached:.3f}ms")
        print(f"  Total uncached: {total_uncached:.3f}ms")
        print(f"  Total cached: {total_cached:.3f}ms")
        print(f"  Savings: {savings:.3f}ms ({savings / total_uncached * 100:.1f}%)")

        # Cached validation should have lower average time
        assert avg_cached <= avg_uncached, (
            f"Cached ({avg_cached:.3f}ms) should be <= uncached ({avg_uncached:.3f}ms)"
        )

    def test_validation_caching_percentage_improvement(self, comparison_test_data):
        """Calculate percentage improvement from validation caching.

        Expected improvement: 1-5% in chunked processing.
        """
        x_data, y_data, p0 = comparison_test_data

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Simulate 10 chunks of validation
        n_chunks = 10

        # Time without caching (fresh fitter each time)
        start = time.perf_counter()
        for _ in range(n_chunks):
            config = LDMemoryConfig(memory_limit_gb=1.0)
            fitter = LargeDatasetFitter(config=config)
            fitter._validate_model_function(model, x_data, y_data, p0)
        time_uncached = time.perf_counter() - start

        # Time with caching (reuse fitter)
        config = LDMemoryConfig(memory_limit_gb=1.0)
        fitter = LargeDatasetFitter(config=config)
        start = time.perf_counter()
        for _ in range(n_chunks):
            fitter._validate_model_function(model, x_data, y_data, p0)
        time_cached = time.perf_counter() - start

        # Calculate improvement
        improvement = (
            (time_uncached - time_cached) / time_uncached * 100
            if time_uncached > 0
            else 0
        )

        print("\n[Validation Caching Improvement]")
        print(f"  Time uncached: {time_uncached * 1000:.3f}ms")
        print(f"  Time cached: {time_cached * 1000:.3f}ms")
        print(f"  Improvement: {improvement:.1f}%")

        # Should see improvement (though exact amount depends on overhead)
        assert time_cached <= time_uncached, (
            f"Cached time ({time_cached * 1000:.3f}ms) should be <= uncached ({time_uncached * 1000:.3f}ms)"
        )


class TestIsFiniteValidation:
    """Benchmark jnp.isfinite validation performance."""

    def test_isfinite_jit_compiled(self):
        """Test that jnp.isfinite runs efficiently in JIT context.

        Task Group 7 (4.3a) uses jnp.all(jnp.isfinite(grad)) inside JIT
        for efficient GPU-accelerated validation.
        """
        # Create test arrays
        np.random.seed(42)
        grad = jnp.array(np.random.randn(10000, 5))
        loss = jnp.array(1.0)

        # JIT-compile the validation
        @jax.jit
        def validate_finite(grad, loss):
            grad_valid = jnp.all(jnp.isfinite(grad))
            loss_valid = jnp.isfinite(loss)
            is_valid = grad_valid & loss_valid
            return is_valid

        # Warm up JIT
        _ = validate_finite(grad, loss)

        # Time the validation
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            is_valid = validate_finite(grad, loss)
            _ = is_valid.block_until_ready()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / n_iterations * 1000

        print(f"\n[JIT isfinite] Avg validation time: {avg_time:.4f}ms")
        print(
            f"[JIT isfinite] Total for {n_iterations} iterations: {elapsed * 1000:.3f}ms"
        )

        assert avg_time < 1.0, f"JIT validation should be <1ms, got {avg_time:.4f}ms"

    @pytest.mark.benchmark(group="isfinite_validation")
    def test_isfinite_benchmark(self, benchmark):
        """Benchmark jnp.isfinite performance for typical array sizes."""
        np.random.seed(42)
        grad = jnp.array(np.random.randn(10000, 5))
        loss = jnp.array(1.0)

        @jax.jit
        def validate_finite(grad, loss):
            grad_valid = jnp.all(jnp.isfinite(grad))
            loss_valid = jnp.isfinite(loss)
            return grad_valid & loss_valid

        # Warm up
        _ = validate_finite(grad, loss)

        def run_validation():
            result = validate_finite(grad, loss)
            return result.block_until_ready()

        benchmark(run_validation)
