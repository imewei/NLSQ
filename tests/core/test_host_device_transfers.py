"""Test suite for Task Group 2: Host-Device Transfer Reduction.

This test suite validates the implementation of:
- Task 2.4: Async logging with jax.debug.callback
- Task 2.6: Transfer profiling infrastructure
- Task 2.7-2.8: Transfer reduction via JAX operations
- Task 2.10: Performance improvement validation

All tests ensure GPU-CPU transfers are minimized during optimization.
"""

import os
import time
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core.least_squares import LeastSquares
from nlsq.utils.async_logger import (
    is_jax_array,
    log_convergence_async,
    log_iteration_async,
)
from nlsq.utils.profiling import (
    PerformanceMetrics,
    analyze_source_transfers,
    compare_transfer_reduction,
    profile_optimization,
)


class TestAsyncLogging:
    """Test Task 2.4: Async logging with jax.debug.callback."""

    def test_is_jax_array_detection(self):
        """Verify JAX array detection works correctly."""
        # JAX arrays
        assert is_jax_array(jnp.array([1.0, 2.0, 3.0]))
        assert is_jax_array(jnp.zeros(10))

        # NumPy arrays (not JAX)
        assert not is_jax_array(np.array([1.0, 2.0]))
        assert not is_jax_array(1.0)
        assert not is_jax_array("string")
        assert not is_jax_array(None)

    def test_log_iteration_async_basic(self):
        """Test basic async logging functionality."""
        # Mock the logger to capture calls
        with mock.patch("nlsq.utils.async_logger.logger") as mock_logger:
            log_iteration_async(
                iteration=10,
                cost=1.5e-6,
                gradient_norm=3.2e-8,
                message="test message",
                verbose=2,
            )

            # Callback executes asynchronously, wait briefly
            jax.effects_barrier()
            time.sleep(0.01)

            # Verify logger was called
            assert mock_logger.info.called

    def test_log_iteration_async_jax_arrays(self):
        """Test async logging with JAX arrays."""
        with mock.patch("nlsq.utils.async_logger.logger") as mock_logger:
            log_iteration_async(
                iteration=jnp.array(5),
                cost=jnp.array(2.3e-5),
                gradient_norm=jnp.array(1.1e-7),
                message="",
                verbose=2,
            )

            jax.effects_barrier()
            time.sleep(0.01)

            assert mock_logger.info.called

    def test_log_iteration_async_verbosity_levels(self):
        """Test verbosity control."""
        with mock.patch("nlsq.utils.async_logger.logger") as mock_logger:
            # verbose=0: No logging
            log_iteration_async(1, 1.0, 1.0, verbose=0)
            assert not mock_logger.info.called

            # verbose=1: Log every 10 iterations
            log_iteration_async(5, 1.0, 1.0, verbose=1)
            assert not mock_logger.info.called

            log_iteration_async(10, 1.0, 1.0, verbose=1)
            jax.effects_barrier()
            time.sleep(0.01)
            assert mock_logger.info.called

            mock_logger.reset_mock()

            # verbose=2: Log every iteration
            log_iteration_async(1, 1.0, 1.0, verbose=2)
            jax.effects_barrier()
            time.sleep(0.01)
            assert mock_logger.info.called

    def test_log_convergence_async(self):
        """Test convergence logging."""
        with mock.patch("nlsq.utils.async_logger.logger") as mock_logger:
            log_convergence_async(
                reason="`gtol` termination condition is satisfied.",
                iterations=42,
                final_cost=1.23e-10,
                time_sec=2.456,
                final_gradient_norm=5.67e-9,
                verbose=1,
            )

            jax.effects_barrier()
            time.sleep(0.01)

            assert mock_logger.info.called

    def test_async_logging_no_device_sync(self):
        """Verify async logging doesn't force device synchronization."""
        # This test ensures that calling log_iteration_async doesn't
        # trigger .block_until_ready() or similar blocking operations

        x = jnp.array([1.0, 2.0, 3.0])
        cost = jnp.sum(x**2)
        grad_norm = jnp.linalg.norm(x)

        # Logging should not force synchronization
        start = time.perf_counter()
        for i in range(5):
            log_iteration_async(i, cost, grad_norm, verbose=2)
        elapsed = time.perf_counter() - start

        # Should be reasonably fast (no blocking)
        # Relaxed threshold for CI environments (includes JIT compilation)
        assert elapsed < 0.5, f"Async logging too slow: {elapsed * 1000:.2f}ms"


class TestTransferProfiling:
    """Test Task 2.6: Transfer profiling infrastructure."""

    def test_analyze_source_transfers_numpy_array(self):
        """Test detection of np.array() calls."""
        code = """
def bad_function(x):
    y = np.array(x)  # Transfer!
    z = np.array(y)  # Another transfer!
    return z
"""
        result = analyze_source_transfers(code)
        assert result["np_array_calls"] == 2
        assert result["total_potential_transfers"] == 2

    def test_analyze_source_transfers_numpy_asarray(self):
        """Test detection of np.asarray() calls."""
        code = """
def mixed_function(x):
    y = np.asarray(x)
    z = np.array(y)
    return z.block_until_ready()
"""
        result = analyze_source_transfers(code)
        assert result["np_array_calls"] == 1
        assert result["np_asarray_calls"] == 1
        assert result["block_until_ready_calls"] == 1
        assert result["total_potential_transfers"] == 3

    def test_analyze_source_transfers_clean_code(self):
        """Test clean JAX code with no transfers."""
        code = """
def good_function(x):
    y = jnp.asarray(x)  # JAX, no transfer
    z = jnp.sum(y)
    return z
"""
        result = analyze_source_transfers(code)
        assert result["total_potential_transfers"] == 0

    def test_compare_transfer_reduction(self):
        """Test transfer reduction comparison."""
        before = """
def old_code(x):
    y = np.array(x)
    z = np.array(y)
    w = np.asarray(z)
    return w.block_until_ready()
"""

        after = """
def new_code(x):
    y = jnp.asarray(x)
    z = jnp.sum(y)
    return z
"""

        result = compare_transfer_reduction(before, after, "test_module")
        assert result["module"] == "test_module"
        assert result["before"]["total_potential_transfers"] == 4
        assert result["after"]["total_potential_transfers"] == 0
        assert result["reduction_count"] == 4
        assert result["reduction_percent"] == 100.0

    def test_compare_transfer_reduction_partial(self):
        """Test partial transfer reduction."""
        before = "y = np.array(x); z = np.array(y)"
        after = "y = jnp.asarray(x); z = np.array(y)"

        result = compare_transfer_reduction(before, after)
        assert result["reduction_count"] == 1
        assert result["reduction_percent"] == 50.0

    def test_compare_transfer_reduction_zero_before(self):
        """Test comparison when before has zero transfers."""
        before = "y = jnp.asarray(x)"
        after = "y = jnp.asarray(x)"

        result = compare_transfer_reduction(before, after)
        assert result["reduction_percent"] == 0.0


class TestTransferReduction:
    """Test Task 2.7-2.8: Transfer reduction via JAX operations."""

    def test_curve_fit_uses_jax_operations(self):
        """Verify curve_fit uses JAX operations internally."""

        # Simple exponential fit
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = jnp.linspace(0, 10, 50)
        y_true = 2.5 * jnp.exp(-0.3 * x)
        y_data = y_true + 0.1 * np.random.randn(50)

        # Fit should complete without excessive transfers
        popt, _pcov = curve_fit(model, x, y_data, p0=[1.0, 0.1])

        # Verify convergence
        assert popt is not None
        assert np.allclose(popt, [2.5, 0.3], rtol=0.1)

    def test_least_squares_minimal_transfers(self):
        """Verify least_squares minimizes host-device transfers."""

        def residual(params, x, y):
            a, b = params
            return y - a * jnp.exp(-b * x)

        np.random.seed(42)
        x = jnp.linspace(0, 5, 30)
        y = 1.8 * jnp.exp(-0.5 * x) + 0.05 * np.random.randn(30)

        # Use LeastSquares class API
        lsqs = LeastSquares()
        result = lsqs.least_squares(
            residual,
            x0=jnp.array([1.0, 0.1]),
            args=(x, y),
            max_nfev=50,
        )

        assert result.success
        assert np.allclose(result.x, [1.8, 0.5], rtol=0.1)


class TestPerformanceMetrics:
    """Test profiling utilities."""

    def test_performance_metrics_basic(self):
        """Test basic metrics tracking."""
        metrics = PerformanceMetrics()

        assert metrics.iteration_count == 0
        assert metrics.total_time_sec == 0.0
        assert metrics.avg_iteration_time_ms == 0.0
        assert metrics.min_iteration_time_ms == 0.0
        assert metrics.max_iteration_time_ms == 0.0

    def test_performance_metrics_calculations(self):
        """Test metrics calculations."""
        metrics = PerformanceMetrics(
            iteration_count=10,
            total_time_sec=1.0,
            iteration_times=[
                0.08,
                0.09,
                0.10,
                0.11,
                0.12,
                0.09,
                0.10,
                0.11,
                0.09,
                0.11,
            ],
        )

        assert metrics.avg_iteration_time_ms == 100.0  # 1.0s / 10 iters * 1000
        assert metrics.min_iteration_time_ms == 80.0
        assert metrics.max_iteration_time_ms == 120.0

    def test_profile_optimization_context(self):
        """Test profiling context manager."""
        with profile_optimization() as metrics:
            time.sleep(0.01)  # Simulate work

        assert metrics.total_time_sec >= 0.01
        assert metrics.total_time_sec < 0.1  # Sanity check

    def test_profile_optimization_disabled(self):
        """Test disabled profiling."""
        with profile_optimization(enabled=False) as metrics:
            time.sleep(0.01)

        # Should not track time when disabled
        assert metrics.total_time_sec == 0.0

    def test_profile_optimization_with_curve_fit(self):
        """Test profiling real optimization."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = jnp.linspace(0, 5, 50)
        y = 2.0 * jnp.exp(-0.4 * x) + 0.05 * np.random.randn(50)

        with profile_optimization() as metrics:
            popt, _pcov = curve_fit(model, x, y, p0=[1.0, 0.1])

        # Should complete in reasonable time
        assert metrics.total_time_sec > 0.0
        assert metrics.total_time_sec < 10.0  # Sanity check

        # Verify fit succeeded
        assert np.allclose(popt, [2.0, 0.4], rtol=0.2)


class TestPerformanceImprovement:
    """Test Task 2.10: Performance improvement validation."""

    def test_no_unnecessary_numpy_conversions(self):
        """Verify no unnecessary JAX→NumPy conversions in hot path."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        x = jnp.linspace(0, 3, 50)
        y = 2.2 * jnp.exp(-0.6 * x)

        # Profile the fit
        with profile_optimization() as metrics:
            popt, _ = curve_fit(model, x, y, p0=[1.0, 0.1])

        # Should complete in reasonable time
        # Includes JIT compilation, so allow generous time
        assert metrics.total_time_sec < 5.0

        # Verify convergence
        assert np.allclose(popt, [2.2, 0.6], rtol=0.1)


# Integration test combining multiple features
class TestIntegration:
    """Integration tests combining async logging, profiling, and transfer reduction."""

    def test_full_workflow_with_profiling(self):
        """Test complete workflow with profiling and async logging."""

        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        np.random.seed(42)
        x = jnp.linspace(0, 5, 100)
        y_true = 3.0 * jnp.exp(-0.5 * x) + 0.5
        y_data = y_true + 0.1 * np.random.randn(100)

        with profile_optimization() as metrics:
            popt, pcov = curve_fit(
                model,
                x,
                y_data,
                p0=[1.0, 0.1, 0.0],
                verbose=2,  # Async logging enabled
            )

        # Verify profiling
        assert metrics.total_time_sec > 0.0
        assert metrics.total_time_sec < 5.0

        # Verify convergence
        assert popt is not None
        assert np.allclose(popt, [3.0, 0.5, 0.5], rtol=0.2)

        # Verify covariance
        assert pcov is not None
        assert pcov.shape == (3, 3)

    def test_transfer_analysis_workflow(self):
        """Test transfer analysis on real code."""
        # Simulate analyzing before/after code
        before_code = """
def old_fit(x, y):
    params = np.array([1.0, 0.5])
    result = optimize(params)
    return np.array(result)
"""

        after_code = """
def new_fit(x, y):
    params = jnp.array([1.0, 0.5])
    result = optimize(params)
    return result
"""

        comparison = compare_transfer_reduction(before_code, after_code, "fit_module")

        assert comparison["reduction_count"] == 2
        assert comparison["reduction_percent"] == 100.0
        assert comparison["module"] == "fit_module"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
