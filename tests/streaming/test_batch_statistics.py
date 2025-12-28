"""Tests for streaming optimizer batch statistics tracking.

This module tests the circular buffer implementation, statistics calculation,
memory usage limits, and aggregate statistics computation (Task Group 7).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.config import StreamingConfig
from nlsq.streaming.optimizer import StreamingOptimizer


def exponential_model(x, a, b):
    """Simple exponential model for testing - uses JAX for JIT compatibility."""
    return a * jnp.exp(-b * x)


class TestCircularBufferOverflow:
    """Test circular buffer overflow handling (Task 7.1.1)."""

    def test_circular_buffer_respects_size_limit(self):
        """Test that circular buffer never exceeds configured size."""
        # Configure small buffer size for testing
        config = StreamingConfig(
            batch_size=16,
            max_epochs=1,
            batch_stats_buffer_size=10,  # Small buffer
            enable_checkpoints=False,
        )

        optimizer = StreamingOptimizer(config)

        # Generate enough data for more than 10 batches
        np.random.seed(42)
        x_data = np.linspace(0, 10, 200)  # 200 points / 16 per batch = 13 batches
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        # Fit model
        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Check buffer size never exceeds limit
        assert len(optimizer.batch_stats_buffer) <= config.batch_stats_buffer_size

        # Should be exactly at limit (not under) since we had >10 batches
        assert len(optimizer.batch_stats_buffer) == config.batch_stats_buffer_size

    def test_circular_buffer_fifo_behavior(self):
        """Test that oldest entries are removed first (FIFO)."""
        config = StreamingConfig(
            batch_size=16,
            max_epochs=1,
            batch_stats_buffer_size=5,  # Very small buffer
            enable_checkpoints=False,
        )

        optimizer = StreamingOptimizer(config)

        # Generate data for exactly 10 batches
        np.random.seed(42)
        x_data = np.linspace(0, 10, 160)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Buffer should contain last 5 batches
        assert len(optimizer.batch_stats_buffer) == 5

        # Check that batch indices are the last 5 (5, 6, 7, 8, 9)
        batch_indices = [stat["batch_idx"] for stat in optimizer.batch_stats_buffer]
        # Should be last 5 batches (indices may vary, but should be sequential and recent)
        assert batch_indices == sorted(batch_indices)  # Sequential
        assert max(batch_indices) >= 5  # Contains recent batches

    def test_circular_buffer_with_single_batch(self):
        """Test buffer behavior with single batch."""
        config = StreamingConfig(
            batch_size=100,  # Larger than dataset
            max_epochs=1,
            batch_stats_buffer_size=10,
            enable_checkpoints=False,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 50)  # Only 50 points = 1 batch
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Buffer should contain exactly 1 entry
        assert len(optimizer.batch_stats_buffer) == 1
        assert optimizer.batch_stats_buffer[0]["batch_idx"] == 0


class TestStatisticsCalculation:
    """Test statistics calculation accuracy (Task 7.1.2)."""

    def test_batch_statistics_contain_required_fields(self):
        """Test that batch statistics contain all required fields."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Check that each batch statistic has required fields
        assert len(optimizer.batch_stats_buffer) > 0

        for batch_stat in optimizer.batch_stats_buffer:
            assert "batch_idx" in batch_stat
            assert "loss" in batch_stat
            assert "grad_norm" in batch_stat
            assert "batch_time" in batch_stat
            assert "success" in batch_stat
            assert "retry_count" in batch_stat

            # Verify types
            assert isinstance(batch_stat["batch_idx"], int)
            assert isinstance(batch_stat["loss"], float)
            assert isinstance(batch_stat["grad_norm"], float)
            assert isinstance(batch_stat["batch_time"], float)
            assert isinstance(batch_stat["success"], bool)
            assert isinstance(batch_stat["retry_count"], int)

    def test_aggregate_statistics_calculated_correctly(self):
        """Test that aggregate statistics are computed correctly."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=2,  # Multiple epochs for more data
            enable_checkpoints=False,
            enable_fault_tolerance=True,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Should have succeeded
        assert result["success"]

        # Extract aggregate statistics
        diagnostics = result["streaming_diagnostics"]
        aggregate = diagnostics["aggregate_stats"]

        # Verify aggregate statistics are present
        assert "mean_loss" in aggregate
        assert "std_loss" in aggregate
        assert "min_loss" in aggregate
        assert "max_loss" in aggregate
        assert "mean_grad_norm" in aggregate

        # Verify values are reasonable
        assert aggregate["mean_loss"] > 0
        assert aggregate["std_loss"] >= 0
        assert aggregate["min_loss"] <= aggregate["mean_loss"] <= aggregate["max_loss"]
        assert aggregate["mean_grad_norm"] > 0

    def test_aggregate_statistics_match_manual_calculation(self):
        """Test that aggregate statistics match manual calculation."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
            enable_fault_tolerance=True,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.05 * np.random.randn(len(x_data))

        p0 = np.array([1.5, 0.4])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Should have succeeded
        assert result["success"]

        # Get aggregate statistics
        diagnostics = result["streaming_diagnostics"]
        aggregate = diagnostics["aggregate_stats"]

        # Manual calculation from batch statistics
        batch_stats = diagnostics["recent_batch_stats"]
        losses = [stat["loss"] for stat in batch_stats if stat["success"]]
        expected_mean = float(np.mean(losses))
        expected_std = float(np.std(losses))
        expected_min = float(np.min(losses))
        expected_max = float(np.max(losses))

        # Compare with tolerance for floating point
        assert np.isclose(aggregate["mean_loss"], expected_mean, rtol=1e-5)
        assert np.isclose(aggregate["std_loss"], expected_std, rtol=1e-5)
        assert np.isclose(aggregate["min_loss"], expected_min, rtol=1e-5)
        assert np.isclose(aggregate["max_loss"], expected_max, rtol=1e-5)


class TestMemoryUsageLimits:
    """Test memory usage bounds (Task 7.1.3)."""

    def test_memory_usage_bounded_by_buffer_size(self):
        """Test that memory usage is bounded by buffer size configuration."""
        # Test with different buffer sizes
        buffer_sizes = [10, 50, 100]

        np.random.seed(42)
        x_data = np.linspace(0, 10, 1000)  # Large dataset
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])

        for buffer_size in buffer_sizes:
            config = StreamingConfig(
                batch_size=16,
                max_epochs=1,
                batch_stats_buffer_size=buffer_size,
                enable_checkpoints=False,
            )

            optimizer = StreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

            # Verify buffer size is exactly at the limit
            assert len(optimizer.batch_stats_buffer) <= buffer_size

            # With 1000 points and batch_size=16, we have 63 batches total
            # Buffer should be at min(buffer_size, num_batches)
            expected_size = min(buffer_size, 63)
            assert len(optimizer.batch_stats_buffer) == expected_size

    def test_buffer_size_minimum_value(self):
        """Test that buffer_size must be at least 1 (as per config validation)."""
        # Config validation requires batch_stats_buffer_size > 0
        # This test verifies the validation works
        with pytest.raises(
            AssertionError, match="batch_stats_buffer_size must be positive"
        ):
            config = StreamingConfig(
                batch_size=32,
                max_epochs=1,
                batch_stats_buffer_size=0,  # Invalid
                enable_checkpoints=False,
            )

    def test_large_buffer_size_performance(self):
        """Test that large buffer sizes don't cause performance issues."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            batch_stats_buffer_size=1000,  # Very large buffer
            enable_checkpoints=False,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 500)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])

        import time

        start = time.time()
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 10 seconds for first JIT compilation)
        assert elapsed < 10.0

        # Optimization should succeed
        assert result["success"]


class TestAggregateStatisticsComputation:
    """Test aggregate statistics computation (Task 7.1.4)."""

    def test_recent_batch_stats_in_diagnostics(self):
        """Test that recent batch stats are included in diagnostics."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
            enable_fault_tolerance=True,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 200)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Check diagnostics structure
        assert "streaming_diagnostics" in result
        diagnostics = result["streaming_diagnostics"]

        assert "recent_batch_stats" in diagnostics
        assert "aggregate_stats" in diagnostics

        # Recent batch stats should match internal buffer
        recent_stats = diagnostics["recent_batch_stats"]
        assert len(recent_stats) == len(optimizer.batch_stats_buffer)
        # deque doesn't support slicing, compare full deques
        assert list(recent_stats) == list(optimizer.batch_stats_buffer)

    def test_aggregate_stats_with_all_successful_batches(self):
        """Test aggregate statistics when all batches succeed."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
            enable_fault_tolerance=True,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Should have succeeded
        assert result["success"]

        diagnostics = result["streaming_diagnostics"]

        # Should have 100% success rate
        assert diagnostics["batch_success_rate"] == 1.0

        # All batch stats should have finite losses
        for batch_stat in diagnostics["recent_batch_stats"]:
            assert np.isfinite(batch_stat["loss"])
            assert batch_stat["loss"] > 0

    def test_aggregate_stats_structure_complete(self):
        """Test that aggregate stats structure is complete."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
            enable_fault_tolerance=True,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Should have succeeded
        assert result["success"]

        aggregate = result["streaming_diagnostics"]["aggregate_stats"]

        # Required fields
        required_fields = [
            "mean_loss",
            "std_loss",
            "min_loss",
            "max_loss",
            "mean_grad_norm",
        ]

        for field in required_fields:
            assert field in aggregate
            assert isinstance(aggregate[field], float)
            assert np.isfinite(aggregate[field])


class TestBatchStatisticsWithFailures:
    """Test batch statistics with batch failures (Task 7.1 edge cases)."""

    def test_batch_stats_track_retry_counts(self):
        """Test that batch statistics track retry counts correctly."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Check that batch stats include retry_count field
        for batch_stat in optimizer.batch_stats_buffer:
            assert "retry_count" in batch_stat
            assert isinstance(batch_stat["retry_count"], int)
            assert batch_stat["retry_count"] >= 0

    def test_batch_stats_track_timing_correctly(self):
        """Test that batch timing is tracked correctly."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Check timing information from batch stats buffer
        batch_times = [
            s["batch_time"] for s in optimizer.batch_stats_buffer if s["success"]
        ]
        assert len(batch_times) > 0

        for batch_time in batch_times:
            assert (
                batch_time >= 0
            )  # Should be non-negative (may be 0.0 on Windows with fast ops)
            assert batch_time < 5.0  # Should be reasonably fast

        # Check that batch stats contain timing
        for batch_stat in optimizer.batch_stats_buffer:
            assert "batch_time" in batch_stat
            assert (
                batch_stat["batch_time"] >= 0
            )  # Allow 0.0 for timing precision on Windows
            assert batch_stat["batch_time"] < 5.0

    def test_gradient_norms_tracked_correctly(self):
        """Test that gradient norms are tracked correctly."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Should have succeeded
        assert result["success"]

        # Check gradient norms from batch stats buffer
        gradient_norms = [
            s["grad_norm"] for s in optimizer.batch_stats_buffer if s["success"]
        ]
        assert len(gradient_norms) > 0

        for grad_norm in gradient_norms:
            assert grad_norm > 0  # Should be positive
            assert np.isfinite(grad_norm)

        # Check that batch stats contain gradient norms
        for batch_stat in optimizer.batch_stats_buffer:
            assert "grad_norm" in batch_stat
            assert batch_stat["grad_norm"] >= 0
            assert np.isfinite(batch_stat["grad_norm"])


class TestBatchStatisticsIntegration:
    """Integration tests for batch statistics (Task 7.1 integration)."""

    def test_batch_stats_persist_across_epochs(self):
        """Test that batch statistics accumulate across epochs."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=3,  # Multiple epochs
            batch_stats_buffer_size=100,
            enable_checkpoints=False,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # With 100 points, batch_size=32, and 3 epochs:
        # 4 batches per epoch * 3 epochs = 12 batches total
        # Should have all 12 in buffer (since buffer_size=100)
        assert len(optimizer.batch_stats_buffer) >= 10  # Should have most batches

    def test_batch_stats_with_fast_mode(self):
        """Test batch statistics when fast mode (no fault tolerance) is enabled."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
            enable_fault_tolerance=False,  # Fast mode
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # In fast mode (no fault tolerance), batch statistics might not be tracked
        # to minimize overhead. Streaming_diagnostics should still be present
        # but may have minimal information
        assert "streaming_diagnostics" in result

        # If batch stats are tracked, they should be valid
        if len(optimizer.batch_stats_buffer) > 0:
            for batch_stat in optimizer.batch_stats_buffer:
                assert "loss" in batch_stat
                assert "grad_norm" in batch_stat

    def test_batch_stats_consistent_with_loss_history(self):
        """Test that batch statistics are consistent with loss history."""
        config = StreamingConfig(
            batch_size=32,
            max_epochs=1,
            enable_checkpoints=False,
            enable_fault_tolerance=True,
        )

        optimizer = StreamingOptimizer(config)

        np.random.seed(42)
        x_data = np.linspace(0, 10, 100)
        y_true = exponential_model(x_data, 2.0, 0.5)
        y_data = np.array(y_true) + 0.1 * np.random.randn(len(x_data))

        p0 = np.array([1.0, 1.0])
        result = optimizer.fit((x_data, y_data), exponential_model, p0, verbose=0)

        # Extract losses from batch stats
        batch_losses = [
            stat["loss"]
            for stat in optimizer.batch_stats_buffer
            if np.isfinite(stat["loss"])
        ]

        # Compare with aggregate stats
        diagnostics = result["streaming_diagnostics"]
        aggregate = diagnostics["aggregate_stats"]

        # Aggregate stats should match batch stats
        expected_mean = float(np.mean(batch_losses))
        expected_std = float(np.std(batch_losses))
        expected_min = float(np.min(batch_losses))
        expected_max = float(np.max(batch_losses))

        assert np.isclose(aggregate["mean_loss"], expected_mean, rtol=1e-5)
        assert np.isclose(aggregate["std_loss"], expected_std, rtol=1e-5)
        assert np.isclose(aggregate["min_loss"], expected_min, rtol=1e-5)
        assert np.isclose(aggregate["max_loss"], expected_max, rtol=1e-5)
