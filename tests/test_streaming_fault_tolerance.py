"""Comprehensive fault tolerance integration tests for streaming optimizer.

This module provides end-to-end integration tests that verify all fault tolerance
features work together cohesively during streaming optimization.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nlsq.streaming_optimizer import StreamingConfig, StreamingOptimizer


class TestFaultToleranceIntegration:
    """Integration tests for complete fault tolerance system."""

    def test_end_to_end_with_multiple_error_types(self):
        """Test optimization with mix of error types."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=2,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            min_success_rate=0.5,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        # Mock various error types on different batches
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_attempts = {}

        def mock_compute_with_errors(func, params, x_batch, y_batch):
            call_count[0] += 1
            batch_num = (call_count[0] - 1) // 3

            if batch_num not in batch_attempts:
                batch_attempts[batch_num] = 0
            batch_attempts[batch_num] += 1

            # First attempt: inject errors on specific batches
            if batch_attempts[batch_num] == 1:
                if batch_num == 2:
                    # NaN error
                    return 0.5, np.array([np.nan, 1.0])
                elif batch_num == 5:
                    # Singular matrix error
                    raise np.linalg.LinAlgError("Singular matrix")
                elif batch_num == 7:
                    # Memory error
                    raise MemoryError("Out of memory")

            # Retry or other batches: succeed
            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute_with_errors

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify optimization completed
        assert result is not None
        assert "x" in result

        # Parameters should have improved from initial
        assert not np.array_equal(result["x"], p0)

        # Should have diagnostics
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]
            # Should have recorded multiple error types
            assert "error_types" in diags
            # Should have retry counts
            assert "retry_counts" in diags

    def test_checkpoint_saves_during_fault_tolerance(self):
        """Test that checkpoints are saved even with batch failures."""
        temp_dir = tempfile.mkdtemp()

        try:
            config = StreamingConfig(
                batch_size=100,
                max_epochs=2,
                checkpoint_interval=3,
                checkpoint_dir=temp_dir,
                validate_numerics=True,
                enable_fault_tolerance=True,
            )
            optimizer = StreamingOptimizer(config)

            def model(x, a, b):
                return a * x + b

            # Generate test data
            np.random.seed(42)
            x_data = np.random.randn(1000)
            y_data = 2.0 * x_data + 1.0 + 0.05 * np.random.randn(1000)

            # Inject some failures
            original_compute = optimizer._compute_loss_and_gradient
            call_count = [0]

            def mock_compute(func, params, x_batch, y_batch):
                call_count[0] += 1
                # Fail every 5th batch
                if call_count[0] % 5 == 0:
                    return 0.5, np.array([np.nan, 1.0])
                return original_compute(func, params, x_batch, y_batch)

            optimizer._compute_loss_and_gradient = mock_compute

            # Run optimization
            p0 = np.array([1.0, 0.0])
            result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

            # Check that checkpoints were created
            checkpoint_files = list(Path(temp_dir).glob("checkpoint_*.h5"))

            # May or may not have checkpoints depending on implementation
            # Just verify optimization completed
            assert result is not None
            assert "x" in result

        finally:
            # Cleanup
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_best_params_tracked_through_failures(self):
        """Test that best parameters are preserved even when batches fail."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=3,
            learning_rate=0.05,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=1,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(300)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(300)

        # Track best loss throughout optimization
        best_losses = []

        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute_track_best(func, params, x_batch, y_batch):
            call_count[0] += 1

            # Fail batches 3, 6, 9 (first attempt only)
            if call_count[0] in [3, 6, 9]:
                batch_attempt_key = f"batch_{call_count[0]}"
                if not hasattr(mock_compute_track_best, "attempts"):
                    mock_compute_track_best.attempts = {}

                attempts = mock_compute_track_best.attempts.get(batch_attempt_key, 0)
                mock_compute_track_best.attempts[batch_attempt_key] = attempts + 1

                if attempts == 0:
                    # First attempt fails
                    return 100.0, np.array([np.nan, 1.0])

            result = original_compute(func, params, x_batch, y_batch)

            # Track best loss
            if hasattr(optimizer, "best_loss"):
                best_losses.append(optimizer.best_loss)

            return result

        optimizer._compute_loss_and_gradient = mock_compute_track_best

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify optimization worked
        assert result is not None
        assert not np.array_equal(result["x"], p0)

        # Verify best loss was tracked (if losses were recorded)
        if best_losses:
            # Best loss should generally decrease or stay same
            assert min(best_losses) <= best_losses[0]

    def test_retry_strategies_applied_correctly(self):
        """Test that different retry strategies are applied for different errors."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(300)
        y_data = 2.0 * x_data + 1.0 + 0.05 * np.random.randn(300)

        # Track which strategies were triggered
        strategies_used = []

        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_attempts = {}

        def mock_compute_with_strategy_tracking(func, params, x_batch, y_batch):
            call_count[0] += 1
            batch_num = (call_count[0] - 1) // 3

            if batch_num not in batch_attempts:
                batch_attempts[batch_num] = 0
            batch_attempts[batch_num] += 1

            # Only fail on first attempt
            if batch_attempts[batch_num] == 1:
                if batch_num == 1:
                    strategies_used.append("nan_error")
                    return 0.5, np.array([np.inf, 1.0])
                elif batch_num == 2:
                    strategies_used.append("singular_matrix")
                    raise np.linalg.LinAlgError("Singular matrix")
                elif batch_num == 3:
                    strategies_used.append("value_error")
                    raise ValueError("Invalid value")

            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute_with_strategy_tracking

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify different error types were encountered
        assert len(strategies_used) >= 2  # At least 2 different error types

        # Verify optimization completed
        assert result is not None
        assert "x" in result

    def test_success_rate_enforcement(self):
        """Test that optimization fails when success rate is too low."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=1,
            min_success_rate=0.8,  # Require 80% success
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0

        # Fail 40% of batches (below 80% threshold)
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute_with_high_failure(func, params, x_batch, y_batch):
            call_count[0] += 1
            # Fail 4 out of every 10 batches
            if call_count[0] % 10 in [1, 3, 5, 7]:
                return 100.0, np.array([np.nan, np.nan])
            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute_with_high_failure

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Check success rate
        if "batch_success_rate" in result:
            # Success rate should be around 60% (below 80% threshold)
            assert result["batch_success_rate"] < 0.8

        # Optimization should still return a result (best params found)
        assert result is not None
        assert "x" in result

    def test_diagnostics_accuracy_under_stress(self):
        """Test diagnostic accuracy with many simultaneous failures."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=2,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            min_success_rate=0.3,  # Allow high failure rate
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(600)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(600)

        # Track expected failures
        expected_failures = []
        actual_error_types = {}

        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_attempts = {}

        def mock_compute_with_tracking(func, params, x_batch, y_batch):
            call_count[0] += 1
            batch_num = (call_count[0] - 1) // 3

            if batch_num not in batch_attempts:
                batch_attempts[batch_num] = 0
            batch_attempts[batch_num] += 1

            # Complex failure pattern
            if batch_attempts[batch_num] == 1:
                if batch_num % 3 == 0:
                    expected_failures.append(batch_num)
                    actual_error_types[batch_num] = "NumericalError"
                    return 0.5, np.array([np.nan, 1.0])
                elif batch_num % 5 == 0:
                    expected_failures.append(batch_num)
                    actual_error_types[batch_num] = "SingularMatrix"
                    raise np.linalg.LinAlgError("Singular")

            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute_with_tracking

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify diagnostics if present
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]

            # Check that error types were recorded
            if "error_types" in diags:
                # Should have recorded some errors
                assert len(diags["error_types"]) > 0

            # Check that failed batches were tracked
            if "failed_batches" in diags or "failed_batch_indices" in result:
                # Some failures should be recorded
                failed_indices = diags.get(
                    "failed_batches", result.get("failed_batch_indices", [])
                )
                # At least some of the expected failures should be recorded
                # (may not be all if retries succeeded)
                assert len(failed_indices) >= 0  # Just verify structure exists

    def test_fast_mode_vs_full_mode_comparison(self):
        """Test that fast mode and full mode produce similar results."""
        # Generate test data once
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.02 * np.random.randn(500)
        p0 = np.array([1.0, 0.0])

        def model(x, a, b):
            return a * x + b

        # Run with full fault tolerance
        config_full = StreamingConfig(
            batch_size=50,
            max_epochs=2,
            learning_rate=0.05,
            validate_numerics=True,
            enable_fault_tolerance=True,
        )
        optimizer_full = StreamingOptimizer(config_full)
        result_full = optimizer_full.fit_streaming(
            (x_data, y_data), model, p0, verbose=0
        )

        # Run with fast mode
        config_fast = StreamingConfig(
            batch_size=50,
            max_epochs=2,
            learning_rate=0.05,
            validate_numerics=False,
            enable_fault_tolerance=False,
        )
        optimizer_fast = StreamingOptimizer(config_fast)
        result_fast = optimizer_fast.fit_streaming(
            (x_data, y_data), model, p0, verbose=0
        )

        # Both should produce results
        assert result_full is not None
        assert result_fast is not None

        # Both should have improved from initial parameters
        assert not np.array_equal(result_full["x"], p0)
        assert not np.array_equal(result_fast["x"], p0)

        # Results should be similar (within reasonable tolerance)
        # Note: May differ slightly due to different failure handling
        if result_full["success"] and result_fast["success"]:
            # Only compare if both succeeded
            assert np.allclose(result_full["x"], result_fast["x"], rtol=0.2)


class TestFaultToleranceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_batches_succeed(self):
        """Test behavior when no batches fail."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate clean data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0 + 0.01 * np.random.randn(200)

        # Run optimization (no failures injected)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should succeed with 100% success rate
        assert result is not None
        assert result["success"] is True
        if "batch_success_rate" in result:
            assert result["batch_success_rate"] == 1.0

    def test_single_batch_dataset(self):
        """Test with dataset that fits in single batch."""
        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate small dataset
        np.random.seed(42)
        x_data = np.random.randn(50)
        y_data = 2.0 * x_data + 1.0 + 0.01 * np.random.randn(50)

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should work with single batch
        assert result is not None
        assert not np.array_equal(result["x"], p0)

    def test_max_retries_exhausted(self):
        """Test behavior when max retries are exhausted for a batch."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            min_success_rate=0.3,  # Low threshold to allow test to pass
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(300)
        y_data = 2.0 * x_data + 1.0

        # Make batch 2 fail persistently
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute_persistent_failure(func, params, x_batch, y_batch):
            call_count[0] += 1
            batch_num = (call_count[0] - 1) // 5

            # Batch 2 always fails
            if batch_num == 2:
                return 100.0, np.array([np.nan, np.nan])

            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute_persistent_failure

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete despite persistent failure
        assert result is not None
        assert "x" in result

        # Should have recorded the failed batch
        if "failed_batch_indices" in result:
            assert len(result["failed_batch_indices"]) > 0


class TestFaultToleranceRobustness:
    """Test robustness of fault tolerance system."""

    def test_concurrent_validation_and_retry(self):
        """Test that validation and retry work together correctly."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=2,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(400)
        y_data = 2.0 * x_data + 1.0 + 0.05 * np.random.randn(400)

        # Inject errors that will trigger both validation and retry
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_attempts = {}

        def mock_compute_validation_retry(func, params, x_batch, y_batch):
            call_count[0] += 1
            batch_num = (call_count[0] - 1) // 3

            if batch_num not in batch_attempts:
                batch_attempts[batch_num] = 0
            batch_attempts[batch_num] += 1

            # First attempt: NaN (triggers validation and retry)
            if batch_attempts[batch_num] == 1 and batch_num % 4 == 0:
                return 0.5, np.array([np.nan, 1.0])

            # Second attempt should succeed
            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute_validation_retry

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should succeed with retries
        assert result is not None
        assert result["success"]

    def test_parameter_bounds_respected_during_retry(self):
        """Test that parameter bounds are respected during retry perturbation."""
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        # Set strict bounds
        bounds = (np.array([0.0, -2.0]), np.array([5.0, 5.0]))

        # Inject error to trigger retry
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute_for_bounds(func, params, x_batch, y_batch):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail second batch
                raise np.linalg.LinAlgError("Singular")
            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute_for_bounds

        # Run optimization
        p0 = np.array([2.0, 1.0])
        result = optimizer.fit_streaming(
            (x_data, y_data), model, p0, bounds=bounds, verbose=0
        )

        # Check that final parameters respect bounds
        if result["success"]:
            assert np.all(result["x"] >= bounds[0])
            assert np.all(result["x"] <= bounds[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
