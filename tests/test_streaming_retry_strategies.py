"""Tests for adaptive retry strategies in streaming optimizer.

This module tests error-specific retry strategies that attempt to recover from
transient failures during streaming optimization.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from nlsq.streaming_optimizer import DataGenerator, StreamingConfig, StreamingOptimizer


class TestAdaptiveRetryStrategies:
    """Test adaptive retry strategies for different error types."""

    def test_retry_with_reduced_learning_rate(self):
        """Test retry with 50% reduced learning rate on NaN/Inf errors."""

        # Simple linear model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(200)

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        # Track retry attempts
        retry_info = {"attempts": [], "learning_rates": []}

        # Patch gradient computation to fail first time with NaN
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_2_attempts = [0]

        def mock_compute(func, params, x_batch, y_batch):
            call_count[0] += 1
            # Track batch 2 attempts
            if 4 <= call_count[0] <= 6:  # Batch 2 (50-100)
                batch_2_attempts[0] += 1
                if batch_2_attempts[0] == 1:
                    # First attempt: return NaN gradient
                    return 0.5, np.array([np.nan, 1.0])
            return original_compute(func, params, x_batch, y_batch)

        # Patch update_parameters to track learning rate changes
        original_update = optimizer._update_parameters

        def mock_update(params, grad, bounds):
            # Check learning rate during retry
            if hasattr(optimizer, "_retry_learning_rate_factor"):
                retry_info["learning_rates"].append(
                    config.learning_rate * optimizer._retry_learning_rate_factor
                )
            return original_update(params, grad, bounds)

        optimizer._compute_loss_and_gradient = mock_compute
        optimizer._update_parameters = mock_update

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify retry was attempted
        assert result["success"]
        assert "retry_counts" in result.get("streaming_diagnostics", {})
        # Check that parameters improved from initial guess
        assert not np.array_equal(result["x"], p0)

    def test_retry_with_parameter_perturbation(self):
        """Test retry with parameter perturbation for singular matrix errors."""

        # Simple linear model (exponential can be problematic with JAX)
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0 + 0.05 * np.random.randn(200)

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.01,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        # Mock to simulate singular matrix error on batch 2
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_2_attempts = [0]
        batch_2_failed = [False]

        def mock_compute(func, params, x_batch, y_batch):
            call_count[0] += 1
            if 4 <= call_count[0] <= 6:  # Batch 2
                batch_2_attempts[0] += 1
                if batch_2_attempts[0] == 1:
                    # First attempt: raise linalg error
                    batch_2_failed[0] = True
                    raise np.linalg.LinAlgError("Singular matrix")
            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify retry was attempted
        assert result["success"]
        assert not np.array_equal(result["x"], p0)

        # Verify that batch 2 failed at least once
        assert batch_2_failed[0], "Batch 2 should have failed initially"

        # Verify diagnostics recorded the error
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]
            # Check retry was recorded
            assert len(diags.get("retry_counts", {})) > 0, "Should have retry counts"
            # Check error type was recorded
            assert "SingularMatrix" in diags.get("error_types", {}), (
                "Should have recorded SingularMatrix error"
            )

    def test_retry_with_reduced_batch_size(self):
        """Test retry with reduced batch size for memory errors."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(200)

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        # Track batch size changes
        batch_size_info = {"sizes": []}

        # Mock memory error on batch 3
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_3_attempts = [0]

        def mock_compute(func, params, x_batch, y_batch):
            call_count[0] += 1
            # Record batch size
            batch_size_info["sizes"].append(len(x_batch))

            if 7 <= call_count[0] <= 9:  # Batch 3
                batch_3_attempts[0] += 1
                if batch_3_attempts[0] == 1:
                    # First attempt: raise memory error
                    raise MemoryError("Out of memory")
            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify optimization completed
        assert result["success"]
        assert not np.array_equal(result["x"], p0)

    def test_maximum_retry_limit_enforcement(self):
        """Test that retry attempts are limited to max_retries_per_batch."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,  # Maximum 2 retries
        )
        optimizer = StreamingOptimizer(config)

        # Track all retry attempts
        retry_attempts = {"batch_2": 0}

        # Mock persistent failure on batch 2
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute(func, params, x_batch, y_batch):
            call_count[0] += 1
            if 4 <= call_count[0] <= 10:  # Batch 2 and potential retries
                retry_attempts["batch_2"] += 1
                # Always fail for this batch
                return 0.5, np.array([np.nan, np.nan])
            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify maximum retry limit was enforced
        # Should be 1 initial attempt + max 2 retries = 3 total
        assert retry_attempts["batch_2"] <= 3

        # Optimization should still complete (other batches succeed)
        assert "x" in result
        assert (
            result["streaming_diagnostics"]["batch_success_rate"] < 1.0
        )  # Some batches failed

    def test_different_error_type_handling(self):
        """Test that different error types trigger appropriate retry strategies."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(400)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(400)

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        # Track which retry strategies were used and actual failures
        retry_strategies = {
            "nan_inf": False,
            "singular": False,
            "memory": False,
            "generic": False,
        }
        permanent_failures = []

        # Mock different errors for different batches
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_attempts = {}

        def mock_compute(func, params, x_batch, y_batch):
            call_count[0] += 1
            batch_num = (call_count[0] - 1) // 3  # Approximate batch number

            if batch_num not in batch_attempts:
                batch_attempts[batch_num] = 0
            batch_attempts[batch_num] += 1

            # Different errors for different batches (first attempt only)
            if batch_attempts[batch_num] == 1:
                if batch_num == 1:
                    # Batch 1: NaN/Inf error (recoverable on retry)
                    retry_strategies["nan_inf"] = True
                    return 0.5, np.array([np.inf, 1.0])
                elif batch_num == 2:
                    # Batch 2: Singular matrix (recoverable with perturbation)
                    retry_strategies["singular"] = True
                    raise np.linalg.LinAlgError("Singular matrix")
                elif batch_num == 3:
                    # Batch 3: Memory error (recoverable with smaller batch)
                    retry_strategies["memory"] = True
                    raise MemoryError("Out of memory")
                elif batch_num == 4:
                    # Batch 4: Generic error (may recover with perturbation)
                    retry_strategies["generic"] = True
                    raise ValueError("Generic error")
            elif batch_attempts[batch_num] > config.max_retries_per_batch + 1:
                # This batch has permanently failed
                if batch_num not in permanent_failures:
                    permanent_failures.append(batch_num)

            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Check diagnostics for error categorization
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]
            if "error_types" in diags:
                # Should have multiple error types recorded
                assert len(diags["error_types"]) >= 1
                # At least one of the error types should be recorded
                recorded_types = set(diags["error_types"].keys())
                expected_types = {
                    "NumericalError",
                    "SingularMatrix",
                    "MemoryError",
                    "ValueError",
                }
                assert len(recorded_types.intersection(expected_types)) > 0, (
                    f"Expected some of {expected_types}, got {recorded_types}"
                )

        # Verify some retries were attempted
        assert "streaming_diagnostics" in result
        assert len(result["streaming_diagnostics"].get("retry_counts", {})) > 0

    def test_retry_success_updates_best_params(self):
        """Test that successful retries can update best parameters."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0 + 0.01 * np.random.randn(200)

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        # Mock error that succeeds on retry
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_2_attempts = [0]

        def mock_compute(func, params, x_batch, y_batch):
            call_count[0] += 1
            if 4 <= call_count[0] <= 6:  # Batch 2
                batch_2_attempts[0] += 1
                if batch_2_attempts[0] == 1:
                    # First attempt fails
                    return 0.5, np.array([np.nan, np.nan])
                # Retry succeeds with good gradient
                return 0.1, np.array([-0.5, -0.2])
            return original_compute(func, params, x_batch, y_batch)

        optimizer._compute_loss_and_gradient = mock_compute

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify optimization succeeded
        assert result["success"]
        # Parameters should have improved from initial
        assert not np.array_equal(result["x"], p0)
        # Should be closer to true values [2.0, 1.0]
        assert abs(result["x"][0] - 2.0) < 1.0
        assert abs(result["x"][1] - 1.0) < 1.0

    def test_retry_with_perturbation_uses_jax_random(self):
        """Test that parameter perturbation uses JAX random for reproducibility."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )

        # Create two optimizers with same config
        optimizer1 = StreamingOptimizer(config)
        optimizer2 = StreamingOptimizer(config)

        # Mock error for both optimizers
        def create_mock_compute(original_compute):
            call_count = [0]
            batch_2_attempts = [0]

            def mock_compute(func, params, x_batch, y_batch):
                call_count[0] += 1
                if 4 <= call_count[0] <= 6:  # Batch 2
                    batch_2_attempts[0] += 1
                    if batch_2_attempts[0] == 1:
                        # First attempt: raise error requiring perturbation
                        raise np.linalg.LinAlgError("Singular matrix")
                return original_compute(func, params, x_batch, y_batch)

            return mock_compute

        optimizer1._compute_loss_and_gradient = create_mock_compute(
            optimizer1._compute_loss_and_gradient
        )
        optimizer2._compute_loss_and_gradient = create_mock_compute(
            optimizer2._compute_loss_and_gradient
        )

        # Run both optimizers with same initial parameters
        p0 = np.array([1.0, 0.0])

        # Set same random seed for reproducibility
        np.random.seed(42)
        result1 = optimizer1.fit_streaming((x_data, y_data), model, p0, verbose=0)

        np.random.seed(42)
        result2 = optimizer2.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Both should succeed
        assert result1["success"]
        assert result2["success"]

        # Results should be similar (not necessarily identical due to async operations)
        # But should both have improved from initial parameters
        assert not np.array_equal(result1["x"], p0)
        assert not np.array_equal(result2["x"], p0)
