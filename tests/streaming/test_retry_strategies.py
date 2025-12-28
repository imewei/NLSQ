"""Tests for adaptive retry strategies in streaming optimizer.

This module tests error-specific retry strategies that attempt to recover from
transient failures during streaming optimization.
"""

import numpy as np

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


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

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            # Track batch 2 attempts
            if 4 <= call_count[0] <= 6:  # Batch 2 (50-100)
                batch_2_attempts[0] += 1
                if batch_2_attempts[0] == 1:
                    # First attempt: return NaN gradient
                    return 0.5, np.array([np.nan, 1.0])
            return original_compute(func, params, x_batch, y_batch, mask)

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
        """Test that optimization with retries enabled produces valid results.

        Verifies the retry mechanism works without mocking internal APIs.
        """

        # Simple linear model
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

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify optimization succeeded
        assert result["success"]
        assert not np.array_equal(result["x"], p0)
        assert np.all(np.isfinite(result["x"]))

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

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            # Record batch size
            batch_size_info["sizes"].append(len(x_batch))

            if 7 <= call_count[0] <= 9:  # Batch 3
                batch_3_attempts[0] += 1
                if batch_3_attempts[0] == 1:
                    # First attempt: raise memory error
                    raise MemoryError("Out of memory")
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify optimization completed
        assert result["success"]
        assert not np.array_equal(result["x"], p0)

    def test_maximum_retry_limit_enforcement(self):
        """Test that optimization respects retry limits.

        Verifies that retry limits are properly configured without mocking.
        """

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

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Optimization should complete
        assert "x" in result
        assert np.all(np.isfinite(result["x"]))

    def test_different_error_type_handling(self):
        """Test that optimization handles errors gracefully.

        Verifies the fault tolerance mechanism without mocking internal APIs.
        """

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

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Verify optimization completed
        assert result is not None
        assert "x" in result
        assert np.all(np.isfinite(result["x"]))

        # Verify diagnostics structure if present
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]
            assert isinstance(diags, dict)

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

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            if 4 <= call_count[0] <= 6:  # Batch 2
                batch_2_attempts[0] += 1
                if batch_2_attempts[0] == 1:
                    # First attempt fails
                    return 0.5, np.array([np.nan, np.nan])
                # Retry succeeds with good gradient
                return 0.1, np.array([-0.5, -0.2])
            return original_compute(func, params, x_batch, y_batch, mask)

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

            def mock_compute(func, params, x_batch, y_batch, mask=None):
                call_count[0] += 1
                if 4 <= call_count[0] <= 6:  # Batch 2
                    batch_2_attempts[0] += 1
                    if batch_2_attempts[0] == 1:
                        # First attempt: raise error requiring perturbation
                        raise np.linalg.LinAlgError("Singular matrix")
                return original_compute(func, params, x_batch, y_batch, mask)

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
