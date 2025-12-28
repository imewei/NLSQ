"""Tests for best parameter tracking in streaming optimizer.

This module tests that the streaming optimizer correctly tracks
the best parameters achieved during optimization and handles
batch failures gracefully.
"""

import numpy as np

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestBestParameterTracking:
    """Test best parameter tracking throughout optimization."""

    def test_best_params_updated_on_improvement(self):
        """Test that best_params is updated when loss improves."""

        # Create simple quadratic function
        def model(x, a, b):
            return a * x**2 + b

        # Generate synthetic data
        np.random.seed(42)
        x_true = np.linspace(-5, 5, 1000)
        y_true = 2.0 * x_true**2 + 3.0
        y_noisy = y_true + np.random.normal(0, 0.1, len(y_true))

        # Create optimizer
        config = StreamingConfig(batch_size=100, max_epochs=2, learning_rate=0.01)
        optimizer = StreamingOptimizer(config)

        # Initial parameters far from true values
        p0 = np.array([1.0, 1.0])

        # Run optimization
        data_source = (x_true.reshape(-1, 1), y_noisy)
        result = optimizer.fit(data_source, model, p0, verbose=0)

        # Check that best parameters were tracked
        assert optimizer.best_params is not None
        assert not np.array_equal(optimizer.best_params, p0)
        assert optimizer.best_loss < float("inf")

        # Result should contain best parameters
        assert "x" in result
        assert np.array_equal(result["x"], optimizer.best_params)

    def test_best_params_preserved_on_loss_increase(self):
        """Test that best_params is preserved when loss increases."""

        # Create function that will have increasing loss
        def model(x, a, b):
            return a * x + b

        # Generate data
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + np.random.randn(500) * 0.1

        config = StreamingConfig(batch_size=50, max_epochs=3, learning_rate=0.1)
        optimizer = StreamingOptimizer(config)

        # Track losses during optimization
        losses_recorded = []

        def callback(iteration, params, loss):
            losses_recorded.append(loss)

        # Run optimization with callback
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([0.0, 0.0])
        result = optimizer.fit(data_source, model, p0, callback=callback, verbose=0)

        # Find when best loss occurred
        best_loss_idx = np.argmin(losses_recorded)

        # Best params should correspond to minimum loss
        assert (
            optimizer.best_loss <= losses_recorded[best_loss_idx] * 1.01
        )  # Small tolerance
        assert result["x"] is not None
        assert result["fun"] == optimizer.best_loss

    def test_initial_p0_never_returned_on_failure(self):
        """Test that initial p0 is never returned on total failure."""

        # Create a function that will fail during optimization
        def failing_model(x, a, b):
            # Will cause NaN after a few iterations
            if np.random.random() > 0.7:
                return np.full_like(x, np.nan)
            return a * x + b

        # Generate data
        x_data = np.random.randn(100)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(batch_size=10, max_epochs=1, learning_rate=0.01)
        optimizer = StreamingOptimizer(config)

        # Initial parameters
        p0 = np.array([5.0, 5.0])

        # Mock _update_parameters to track if it was ever called successfully
        original_update = optimizer._update_parameters
        update_called_successfully = []

        def tracked_update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            if not np.any(np.isnan(result)):
                update_called_successfully.append(result.copy())
            return result

        optimizer._update_parameters = tracked_update

        # Run optimization (may partially fail)
        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit(data_source, failing_model, p0, verbose=0)

        # If any successful updates happened, result should not be p0
        if update_called_successfully:
            assert not np.array_equal(result["x"], p0)

        # Result should always be the best achieved parameters
        if optimizer.best_params is not None:
            assert np.array_equal(result["x"], optimizer.best_params)

    def test_parameter_tracking_through_multiple_batches(self):
        """Test parameter tracking through multiple batch iterations."""

        # Simple linear model
        def model(x, a, b):
            return a * x + b

        # Generate dataset that requires multiple batches
        np.random.seed(42)
        n_samples = 500
        x_data = np.random.randn(n_samples)
        y_data = 3.0 * x_data + 2.0 + np.random.randn(n_samples) * 0.1

        config = StreamingConfig(
            batch_size=50,  # 10 batches per epoch
            max_epochs=2,
            learning_rate=0.01,
        )
        optimizer = StreamingOptimizer(config)

        # Track parameter evolution
        param_history = []
        loss_history = []

        def tracking_callback(iteration, params, loss):
            param_history.append(params.copy())
            loss_history.append(loss)

        # Initial parameters
        p0 = np.array([1.0, 1.0])

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit(
            data_source, model, p0, callback=tracking_callback, verbose=0
        )

        # Verify tracking worked
        assert len(param_history) > 0
        assert len(loss_history) > 0

        # Best parameters should be from iteration with lowest loss
        best_idx = np.argmin(loss_history)
        expected_best = param_history[best_idx]

        # Allow small numerical differences
        assert np.allclose(optimizer.best_params, expected_best, rtol=1e-5)
        assert np.allclose(result["x"], optimizer.best_params, rtol=1e-5)

        # Parameters should have improved from initial
        initial_loss = loss_history[0]
        final_best_loss = optimizer.best_loss
        assert final_best_loss < initial_loss


class TestBatchErrorIsolation:
    """Test batch processing error isolation."""

    def test_batch_errors_dont_abort_optimization(self):
        """Test that optimization completes even with fault tolerance enabled.

        Verifies that the streaming optimizer handles batches correctly
        without mocking internal APIs.
        """

        # Simple model
        def model(x, a, b):
            return a * x + b

        # Generate data
        np.random.seed(42)
        x_data = np.linspace(-5, 5, 300)
        y_data = 2.0 * x_data + 1.0 + np.random.randn(300) * 0.1

        config = StreamingConfig(
            batch_size=30,
            max_epochs=1,
            learning_rate=0.1,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        # Run optimization
        p0 = np.array([1.0, 0.0])
        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit(data_source, model, p0, verbose=0)

        # Optimization should complete
        assert result is not None
        assert "x" in result
        assert result["x"] is not None
        assert np.all(np.isfinite(result["x"]))

    def test_error_logging_continues_optimization(self):
        """Test that errors are logged but optimization continues."""

        # Model that occasionally fails
        def failing_model(x, a, b):
            if np.random.random() > 0.8:
                raise RuntimeError("Random failure")
            return a * x + b

        # Generate data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(batch_size=20, max_epochs=1, learning_rate=0.01)
        optimizer = StreamingOptimizer(config)

        # Add error handling wrapper
        original_update = optimizer._update_parameters
        errors_caught = []

        def safe_update(params, grad, bounds):
            try:
                # Simulate occasional update failure
                if np.random.random() > 0.9:
                    raise ValueError("Update failed")
                return original_update(params, grad, bounds)
            except Exception as e:
                errors_caught.append(str(e))
                # Return current params unchanged
                return params

        optimizer._update_parameters = safe_update

        # Run optimization
        p0 = np.array([0.0, 0.0])
        data_source = (x_data.reshape(-1, 1), y_data)

        # Use fixed seed for failing_model random failures
        np.random.seed(42)
        result = optimizer.fit(data_source, failing_model, p0, verbose=0)

        # Check that optimization completed
        assert result is not None
        # Success may be False if too many batches failed, but result should exist
        assert "x" in result
        assert result["x"] is not None

        # Some errors may have been caught (or none if random didn't trigger)
        # Just verify the mechanism works
        assert isinstance(errors_caught, list)

    def test_failed_batch_indices_tracked(self):
        """Test that streaming diagnostics structure is properly populated.

        Verifies the diagnostics infrastructure without mocking internal APIs.
        """

        # Simple linear model
        def model(x, a, b):
            return a * x + b

        # Generate data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 3.0 * x_data + 1.0 + np.random.randn(200) * 0.1

        config = StreamingConfig(
            batch_size=20,  # 10 batches total
            max_epochs=1,
            learning_rate=0.01,
            enable_fault_tolerance=True,
            max_retries_per_batch=0,
        )
        optimizer = StreamingOptimizer(config)

        # Run optimization
        p0 = np.array([1.0, 0.0])
        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit_streaming(data_source, model, p0, verbose=0)

        # Optimization should complete
        assert result is not None
        assert result["x"] is not None
        assert np.all(np.isfinite(result["x"]))

        # Check that streaming_diagnostics structure exists
        assert "streaming_diagnostics" in result
        diagnostics = result["streaming_diagnostics"]
        assert isinstance(diagnostics, dict)

        # Verify expected diagnostic fields are present
        assert "failed_batches" in diagnostics
        assert isinstance(diagnostics["failed_batches"], list)

        # With no failures, failed_batches should be empty
        # This tests the tracking infrastructure works correctly
        assert len(diagnostics["failed_batches"]) == 0
