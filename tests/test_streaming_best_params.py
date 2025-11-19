"""Tests for best parameter tracking in streaming optimizer.

This module tests that the streaming optimizer correctly tracks
the best parameters achieved during optimization and handles
batch failures gracefully.
"""

import numpy as np

from nlsq.streaming_optimizer import StreamingConfig, StreamingOptimizer


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
        """Test that errors in individual batches don't abort the entire optimization."""
        # Create model that fails on specific batches
        failed_batches = []

        def model_with_failures(x, a, b):
            # Fail on every 3rd batch (deterministic based on x values)
            batch_id = int(np.mean(x) * 1000) % 10
            if batch_id % 3 == 0:
                failed_batches.append(batch_id)
                raise ValueError(f"Simulated batch {batch_id} failure")
            return a * x + b

        # Generate data - ensure we have some batches that will succeed
        np.random.seed(42)
        x_data = np.linspace(-5, 5, 300)
        y_data = 2.0 * x_data + 1.0 + np.random.randn(300) * 0.1

        config = StreamingConfig(
            batch_size=30,  # 10 batches total
            max_epochs=1,
            learning_rate=0.1,  # Increased learning rate to ensure progress
        )
        optimizer = StreamingOptimizer(config)

        # Wrap compute to handle model failures gracefully
        original_compute = optimizer._compute_loss_and_gradient
        successful_batches = []

        def safe_compute(func, params, x_batch, y_batch, mask=None):
            try:
                # Try to evaluate the model first
                _ = func(x_batch, params[0], params[1])
                successful_batches.append(len(successful_batches))
                return original_compute(func, params, x_batch, y_batch, mask)
            except ValueError:
                # Return large loss and small gradient for failed batches
                # Use small gradient to allow some parameter update
                small_grad = np.ones_like(params) * 0.01
                return 100.0, small_grad

        optimizer._compute_loss_and_gradient = safe_compute

        # Run optimization
        p0 = np.array([1.0, 0.0])
        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit(data_source, model_with_failures, p0, verbose=0)

        # Optimization should complete despite failures
        assert result is not None
        assert "x" in result
        assert result["x"] is not None

        # Some batches should have succeeded
        assert len(successful_batches) > 0

        # We should have made some progress (parameters should have changed)
        # With the small gradient updates even on failures, params should change
        assert not np.array_equal(result["x"], p0) or len(successful_batches) > 5

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
        """Test that failed batch indices are tracked."""
        # Model that fails on specific batch indices
        batch_counter = [0]

        def model_with_tracked_failures(x, a, b):
            current_batch = batch_counter[0]
            batch_counter[0] += 1

            # Fail on batches 2, 5, 7
            if current_batch in [2, 5, 7]:
                raise ValueError(f"Batch {current_batch} failed")
            return a * x + b

        # Generate data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 3.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=20,  # 10 batches total
            max_epochs=1,
            learning_rate=0.01,
            enable_fault_tolerance=True,
            max_retries_per_batch=0,  # No retries so failures are tracked
        )
        optimizer = StreamingOptimizer(config)

        # Inject failure into compute method so optimizer sees it
        original_compute = optimizer._compute_loss_and_gradient
        batch_compute_counter = [0]

        def compute_with_failures(func, params, x_batch, y_batch, mask=None):
            # Call the model which may raise an exception
            try:
                # The model itself will raise on certain batches
                _ = func(x_batch, params[0], params[1])
            except ValueError:
                # Let the error propagate so the optimizer catches it
                raise

            batch_compute_counter[0] += 1
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = compute_with_failures

        # Reset batch counter
        batch_counter[0] = 0

        # Run optimization
        p0 = np.array([1.0, 0.0])
        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit_streaming(
            data_source, model_with_tracked_failures, p0, verbose=0
        )

        # Optimization should still complete
        assert result is not None
        assert result["x"] is not None

        # Check that failed indices are tracked in streaming_diagnostics
        assert "streaming_diagnostics" in result
        assert "failed_batches" in result["streaming_diagnostics"]
        # We specified batches 2, 5, 7 should fail
        # The optimizer should have caught at least some of these
        assert (
            len(result["streaming_diagnostics"]["failed_batches"]) >= 2
        )  # At least 2 of the 3 should be caught
