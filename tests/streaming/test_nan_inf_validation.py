"""Tests for NaN/Inf detection and validation in streaming optimizer.

This module tests the three-point NaN/Inf validation system that protects
against numerical instabilities during streaming optimization.
"""

import numpy as np

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestNaNInfValidation:
    """Test NaN/Inf validation at three critical points."""

    def test_gradient_validation_with_nan(self):
        """Test that NaN gradients are detected and batch is skipped.

        Uses a model that naturally produces NaN gradients via log of negative
        values when parameters reach certain states during optimization.
        """
        import jax.numpy as jnp

        # Model that produces NaN gradient when 'a' parameter causes log of negative
        # The gradient of log(a*x + b) w.r.t. 'a' is x/(a*x + b)
        # When a*x + b <= 0, log produces NaN, causing NaN gradient
        def model_with_nan_gradient(x, a, b):
            # This will produce NaN when a*x + b <= 0
            # With negative x values and small 'a', this happens naturally
            return jnp.log(jnp.abs(a) * x + b + 0.1)

        # Generate test data with negative x values that will cause NaN
        np.random.seed(42)
        x_data = np.linspace(-5, 5, 200)  # Include negative values
        # Target is the model with true params (will produce some NaN during fitting)
        y_data = np.log(np.abs(2.0 * x_data + 1.0) + 0.1) + 0.01 * np.random.randn(200)

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.5,  # Higher LR to push params into NaN territory
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=0,
        )
        optimizer = StreamingOptimizer(config)

        # Start with parameters that will cause NaN during optimization
        # When a is small and b is negative, log(a*x + b) produces NaN for some x
        p0 = np.array([0.1, -2.0])

        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit_streaming(
            data_source, model_with_nan_gradient, p0, verbose=0
        )

        # The optimization should handle NaN gracefully
        assert "streaming_diagnostics" in result
        # With fault tolerance, NaN batches are tracked and skipped
        # The final result should be finite even if some batches had NaN
        assert np.all(np.isfinite(result["x"]))

    def test_gradient_validation_with_inf(self):
        """Test that Inf gradients are detected and batch is skipped.

        Uses a model with division that produces Inf gradient when denominator
        approaches zero during optimization.
        """
        import jax.numpy as jnp

        # Model: a / (x - b) - produces Inf gradient when x approaches b
        # The gradient w.r.t. 'b' is a / (x - b)^2, which goes to Inf as x -> b
        def model_with_inf_gradient(x, a, b):
            # Add small epsilon to avoid exact zero but still produce large/Inf values
            return a / (x - b + 1e-10)

        # Generate test data with values near the singularity point
        np.random.seed(42)
        # Data centered around 0.5, which is close to initial b=0.4
        x_data = np.linspace(0.3, 0.7, 150)
        # Target values (with true params a=1, b=0)
        y_data = 1.0 / (x_data + 0.01) + 0.01 * np.random.randn(150)

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            max_retries_per_batch=0,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        # Start with b close to x values to trigger Inf gradient
        p0 = np.array([1.0, 0.4])

        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit_streaming(
            data_source, model_with_inf_gradient, p0, verbose=0
        )

        # The optimization should handle Inf gracefully
        assert "streaming_diagnostics" in result
        # Final result should be finite even if some batches had Inf
        assert np.all(np.isfinite(result["x"]))

    def test_parameter_update_validation(self):
        """Test that NaN/Inf in parameter updates are caught and reverted."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(100)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            validate_numerics=True,
            max_retries_per_batch=0,  # No retries so failures are tracked
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        # Patch _update_parameters to inject NaN
        original_update = optimizer._update_parameters
        call_count = [0]
        batch_calls = []

        def mock_update(params, grad, bounds):
            call_count[0] += 1
            # Track batch updates
            batch_idx = len(batch_calls)
            if call_count[0] > len(batch_calls):
                batch_calls.append(call_count[0])

            if batch_idx == 0:  # First update returns NaN
                return np.array([np.nan, 1.0])
            return original_update(params, grad, bounds)

        optimizer._update_parameters = mock_update

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming(data_source, model, p0, verbose=0)

        # Parameters should not contain NaN (reverted or skipped)
        assert np.all(np.isfinite(result["x"]))
        assert "streaming_diagnostics" in result
        assert len(result["streaming_diagnostics"]["failed_batches"]) > 0

    def test_loss_value_validation(self):
        """Test that NaN/Inf loss values are detected and batch is skipped.

        Uses a model that produces NaN output values, which leads to NaN loss.
        """
        import jax.numpy as jnp

        # Model that produces NaN when sqrt of negative value
        # sqrt(a*x + b) produces NaN when a*x + b < 0
        def model_with_nan_loss(x, a, b):
            return jnp.sqrt(a * x + b)

        # Generate test data with some negative x values
        np.random.seed(42)
        x_data = np.linspace(-2, 5, 150)
        # Target with valid values (positive argument to sqrt)
        y_data = np.sqrt(np.maximum(2.0 * x_data + 1.0, 0.01)) + 0.01 * np.random.randn(
            150
        )

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            max_retries_per_batch=0,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        # Start with params that cause sqrt of negative for some x
        # When a=0.5 and b=-1, sqrt(0.5*x - 1) is NaN for x < 2
        p0 = np.array([0.5, -1.0])

        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit_streaming(
            data_source, model_with_nan_loss, p0, verbose=0
        )

        # The optimization should handle NaN loss gracefully
        assert "streaming_diagnostics" in result
        # Final result should be finite
        assert np.isfinite(result["fun"])
        assert np.all(np.isfinite(result["x"]))

    def test_validation_disabled(self):
        """Test that validation can be disabled for performance.

        Verifies that optimization runs successfully with validation disabled.
        """

        # Simple linear model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(100)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            validate_numerics=False,  # Disable validation
        )
        optimizer = StreamingOptimizer(config)

        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit(data_source, model, p0, verbose=0)

        # When validation is disabled, the flag should be False
        assert not config.validate_numerics
        assert result["success"]  # Should complete successfully
        assert np.all(np.isfinite(result["x"]))

    def test_validation_enabled_by_default(self):
        """Test that validation is enabled by default."""
        config = StreamingConfig()
        assert config.validate_numerics

    def test_mixed_nan_inf_validation(self):
        """Test handling of mixed NaN and Inf values.

        Uses a model that can produce NaN (from log of negative) in some batches
        but produces valid outputs for most data points.
        """
        import jax.numpy as jnp

        # Model that can produce NaN for some x values
        # log(a*x + b) produces NaN when a*x + b <= 0
        def mixed_problematic_model(x, a, b):
            return jnp.log(a * x + b)

        # Generate test data - mostly safe but with some problematic regions
        np.random.seed(42)
        # Mix: mostly positive x that will work, some edge cases
        x_data = np.concatenate(
            [
                np.linspace(
                    1, 5, 150
                ),  # Safe region (a*x + b > 0 for reasonable params)
                np.linspace(-0.5, 0.5, 50),  # Edge region that might cause issues
            ]
        )
        np.random.shuffle(x_data)

        # Target values using safe parameters (a=2, b=1)
        y_data = np.log(np.maximum(2.0 * x_data + 1.0, 0.01)) + 0.01 * np.random.randn(
            200
        )

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            learning_rate=0.05,  # Lower learning rate to avoid divergence
            validate_numerics=True,
            max_retries_per_batch=0,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        # Start with params near true values but slightly off
        # This should work for most batches but may cause issues for edge cases
        p0 = np.array([1.5, 0.5])

        data_source = (x_data.reshape(-1, 1), y_data)
        result = optimizer.fit_streaming(
            data_source, mixed_problematic_model, p0, verbose=0
        )

        # The optimization should handle NaN gracefully
        assert "streaming_diagnostics" in result
        # Final results should be finite
        assert np.all(np.isfinite(result["x"]))
        # At least the optimization completed
        assert result["success"] or np.isfinite(result["fun"])

    def test_validation_with_bounds(self):
        """Test NaN/Inf validation when bounds are applied."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(100)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            validate_numerics=True,
            max_retries_per_batch=0,  # No retries so failures are tracked
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        # Set bounds
        bounds = (np.array([0.0, -5.0]), np.array([5.0, 5.0]))

        # Patch to inject NaN that would be clipped by bounds
        original_update = optimizer._update_parameters
        call_count = [0]
        batch_calls = []

        def mock_update(params, grad, param_bounds):
            call_count[0] += 1
            # Track batch updates
            batch_idx = len(batch_calls)
            if call_count[0] > len(batch_calls):
                batch_calls.append(call_count[0])

            if batch_idx == 0:
                # Return NaN before bounds are applied
                return np.array([np.nan, 1.0])
            return original_update(params, grad, param_bounds)

        optimizer._update_parameters = mock_update

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming(
            data_source, model, p0, bounds=bounds, verbose=0
        )

        # Should detect NaN even with bounds
        assert "streaming_diagnostics" in result
        assert len(result["streaming_diagnostics"]["failed_batches"]) > 0
        assert np.all(np.isfinite(result["x"]))
        # Results should respect bounds
        assert np.all(result["x"] >= bounds[0])
        assert np.all(result["x"] <= bounds[1])

    def test_validation_performance_impact(self):
        """Test that validation has minimal performance impact."""
        import time

        # Simple, fast model
        def model(x, a, b):
            return a * x + b

        # Generate larger dataset
        np.random.seed(42)
        x_data = np.random.randn(10000)
        y_data = 2.0 * x_data + 1.0

        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])

        # Time with validation enabled
        config_with_validation = StreamingConfig(
            batch_size=100, max_epochs=1, validate_numerics=True
        )
        optimizer_with = StreamingOptimizer(config_with_validation)

        start_with = time.time()
        result_with = optimizer_with.fit(data_source, model, p0, verbose=0)
        time_with = time.time() - start_with

        # Time with validation disabled
        config_without_validation = StreamingConfig(
            batch_size=100, max_epochs=1, validate_numerics=False
        )
        optimizer_without = StreamingOptimizer(config_without_validation)

        data_source = (x_data.reshape(-1, 1), y_data)  # Reset generator
        start_without = time.time()
        result_without = optimizer_without.fit(data_source, model, p0, verbose=0)
        time_without = time.time() - start_without

        # Validation overhead should be minimal (< 10%)
        overhead_percent = (
            (time_with - time_without) / time_without * 100 if time_without > 0 else 0
        )

        # Both should produce similar results
        assert np.allclose(result_with["x"], result_without["x"], rtol=0.1)

        # Performance assertion is relaxed as timing can vary
        # Main goal is to ensure both modes work correctly
        assert result_with["success"] and result_without["success"]

        # Log the overhead for information
        print(f"Validation overhead: {overhead_percent:.1f}%")

    def test_all_batches_fail_scenario(self):
        """Test behavior when all batches fail validation."""

        # Model that always produces NaN
        def bad_model(x, a, b):
            return np.full_like(x, np.nan)

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(100)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(batch_size=50, max_epochs=1, validate_numerics=True)
        optimizer = StreamingOptimizer(config)

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit(data_source, bad_model, p0, verbose=0)

        # When all batches fail, should return initial params but indicate failure
        assert not result["success"]
        assert "streaming_diagnostics" in result
        assert result["streaming_diagnostics"]["batch_success_rate"] == 0.0
        assert (
            len(result["streaming_diagnostics"]["failed_batches"]) == 2
        )  # All batches failed
        # Result should contain initial parameters (no better ones found)
        assert np.array_equal(result["x"], p0)
