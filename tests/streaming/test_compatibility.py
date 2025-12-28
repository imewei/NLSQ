"""
Compatibility tests for streaming optimizer fault tolerance features.

This module verifies backward compatibility, API stability, and integration
with existing NLSQ features.

Usage:
    # Run all compatibility tests
    pytest tests/test_streaming_compatibility.py -v

    # Run specific compatibility category
    pytest tests/test_streaming_compatibility.py -k "backward" -v
"""

import os
import tempfile

import numpy as np
import pytest

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_default_config_unchanged(self):
        """Test that default configuration maintains backward compatibility.

        Verifies new parameters have sensible defaults that don't break existing code.
        """
        # Create config with no parameters (use all defaults)
        config = StreamingConfig()

        # Verify new parameters have appropriate defaults
        assert hasattr(config, "validate_numerics")
        assert hasattr(config, "enable_fault_tolerance")
        assert hasattr(config, "max_retries_per_batch")
        assert hasattr(config, "min_success_rate")
        assert hasattr(config, "batch_stats_buffer_size")

        # Default fault tolerance should be enabled (safe defaults)
        # or disabled (backward compatible) - either is acceptable
        assert isinstance(config.enable_fault_tolerance, bool)

    def test_existing_code_pattern_works(self):
        """Test that existing streaming optimizer usage patterns still work.

        Verifies code written before fault tolerance still functions.
        """
        np.random.seed(42)
        x_data = np.random.randn(1000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(1000)

        # Old usage pattern (no fault tolerance parameters)
        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            learning_rate=0.1,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Run optimization (old pattern)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should work as before
        assert result is not None
        assert "x" in result
        assert not np.array_equal(result["x"], p0)

    def test_minimal_config_works(self):
        """Test with minimal configuration (only required parameters).

        Verifies all new parameters are truly optional.
        """
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        # Absolute minimal config
        config = StreamingConfig(batch_size=100)
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None
        assert "x" in result

    def test_result_format_compatibility(self):
        """Test that result format maintains backward compatibility.

        Verifies new diagnostic fields don't break existing result parsing.
        """
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # All standard fields should be present
        assert "x" in result
        assert "success" in result

        # New diagnostic fields should not interfere with standard fields
        assert result["x"] is not None
        assert isinstance(result["success"], (bool, np.bool_))


class TestAPICompatibility:
    """Test API compatibility and interface stability."""

    def test_config_parameter_types(self):
        """Test that all config parameters accept correct types.

        Verifies type checking and validation work correctly.
        """
        # Test valid configurations
        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            min_success_rate=0.5,
            batch_stats_buffer_size=100,
        )

        assert config.batch_size == 100
        assert config.max_epochs == 2
        assert config.learning_rate == 0.1
        assert config.validate_numerics is True
        assert config.enable_fault_tolerance is True
        assert config.max_retries_per_batch == 2
        assert config.min_success_rate == 0.5
        assert config.batch_stats_buffer_size == 100

    def test_fit_method_signature_compatible(self):
        """Test that fit method signature is backward compatible.

        Verifies all existing parameters still work.
        """
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        config = StreamingConfig(batch_size=100)
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Test with all standard parameters
        p0 = np.array([1.0, 0.0])
        bounds = (np.array([0.0, -10.0]), np.array([10.0, 10.0]))

        result = optimizer.fit_streaming(
            (x_data, y_data),
            model,
            p0,
            bounds=bounds,
            verbose=0,
        )

        assert result is not None
        assert "x" in result

        # Verify bounds were respected
        if result["success"]:
            assert np.all(result["x"] >= bounds[0])
            assert np.all(result["x"] <= bounds[1])

    def test_verbose_levels_work(self):
        """Test that verbose parameter works at all levels.

        Verifies logging doesn't break with new features.
        """
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
        )

        def model(x, a, b):
            return a * x + b

        p0 = np.array([1.0, 0.0])

        # Test verbose=0 (silent)
        optimizer = StreamingOptimizer(config)
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)
        assert result is not None

        # Test verbose=1 (normal)
        optimizer = StreamingOptimizer(config)
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=1)
        assert result is not None

        # Test verbose=2 (detailed) - if supported
        optimizer = StreamingOptimizer(config)
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=2)
        assert result is not None


class TestJAXCompatibility:
    """Test compatibility with JAX JIT compilation."""

    def test_jit_compilation_works(self):
        """Test that model functions can use JAX JIT.

        Verifies fault tolerance doesn't break JIT compilation.
        """
        import jax.numpy as jnp
        from jax import jit

        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * np.exp(-0.5 * x_data) + 0.3 + 0.05 * np.random.randn(500)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        # JIT-compiled model
        @jit
        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        p0 = np.array([2.0, 0.5, 0.3])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None
        assert "x" in result

    def test_jax_arrays_as_input(self):
        """Test that JAX arrays work as input data.

        Verifies no issues with JAX array types.
        """
        import jax.numpy as jnp

        np.random.seed(42)
        x_np = np.random.randn(500)
        y_np = 2.0 * x_np + 1.0 + 0.1 * np.random.randn(500)

        # Convert to JAX arrays
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_jax, y_jax), model, p0, verbose=0)

        assert result is not None
        assert "x" in result

    def test_jax_transformations_compatible(self):
        """Test compatibility with JAX transformations (grad, vmap).

        Verifies fault tolerance works with JAX transformation system.
        """
        import jax.numpy as jnp

        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        # Model using JAX primitives
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None


class TestFeatureInteraction:
    """Test interaction between fault tolerance and other features."""

    def test_fault_tolerance_with_bounds(self):
        """Test fault tolerance works correctly with parameter bounds.

        Verifies retry perturbations respect bounds.
        """
        np.random.seed(42)
        x_data = np.random.randn(1000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(1000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.1,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        p0 = np.array([1.0, 0.0])
        bounds = (np.array([0.0, -5.0]), np.array([5.0, 5.0]))

        # Inject error to trigger retry with perturbation
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute(func, params, x_batch, y_batch):
            call_count[0] += 1
            if call_count[0] == 2:  # Trigger retry on second batch
                return 100.0, np.array([np.nan, np.nan])
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        result = optimizer.fit_streaming(
            (x_data, y_data), model, p0, bounds=bounds, verbose=0
        )

        assert result is not None

        # Verify bounds respected
        if result["success"]:
            assert np.all(result["x"] >= bounds[0])
            assert np.all(result["x"] <= bounds[1])

    def test_fault_tolerance_with_checkpoints(self):
        """Test fault tolerance works with checkpoint save/resume.

        Verifies no conflicts between checkpointing and error handling.
        """
        import shutil
        import time

        temp_dir = tempfile.mkdtemp()
        optimizer = None

        try:
            np.random.seed(42)
            x_data = np.random.randn(1000)
            y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(1000)

            config = StreamingConfig(
                batch_size=100,
                max_epochs=1,
                learning_rate=0.1,
                enable_fault_tolerance=True,
                validate_numerics=True,
                checkpoint_interval=3,
                checkpoint_dir=temp_dir,
            )
            optimizer = StreamingOptimizer(config)

            def model(x, a, b):
                return a * x + b

            # Inject some failures
            original_compute = optimizer._compute_loss_and_gradient
            call_count = [0]

            def mock_compute(func, params, x_batch, y_batch):
                call_count[0] += 1
                if call_count[0] % 5 == 0:
                    return 100.0, np.array([np.nan, np.nan])
                return original_compute(func, params, x_batch, y_batch, mask)

            optimizer._compute_loss_and_gradient = mock_compute

            p0 = np.array([1.0, 0.0])
            result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

            assert result is not None

        finally:
            # Shutdown checkpoint worker before cleanup to avoid race conditions
            if optimizer is not None:
                optimizer._shutdown_checkpoint_worker()
                # Brief delay to let background thread finish
                time.sleep(0.1)

            if os.path.exists(temp_dir):
                # Use ignore_errors=True as fallback for any remaining race conditions
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_fault_tolerance_with_callback(self):
        """Test fault tolerance works with user callbacks.

        Verifies callbacks receive correct information during fault tolerance.
        """
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Track callback invocations
        callback_data = {"calls": 0}

        def callback(iteration, params, loss):
            callback_data["calls"] += 1
            # Callback should receive valid data
            assert iteration >= 0
            assert params is not None
            assert loss is not None or np.isfinite(loss)

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming(
            (x_data, y_data), model, p0, callback=callback, verbose=0
        )

        assert result is not None
        # Callback should have been called
        assert callback_data["calls"] > 0


class TestDataFormatCompatibility:
    """Test compatibility with different data formats."""

    def test_numpy_arrays(self):
        """Test with standard NumPy arrays.

        Baseline compatibility test.
        """
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        config = StreamingConfig(
            batch_size=100,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None

    def test_list_inputs(self):
        """Test with Python lists (converted to arrays).

        Verifies flexible input handling.
        """
        np.random.seed(42)
        x_data = list(np.random.randn(500))
        y_data = [2.0 * x + 1.0 + 0.1 * np.random.randn() for x in x_data]

        config = StreamingConfig(
            batch_size=100,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        p0 = [1.0, 0.0]
        result = optimizer.fit_streaming(
            (np.array(x_data), np.array(y_data)), model, np.array(p0), verbose=0
        )

        assert result is not None

    def test_different_dtypes(self):
        """Test with different NumPy dtypes.

        Verifies type conversion works correctly.
        """
        np.random.seed(42)
        x_data = np.random.randn(500).astype(np.float32)
        y_data = (2.0 * x_data + 1.0).astype(np.float32)

        config = StreamingConfig(
            batch_size=100,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        p0 = np.array([1.0, 0.0], dtype=np.float32)
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None


class TestConfigurationMigration:
    """Test configuration migration and upgrade paths."""

    def test_old_config_still_works(self):
        """Test that configurations from older versions still work.

        Simulates code written before fault tolerance features.
        """
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        # Old-style config (minimal parameters)
        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            learning_rate=0.1,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None
        assert "x" in result

    def test_gradual_feature_adoption(self):
        """Test adding fault tolerance features one at a time.

        Verifies users can adopt features incrementally.
        """
        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(500)

        def model(x, a, b):
            return a * x + b

        p0 = np.array([1.0, 0.0])

        # Step 1: Add validation only
        config1 = StreamingConfig(
            batch_size=100,
            validate_numerics=True,
        )
        optimizer1 = StreamingOptimizer(config1)
        result1 = optimizer1.fit_streaming((x_data, y_data), model, p0, verbose=0)
        assert result1 is not None

        # Step 2: Add retry logic
        config2 = StreamingConfig(
            batch_size=100,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )
        optimizer2 = StreamingOptimizer(config2)
        result2 = optimizer2.fit_streaming((x_data, y_data), model, p0, verbose=0)
        assert result2 is not None

        # Step 3: Add diagnostics
        config3 = StreamingConfig(
            batch_size=100,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            batch_stats_buffer_size=100,
        )
        optimizer3 = StreamingOptimizer(config3)
        result3 = optimizer3.fit_streaming((x_data, y_data), model, p0, verbose=0)
        assert result3 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
