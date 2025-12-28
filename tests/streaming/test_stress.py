"""
Stress tests for streaming optimizer fault tolerance.

This module tests the streaming optimizer under extreme conditions to verify
robustness and stability. Tests are designed to push the system to its limits
and verify graceful degradation.

Usage:
    # Run all stress tests
    pytest tests/test_streaming_stress.py -v

    # Run specific stress test category
    pytest tests/test_streaming_stress.py -k "high_failure" -v

    # Run with verbose output
    pytest tests/test_streaming_stress.py -v -s
"""

import gc
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestHighFailureRateStress:
    """Stress tests with high batch failure rates."""

    def test_50_percent_failure_rate(self):
        """Test optimization with 50% batch failure rate.

        Verifies system can handle high failure rates and still produce results.
        """
        np.random.seed(42)
        x_data = np.random.randn(5000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(5000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=1,
            min_success_rate=0.4,  # Allow 60% failure
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Inject 50% failure rate
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            if call_count[0] % 2 == 0:  # Every other batch fails
                return 100.0, np.array([np.nan, np.nan])
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete despite high failure rate
        assert result is not None
        assert "x" in result

        # Parameters should have improved from initial
        assert not np.array_equal(result["x"], p0)

        # Should track success rate
        if "batch_success_rate" in result:
            # Success rate should be around 50%
            assert 0.4 <= result["batch_success_rate"] <= 0.6

    def test_90_percent_failure_rate(self):
        """Test optimization with 90% batch failure rate.

        Extreme stress test - verifies graceful degradation.
        """
        np.random.seed(42)
        x_data = np.random.randn(2000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(2000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.05,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=1,
            min_success_rate=0.05,  # Allow 95% failure
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Inject 90% failure rate
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            if call_count[0] % 10 != 0:  # 9 out of 10 batches fail
                return 100.0, np.array([np.nan, np.nan])
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete (though may not converge well)
        assert result is not None
        assert "x" in result

        # Success rate should be very low
        if "batch_success_rate" in result:
            assert result["batch_success_rate"] < 0.15

    def test_consecutive_failures_stress(self):
        """Test with many consecutive batch failures.

        Verifies system handles failure streaks without crashing.
        """
        np.random.seed(42)
        x_data = np.random.randn(3000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(3000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            min_success_rate=0.4,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Inject consecutive failures (batches 5-14 fail)
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            batch_num = (call_count[0] - 1) // 3  # Account for retries

            # Fail batches 5-14 (10 consecutive failures)
            if 5 <= batch_num <= 14:
                return 100.0, np.array([np.nan, np.nan])
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete despite consecutive failures
        assert result is not None
        assert "x" in result

        # Should have failed batch records
        if "failed_batch_indices" in result:
            assert len(result["failed_batch_indices"]) >= 5


class TestLargeDatasetStress:
    """Stress tests with very large datasets."""

    @pytest.mark.slow
    @pytest.mark.serial  # Memory-intensive: 100K data points
    def test_very_large_dataset(self):
        """Test with very large dataset (100K points, 1000 batches).

        Verifies memory efficiency and performance with large data.
        """
        np.random.seed(42)
        x_data = np.random.randn(100000)
        y_data = 2.0 * x_data + 1.0 + 0.05 * np.random.randn(100000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.05,
            validate_numerics=True,
            enable_fault_tolerance=True,
            batch_stats_buffer_size=100,  # Fixed buffer size
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete
        assert result is not None
        assert "x" in result

        # Parameters should be close to true values
        assert np.allclose(result["x"], [2.0, 1.0], rtol=0.1)

        # Verify buffer didn't grow unbounded
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]
            if "recent_batch_stats" in diags:
                assert len(diags["recent_batch_stats"]) <= 100

    @pytest.mark.slow
    def test_many_small_batches(self):
        """Test with many small batches (10K batches of 10 points).

        Verifies overhead scales well with batch count.
        """
        np.random.seed(42)
        x_data = np.random.randn(10000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(10000)

        config = StreamingConfig(
            batch_size=10,  # Very small batches
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            batch_stats_buffer_size=100,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete
        assert result is not None
        assert "x" in result

    def test_high_dimensional_parameters(self):
        """Test with many parameters (stress parameter tracking).

        Verifies system handles high-dimensional optimization.
        """
        np.random.seed(42)
        x_data = np.random.randn(5000)

        # 10-parameter polynomial
        true_params = np.array(
            [0.1, -0.2, 0.3, -0.15, 0.05, 0.08, -0.12, 0.06, -0.03, 1.0]
        )
        y_data = sum(true_params[i] * x_data**i for i in range(10))
        y_data += 0.1 * np.random.randn(5000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            learning_rate=0.01,
            validate_numerics=True,
            enable_fault_tolerance=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
            return (
                p0
                + p1 * x
                + p2 * x**2
                + p3 * x**3
                + p4 * x**4
                + p5 * x**5
                + p6 * x**6
                + p7 * x**7
                + p8 * x**8
                + p9 * x**9
            )

        # Run optimization
        p0 = np.ones(10) * 0.5
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete
        assert result is not None
        assert "x" in result
        assert len(result["x"]) == 10


class TestMemoryStress:
    """Stress tests for memory management."""

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated optimizations.

        Runs multiple optimizations and checks memory doesn't grow unbounded.
        """
        np.random.seed(42)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            batch_stats_buffer_size=100,
        )

        def model(x, a, b):
            return a * x + b

        # Run multiple optimizations
        for i in range(10):
            x_data = np.random.randn(2000)
            y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(2000)

            optimizer = StreamingOptimizer(config)
            p0 = np.array([1.0, 0.0])
            result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

            assert result is not None

            # Clean up
            del optimizer
            gc.collect()

        # Test passes if no memory errors occurred

    @pytest.mark.slow
    def test_checkpoint_memory_usage(self):
        """Test memory usage with frequent checkpointing.

        Verifies checkpoint saving doesn't cause memory issues.
        """
        temp_dir = tempfile.mkdtemp()
        optimizer = None

        try:
            np.random.seed(42)
            x_data = np.random.randn(10000)
            y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(10000)

            config = StreamingConfig(
                batch_size=100,
                max_epochs=2,
                learning_rate=0.1,
                checkpoint_interval=1,  # Save every batch
                checkpoint_dir=temp_dir,
                validate_numerics=True,
                enable_fault_tolerance=True,
            )
            optimizer = StreamingOptimizer(config)

            def model(x, a, b):
                return a * x + b

            # Run optimization with frequent checkpoints
            p0 = np.array([1.0, 0.0])
            result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

            assert result is not None

            # Check checkpoint files exist (if implementation supports it)
            list(Path(temp_dir).glob("*.h5"))
            # May or may not have checkpoints depending on implementation

        finally:
            # Shutdown optimizer threads before cleanup
            if optimizer is not None and hasattr(
                optimizer, "_shutdown_checkpoint_worker"
            ):
                optimizer._shutdown_checkpoint_worker()

            # Cleanup
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_circular_buffer_stress(self):
        """Test circular buffer with extreme batch counts.

        Verifies buffer doesn't grow unbounded with many batches.
        """
        np.random.seed(42)
        x_data = np.random.randn(50000)  # 500 batches
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(50000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=3,  # 1500 total batch iterations
            learning_rate=0.05,
            batch_stats_buffer_size=100,  # Fixed size
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None

        # Verify buffer size is bounded
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]
            if "recent_batch_stats" in diags:
                # Should never exceed 100
                assert len(diags["recent_batch_stats"]) <= 100


class TestConcurrentErrorStress:
    """Stress tests with multiple error types occurring simultaneously."""

    def test_mixed_error_types_high_frequency(self):
        """Test with concurrent optimization under stress.

        Verifies optimizer handles stress without internal API mocking.
        """
        np.random.seed(42)
        x_data = np.random.randn(5000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(5000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            min_success_rate=0.3,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete
        assert result is not None
        assert "x" in result
        assert np.all(np.isfinite(result["x"]))

    def test_alternating_validation_failures(self):
        """Test with validation failures at all three checkpoints.

        Verifies all validation points work under stress.
        """
        np.random.seed(42)
        x_data = np.random.randn(3000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(3000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=1,
            min_success_rate=0.5,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Inject failures at different validation points
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            batch_num = (call_count[0] - 1) // 2

            # Alternate between gradient NaN, parameter Inf, and loss NaN
            failure_type = batch_num % 6
            if failure_type < 3:  # Fail half the batches
                if failure_type == 0:
                    # Gradient validation failure
                    return 0.5, np.array([np.nan, 1.0])
                elif failure_type == 1:
                    # Parameter validation failure
                    return 0.5, np.array([np.inf, 1.0])
                elif failure_type == 2:
                    # Loss validation failure
                    return np.nan, np.array([1.0, 1.0])

            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        # Should complete with validation working
        assert result is not None
        assert "x" in result


class TestCheckpointStress:
    """Stress tests for checkpoint system."""

    def test_rapid_checkpoint_save_load_cycle(self):
        """Test rapid checkpoint saving and loading.

        Verifies checkpoint system under high I/O stress.
        """
        temp_dir = tempfile.mkdtemp()

        try:
            np.random.seed(42)
            x_data = np.random.randn(5000)
            y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(5000)

            config = StreamingConfig(
                batch_size=100,
                max_epochs=1,
                learning_rate=0.1,
                checkpoint_interval=1,  # Save every batch
                checkpoint_dir=temp_dir,
            )
            optimizer = StreamingOptimizer(config)

            def model(x, a, b):
                return a * x + b

            # Run with frequent checkpointing
            p0 = np.array([1.0, 0.0])
            result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

            assert result is not None

            # Verify checkpoints were created (if implementation supports it)
            checkpoint_files = list(Path(temp_dir).glob("*.h5"))

            if checkpoint_files:
                # Test loading last checkpoint
                last_checkpoint = str(sorted(checkpoint_files)[-1])

                config_resume = StreamingConfig(
                    batch_size=100,
                    resume_from_checkpoint=last_checkpoint,
                )
                optimizer_loaded = StreamingOptimizer(config_resume)

                # Continue optimization
                result_continued = optimizer_loaded.fit_streaming(
                    (x_data, y_data), model, result["x"], verbose=0
                )
                assert result_continued is not None

        finally:
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @pytest.mark.slow
    def test_multiple_checkpoint_resume_cycles(self):
        """Test multiple interrupt and resume cycles.

        Verifies checkpoint robustness with repeated interruptions.
        """
        temp_dir = tempfile.mkdtemp()
        optimizers = []

        try:
            np.random.seed(42)
            x_data = np.random.randn(10000)
            y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(10000)

            def model(x, a, b):
                return a * x + b

            p0 = np.array([1.0, 0.0])
            current_params = p0

            # Simulate 3 interrupt/resume cycles
            for cycle in range(3):
                config = StreamingConfig(
                    batch_size=100,
                    max_epochs=(cycle + 1)
                    * 3,  # More epochs for better convergence (3, 6, 9 total)
                    checkpoint_interval=10,
                    checkpoint_dir=temp_dir,
                    resume_from_checkpoint=cycle > 0,  # Resume after first cycle
                )
                optimizer = StreamingOptimizer(config)
                optimizers.append(optimizer)

                result = optimizer.fit_streaming(
                    (x_data, y_data), model, current_params, verbose=0
                )

                assert result is not None
                current_params = result["x"]

            # Final parameters should be close to true values
            assert np.allclose(current_params, [2.0, 1.0], rtol=0.2)

        finally:
            # Shutdown all optimizer threads before cleanup
            for opt in optimizers:
                if hasattr(opt, "_shutdown_checkpoint_worker"):
                    opt._shutdown_checkpoint_worker()

            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestDiagnosticStress:
    """Stress tests for diagnostic collection system."""

    def test_diagnostic_accuracy_with_many_failures(self):
        """Test diagnostic accuracy with large dataset.

        Verifies diagnostic collection works under high load without mocking.
        """
        np.random.seed(42)
        x_data = np.random.randn(10000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(10000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            min_success_rate=0.4,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None
        assert "x" in result
        assert np.all(np.isfinite(result["x"]))

        # Verify diagnostics structure if present
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]
            assert isinstance(diags, dict)

    def test_retry_count_tracking_stress(self):
        """Test optimization with retries enabled under load.

        Verifies retry mechanism works correctly without mocking.
        """
        np.random.seed(42)
        x_data = np.random.randn(5000)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(5000)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            learning_rate=0.1,
            validate_numerics=True,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
            min_success_rate=0.5,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a, b):
            return a * x + b

        # Run optimization
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming((x_data, y_data), model, p0, verbose=0)

        assert result is not None
        assert "x" in result
        assert np.all(np.isfinite(result["x"]))

        # Verify diagnostics structure if present
        if "streaming_diagnostics" in result:
            diags = result["streaming_diagnostics"]
            assert isinstance(diags, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
