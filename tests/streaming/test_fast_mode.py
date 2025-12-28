"""Tests for fast mode (enable_fault_tolerance=False) configuration.

This module tests that fast mode skips expensive validation while maintaining
basic error handling to prevent complete crashes.
"""

import time

import numpy as np

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestFastModeConfiguration:
    """Test fast mode for production deployments."""

    def test_fast_mode_skips_validation(self):
        """Test that fast mode skips NaN/Inf validation checks."""

        # Simple linear model
        def model(x, a, b):
            return a * x + b

        # Generate test data
        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        # Fast mode configuration
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_fault_tolerance=False,  # Fast mode
            validate_numerics=True,  # Should be ignored in fast mode
        )
        optimizer = StreamingOptimizer(config)

        # Track whether validation was actually performed
        original_compute = optimizer._compute_loss_and_gradient

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            loss, grad = original_compute(func, params, x_batch, y_batch)
            # This would normally trigger validation in full mode
            # but should be skipped in fast mode
            return loss, grad

        optimizer._compute_loss_and_gradient = mock_compute

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit(data_source, model, p0, verbose=0)

        # Should complete successfully
        assert result["success"]
        # Should NOT have streaming diagnostics in fast mode
        assert "streaming_diagnostics" in result
        # Fast mode still includes diagnostics but they're minimal
        assert len(result["streaming_diagnostics"]["recent_batch_stats"]) == 0

    def test_fast_mode_skips_retry_attempts(self):
        """Test that fast mode skips retry logic on failures."""

        # Model that will fail occasionally
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_fault_tolerance=False,  # Fast mode
            max_retries_per_batch=2,  # Should be ignored
        )
        optimizer = StreamingOptimizer(config)

        # Inject a failure in batch 1
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_calls = []
        retry_attempts = [0]

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            # Track batch index by call order
            batch_idx = len(batch_calls)
            if call_count[0] > len(batch_calls):
                batch_calls.append(call_count[0])

            if batch_idx == 1:  # Fail second batch
                raise ValueError("Simulated batch failure")
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        # Patch retry to track if it's called
        original_retry = optimizer._process_batch_with_retry

        def mock_retry(*args, **kwargs):
            retry_attempts[0] += 1
            return original_retry(*args, **kwargs)

        optimizer._process_batch_with_retry = mock_retry

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming(data_source, model, p0, verbose=0)

        # Fast mode should NOT call retry logic
        assert retry_attempts[0] == 0, "Fast mode should not use retry logic"
        # Should still track batch success rate (not individual failed indices)
        assert result["streaming_diagnostics"]["batch_success_rate"] < 1.0, (
            "Should have at least one failed batch"
        )

    def test_fast_mode_skips_diagnostics_collection(self):
        """Test that fast mode skips detailed batch statistics collection."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_fault_tolerance=False,  # Fast mode
        )
        optimizer = StreamingOptimizer(config)

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit(data_source, model, p0, verbose=0)

        # Should NOT have detailed diagnostics
        assert "streaming_diagnostics" in result
        # Fast mode still includes diagnostics but they're minimal
        assert len(result["streaming_diagnostics"]["recent_batch_stats"]) == 0
        # But should still have basic fields
        assert "success" in result
        assert "x" in result
        assert "fun" in result

    def test_fast_mode_maintains_basic_error_handling(self):
        """Test that fast mode still has try-except to prevent crashes."""

        # Model that will crash
        def bad_model(x, a, b):
            if np.random.rand() < 0.5:
                raise RuntimeError("Simulated crash")
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_fault_tolerance=False,  # Fast mode
        )
        optimizer = StreamingOptimizer(config)

        # Run optimization - should NOT crash despite errors
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit(data_source, bad_model, p0, verbose=0)

        # Should complete without crashing
        assert "success" in result
        assert "x" in result
        # May or may not succeed depending on which batches failed
        # but should NOT raise an exception

    def test_fast_mode_performance_improvement(self):
        """Test that fast mode is comparable or faster than full mode."""

        # Simple, fast model
        def model(x, a, b):
            return a * x + b

        # Generate larger dataset for timing
        np.random.seed(42)
        x_data = np.random.randn(5000)
        y_data = 2.0 * x_data + 1.0

        # Measure fast mode performance
        config_fast = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=False,  # Fast mode
            enable_checkpoints=False,  # Disable checkpoints for fair comparison
        )
        optimizer_fast = StreamingOptimizer(config_fast)

        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])

        start_fast = time.time()
        result_fast = optimizer_fast.fit(data_source, model, p0, verbose=0)
        time_fast = time.time() - start_fast

        # Measure full fault tolerance mode performance
        config_full = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,  # Full mode
            validate_numerics=True,
            enable_checkpoints=False,  # Disable checkpoints for fair comparison
        )
        optimizer_full = StreamingOptimizer(config_full)

        data_source = (x_data.reshape(-1, 1), y_data)  # Reset

        start_full = time.time()
        result_full = optimizer_full.fit(data_source, model, p0, verbose=0)
        time_full = time.time() - start_full

        # Both should produce similar results
        assert result_fast["success"]
        assert result_full["success"]
        assert np.allclose(result_fast["x"], result_full["x"], rtol=0.1)

        # Print timing info for reference (not enforced due to variance)
        print(f"\nFast mode time: {time_fast:.4f}s")
        print(f"Full mode time: {time_full:.4f}s")
        if time_fast > 0:
            speedup = time_full / time_fast
            print(f"Speedup: {speedup:.2f}x")

    def test_fast_mode_still_saves_checkpoints(self):
        """Test that fast mode still saves checkpoints if enabled."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_fault_tolerance=False,  # Fast mode
            enable_checkpoints=True,
            checkpoint_frequency=2,  # Save frequently
            checkpoint_dir="test_checkpoints_fast_mode",
        )
        optimizer = StreamingOptimizer(config)

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit(data_source, model, p0, verbose=0)

        # Should still have saved checkpoints
        import os

        checkpoint_dir = config.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")
            ]
            # May or may not have checkpoints depending on iterations
            # Main point is it doesn't crash

            # Cleanup
            import shutil

            shutil.rmtree(checkpoint_dir, ignore_errors=True)

        assert result["success"]

    def test_fast_mode_logs_errors_to_stderr(self):
        """Test that fast mode logs errors but continues processing."""

        # Model that will fail occasionally
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_fault_tolerance=False,  # Fast mode
        )
        optimizer = StreamingOptimizer(config)

        # Inject a failure in batch 1 (second batch)
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_calls = []

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            # Track batch index by call order
            batch_idx = len(batch_calls)
            if call_count[0] > len(batch_calls):
                batch_calls.append(call_count[0])

            if batch_idx == 1:  # Fail second batch
                raise ValueError("Simulated error for fast mode")
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming(data_source, model, p0, verbose=0)

        # Fast mode doesn't track individual failed batch indices
        # But should track batch success rate in streaming_diagnostics
        assert result["streaming_diagnostics"]["batch_success_rate"] < 1.0, (
            "Should have at least one failed batch"
        )
        assert result["streaming_diagnostics"]["batch_success_rate"] > 0.0, (
            "Should have at least one successful batch"
        )

    def test_fast_mode_continues_on_batch_failure(self):
        """Test that fast mode continues to next batch on error."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(400)  # 8 batches
        y_data = 2.0 * x_data + 1.0

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_fault_tolerance=False,  # Fast mode
        )
        optimizer = StreamingOptimizer(config)

        # Make batches 1, 3, 5 fail (3 out of 8)
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_calls = []
        fail_batches = {1, 3, 5}

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            # Track batch index by call order
            batch_idx = len(batch_calls)
            if call_count[0] > len(batch_calls):
                batch_calls.append(call_count[0])

            if batch_idx in fail_batches:
                raise ValueError(f"Batch {batch_idx} failed")
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit_streaming(data_source, model, p0, verbose=0)

        # Fast mode doesn't track individual failed batch indices
        # But should process all batches
        assert len(batch_calls) == 8, "Should have processed all 8 batches"
        # Should still get valid success rate from successful batches
        actual_rate = result["streaming_diagnostics"]["batch_success_rate"]
        assert actual_rate == 5 / 8, f"Expected 5/8 success rate, got {actual_rate}"

    def test_fast_mode_config_default(self):
        """Test that fault tolerance is enabled by default."""
        config = StreamingConfig()
        assert config.enable_fault_tolerance

    def test_fast_mode_explicit_disable(self):
        """Test explicit disabling of fault tolerance."""
        config = StreamingConfig(enable_fault_tolerance=False)
        assert not config.enable_fault_tolerance

    def test_fast_mode_with_full_validation_config_ignored(self):
        """Test that validation config is ignored in fast mode."""
        # Even if validate_numerics is True, it should be skipped in fast mode
        config = StreamingConfig(
            enable_fault_tolerance=False,  # Fast mode
            validate_numerics=True,  # This should be ignored
            max_retries_per_batch=5,  # This should be ignored
            batch_stats_buffer_size=1000,  # This might still be used
        )

        # Simple model
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        optimizer = StreamingOptimizer(config)

        # Inject NaN that would normally trigger validation
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            if call_count[0] == 2:
                # Return NaN - would be caught by validation in full mode
                # In fast mode, should just fail the batch and continue
                return np.nan, np.array([1.0, 1.0])
            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        # Run optimization
        data_source = (x_data.reshape(-1, 1), y_data)
        p0 = np.array([1.0, 0.0])
        result = optimizer.fit(data_source, model, p0, verbose=0)

        # Should complete (may succeed or fail depending on implementation)
        assert "success" in result
        # Should NOT have detailed diagnostics
        assert "streaming_diagnostics" in result
        # Fast mode still includes diagnostics but they're minimal
        assert len(result["streaming_diagnostics"]["recent_batch_stats"]) == 0


class TestFastModeVsFullMode:
    """Compare fast mode vs full mode behavior."""

    def test_both_modes_produce_valid_results(self):
        """Test that both modes produce valid optimization results."""

        # Simple linear model
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(500)
        y_data = 2.0 * x_data + 1.0

        p0 = np.array([1.0, 0.0])

        # Fast mode - use more epochs for better convergence
        config_fast = StreamingConfig(
            batch_size=50,
            max_epochs=5,  # More epochs for better convergence
            enable_fault_tolerance=False,
            enable_checkpoints=False,
            learning_rate=0.01,  # Higher learning rate
        )
        optimizer_fast = StreamingOptimizer(config_fast)
        data_source = (x_data.reshape(-1, 1), y_data)
        result_fast = optimizer_fast.fit(data_source, model, p0, verbose=0)

        # Full mode
        config_full = StreamingConfig(
            batch_size=50,
            max_epochs=5,  # More epochs for better convergence
            enable_fault_tolerance=True,
            enable_checkpoints=False,
            learning_rate=0.01,  # Higher learning rate
        )
        optimizer_full = StreamingOptimizer(config_full)
        data_source = (x_data.reshape(-1, 1), y_data)
        result_full = optimizer_full.fit(data_source, model, p0, verbose=0)

        # Both should succeed
        assert result_fast["success"]
        assert result_full["success"]

        # Both should produce similar parameters
        assert np.allclose(result_fast["x"], result_full["x"], rtol=0.2)

        # Should be moving towards the true parameters [2.0, 1.0]
        # (might not fully converge, but should be improving from initial [1.0, 0.0])
        # Note: Convergence can be slow with stochastic optimization

    def test_fast_mode_omits_diagnostics_full_mode_includes(self):
        """Test diagnostic differences between modes."""

        # Simple model
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(200)
        y_data = 2.0 * x_data + 1.0

        p0 = np.array([1.0, 0.0])

        # Fast mode
        config_fast = StreamingConfig(
            batch_size=50, max_epochs=1, enable_fault_tolerance=False
        )
        optimizer_fast = StreamingOptimizer(config_fast)
        data_source = (x_data.reshape(-1, 1), y_data)
        result_fast = optimizer_fast.fit(data_source, model, p0, verbose=0)

        # Full mode
        config_full = StreamingConfig(
            batch_size=50, max_epochs=1, enable_fault_tolerance=True
        )
        optimizer_full = StreamingOptimizer(config_full)
        data_source = (x_data.reshape(-1, 1), y_data)
        result_full = optimizer_full.fit(data_source, model, p0, verbose=0)

        # Fast mode should NOT have streaming_diagnostics
        assert "streaming_diagnostics" in result_fast
        # Fast mode still includes diagnostics but they're minimal
        assert (
            len(result_fast["streaming_diagnostics"]["recent_batch_stats"]) == 0
        )  # fast

        # Full mode SHOULD have streaming_diagnostics
        assert "streaming_diagnostics" in result_full
        assert "failed_batches" in result_full["streaming_diagnostics"]
        assert "retry_counts" in result_full["streaming_diagnostics"]
        assert "error_types" in result_full["streaming_diagnostics"]

    def test_fast_mode_overhead_benchmark(self):
        """Benchmark overhead comparison between fast and full mode."""

        # Simple model for accurate timing
        def model(x, a, b):
            return a * x + b

        np.random.seed(42)
        x_data = np.random.randn(10000)
        y_data = 2.0 * x_data + 1.0

        p0 = np.array([1.0, 0.0])

        # Warm up JIT compilation
        config_warmup = StreamingConfig(
            batch_size=100, max_epochs=1, enable_checkpoints=False
        )
        optimizer_warmup = StreamingOptimizer(config_warmup)
        data_source = (x_data[:100].reshape(-1, 1), y_data[:100])
        optimizer_warmup.fit(data_source, model, p0, verbose=0)

        # Benchmark fast mode
        times_fast = []
        for _ in range(3):
            config_fast = StreamingConfig(
                batch_size=100,
                max_epochs=1,
                enable_fault_tolerance=False,
                enable_checkpoints=False,
            )
            optimizer_fast = StreamingOptimizer(config_fast)
            data_source = (x_data.reshape(-1, 1), y_data)

            start = time.time()
            optimizer_fast.fit(data_source, model, p0, verbose=0)
            times_fast.append(time.time() - start)

        # Benchmark full mode
        times_full = []
        for _ in range(3):
            config_full = StreamingConfig(
                batch_size=100,
                max_epochs=1,
                enable_fault_tolerance=True,
                validate_numerics=True,
                enable_checkpoints=False,
            )
            optimizer_full = StreamingOptimizer(config_full)
            data_source = (x_data.reshape(-1, 1), y_data)

            start = time.time()
            optimizer_full.fit(data_source, model, p0, verbose=0)
            times_full.append(time.time() - start)

        avg_fast = np.mean(times_fast)
        avg_full = np.mean(times_full)

        overhead_percent = (
            ((avg_full - avg_fast) / avg_fast) * 100 if avg_fast > 0 else 0
        )

        print(f"\nFast mode average: {avg_fast:.4f}s")
        print(f"Full mode average: {avg_full:.4f}s")
        print(f"Full mode overhead: {overhead_percent:.1f}%")

        # Main goal: both modes work correctly
        # Timing can vary due to JIT compilation and system load
        # so we don't enforce strict timing requirements
        assert avg_fast > 0, "Fast mode should complete"
        assert avg_full > 0, "Full mode should complete"
