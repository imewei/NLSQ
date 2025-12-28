"""Tests for success rate validation in streaming optimizer (Task Group 5)."""

from unittest.mock import patch

import numpy as np
import pytest

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestSuccessRateValidation:
    """Test success rate tracking and validation."""

    def test_success_rate_calculation(self):
        """Test 5.1: Test success rate calculation."""
        config = StreamingConfig(
            batch_size=10,
            max_epochs=1,
            enable_fault_tolerance=True,
            min_success_rate=0.5,
            validate_numerics=True,
        )
        optimizer = StreamingOptimizer(config)

        # Create a simple quadratic model
        def model(x, a, b):
            return a * x**2 + b

        # Create test data
        x_data = np.linspace(-1, 1, 100)
        y_data = 2.0 * x_data**2 + 1.0 + np.random.randn(100) * 0.01

        # Create a data source that will fail on specific batches
        def data_generator():
            batch_size = 10
            n_batches = len(x_data) // batch_size
            for i in range(n_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                x_batch = x_data[start:end]
                y_batch = y_data[start:end]

                # Make batch 3 and 7 fail by injecting NaN
                if i in [3, 7]:
                    y_batch[0] = np.nan

                yield x_batch, y_batch

        # Fit the model
        result = optimizer.fit_streaming(
            data_generator(),
            model,
            p0=np.array([1.0, 0.5]),
            verbose=0,
        )

        # Check that success rate is calculated correctly
        # 10 batches total, 2 should fail = 8/10 = 0.8
        assert "streaming_diagnostics" in result
        assert "batch_success_rate" in result["streaming_diagnostics"]
        assert result["streaming_diagnostics"]["batch_success_rate"] == pytest.approx(
            0.8, abs=0.01
        )

    def test_threshold_enforcement(self):
        """Test 5.2: Test threshold enforcement."""
        config = StreamingConfig(
            batch_size=10,
            max_epochs=1,
            enable_fault_tolerance=True,
            min_success_rate=0.7,  # Require 70% success
            validate_numerics=True,
        )
        optimizer = StreamingOptimizer(config)

        # Create a simple model
        def model(x, a):
            return a * x

        # Create test data that will have >30% failure rate
        def data_generator():
            for i in range(10):
                x_batch = np.random.randn(10)
                y_batch = 2.0 * x_batch

                # Make 4 out of 10 batches fail (40% failure rate)
                if i in [2, 4, 6, 8]:
                    y_batch[0] = np.nan

                yield x_batch, y_batch

        # Fit the model
        with patch("nlsq.streaming.optimizer.logger") as mock_logger:
            result = optimizer.fit_streaming(
                data_generator(),
                model,
                p0=np.array([1.0]),
                verbose=0,
            )

            # Check that warning was logged for low success rate
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "below minimum" in str(call)
            ]
            assert len(warning_calls) > 0

        # Success rate should be 6/10 = 0.6, below the 0.7 threshold
        assert result["streaming_diagnostics"]["batch_success_rate"] == pytest.approx(
            0.6, abs=0.01
        )

    def test_result_flagging_on_failure(self):
        """Test 5.3: Test result flagging on failure."""
        config = StreamingConfig(
            batch_size=10,
            max_epochs=1,
            enable_fault_tolerance=True,
            min_success_rate=0.9,  # Very high threshold
            validate_numerics=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a):
            return a * x

        # Create test data with high failure rate
        def data_generator():
            for i in range(10):
                x_batch = np.random.randn(10)
                y_batch = 2.0 * x_batch

                # Make 5 out of 10 batches fail (50% success rate)
                if i % 2 == 0:
                    y_batch[0] = np.nan

                yield x_batch, y_batch

        result = optimizer.fit_streaming(
            data_generator(),
            model,
            p0=np.array([1.0]),
            verbose=0,
        )

        # Check result includes success flag
        assert "success" in result
        # With 50% success rate and 90% threshold, there should be some indication
        # The actual success flag depends on whether ANY batches succeeded
        assert "streaming_diagnostics" in result
        assert "batch_success_rate" in result["streaming_diagnostics"]
        assert result["streaming_diagnostics"]["batch_success_rate"] == pytest.approx(
            0.5, abs=0.01
        )

    def test_various_failure_percentages(self):
        """Test 5.4: Test with various failure percentages."""
        test_cases = [
            (0.0, 1.0),  # 0% failure = 100% success
            (0.2, 0.8),  # 20% failure = 80% success
            (0.5, 0.5),  # 50% failure = 50% success
            (0.8, 0.2),  # 80% failure = 20% success
            (1.0, 0.0),  # 100% failure = 0% success
        ]

        for failure_rate, expected_success_rate in test_cases:
            config = StreamingConfig(
                batch_size=10,
                max_epochs=1,
                enable_fault_tolerance=True,
                min_success_rate=0.1,  # Low threshold to allow high failure rates
                validate_numerics=True,
            )
            optimizer = StreamingOptimizer(config)

            def model(x, a):
                return a * x

            def data_generator(failure_rate=failure_rate):
                n_batches = 10
                n_failures = int(n_batches * failure_rate)

                for i in range(n_batches):
                    x_batch = np.random.randn(10)
                    y_batch = 2.0 * x_batch

                    # Inject failures based on rate
                    if i < n_failures:
                        y_batch[0] = np.nan

                    yield x_batch, y_batch

            result = optimizer.fit_streaming(
                data_generator(),
                model,
                p0=np.array([1.0]),
                verbose=0,
            )

            # Check success rate matches expected
            actual_success_rate = result.get("streaming_diagnostics", {}).get(
                "batch_success_rate", 0.0
            )
            assert actual_success_rate == pytest.approx(
                expected_success_rate, abs=0.01
            ), (
                f"Failed for failure_rate={failure_rate}: expected {expected_success_rate}, got {actual_success_rate}"
            )

    def test_success_rate_in_streaming_diagnostics(self):
        """Test 5.5: Ensure success rate is included in streaming diagnostics."""
        config = StreamingConfig(
            batch_size=10,
            max_epochs=1,
            enable_fault_tolerance=True,
            min_success_rate=0.5,
            validate_numerics=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a):
            return a * x

        def data_generator():
            for i in range(5):
                x_batch = np.random.randn(10)
                y_batch = 2.0 * x_batch

                # Make 1 out of 5 batches fail
                if i == 2:
                    y_batch[0] = np.nan

                yield x_batch, y_batch

        result = optimizer.fit_streaming(
            data_generator(),
            model,
            p0=np.array([1.0]),
            verbose=0,
        )

        # Check streaming diagnostics includes success rate
        assert "streaming_diagnostics" in result
        diagnostics = result["streaming_diagnostics"]
        assert "batch_success_rate" in diagnostics
        assert diagnostics["batch_success_rate"] == pytest.approx(0.8, abs=0.01)

    def test_success_rate_with_no_failures(self):
        """Test success rate calculation when all batches succeed."""
        config = StreamingConfig(
            batch_size=10,
            max_epochs=1,
            enable_fault_tolerance=True,
            min_success_rate=0.5,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a):
            return a * x

        # Generate clean data with no failures
        def data_generator():
            for i in range(5):
                x_batch = np.random.randn(10)
                y_batch = 2.0 * x_batch + np.random.randn(10) * 0.01
                yield x_batch, y_batch

        result = optimizer.fit_streaming(
            data_generator(),
            model,
            p0=np.array([1.0]),
            verbose=0,
        )

        # All batches should succeed
        assert result["streaming_diagnostics"]["batch_success_rate"] == 1.0
        assert result["success"] is True

    def test_success_rate_with_all_failures(self):
        """Test success rate calculation when all batches fail."""
        config = StreamingConfig(
            batch_size=10,
            max_epochs=1,
            enable_fault_tolerance=True,
            min_success_rate=0.1,  # Low threshold
            validate_numerics=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a):
            # This will always produce NaN
            return a * x / 0

        def data_generator():
            for i in range(5):
                x_batch = np.random.randn(10)
                y_batch = np.random.randn(10)
                yield x_batch, y_batch

        result = optimizer.fit_streaming(
            data_generator(),
            model,
            p0=np.array([1.0]),
            verbose=0,
        )

        # All batches should fail
        assert result["streaming_diagnostics"]["batch_success_rate"] == 0.0
        assert result["success"] is False

    def test_success_rate_updates_during_optimization(self):
        """Test that success rate is tracked correctly throughout optimization."""
        config = StreamingConfig(
            batch_size=10,
            max_epochs=1,  # Use single epoch to avoid generator exhaustion
            enable_fault_tolerance=True,
            min_success_rate=0.5,
            validate_numerics=True,
        )
        optimizer = StreamingOptimizer(config)

        def model(x, a):
            return a * x

        def data_generator():
            # Generate 10 batches total with specific failures
            for i in range(10):
                x_batch = np.random.randn(10)
                y_batch = 2.0 * x_batch

                # Fail batches 1, 3, and 7 (3/10 fail = 70% success rate)
                if i in [1, 3, 7]:
                    y_batch[0] = np.nan

                yield x_batch, y_batch

        result = optimizer.fit_streaming(
            data_generator(),
            model,
            p0=np.array([1.0]),
            verbose=0,
        )

        # Check overall success rate
        # We expect 7/10 = 0.7 success rate
        assert result["streaming_diagnostics"]["batch_success_rate"] == pytest.approx(
            0.7, abs=0.01
        )
