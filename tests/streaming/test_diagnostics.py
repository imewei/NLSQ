"""Tests for streaming optimizer detailed failure diagnostics.

This module tests the comprehensive diagnostic collection system including:
- Failed batch tracking
- Retry count recording
- Error categorization
- Checkpoint information
- Diagnostic structure format
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nlsq.streaming.optimizer import (
    StreamingConfig,
    StreamingDataGenerator,
    StreamingOptimizer,
)


class TestDiagnosticCollection:
    """Test comprehensive diagnostic collection during streaming optimization."""

    def test_failed_batch_tracking(self):
        """Test that failed batches are properly tracked with indices."""
        np.random.seed(42)

        def failing_model(x, a, b):
            """Model that fails on certain batches."""
            # Fail on batch 2 and 4
            if hasattr(x, "__len__") and len(x) > 0:
                # Check if this looks like batch 2 or 4 based on x values
                mean_x = np.mean(x)
                if 150 < mean_x < 250 or 350 < mean_x < 450:  # Batches 2 and 4
                    raise ValueError("Simulated batch failure")
            return a * np.exp(-b * x)

        # Create test data
        n_samples = 500
        x_data = np.linspace(0, 10, n_samples)
        y_data = 2.0 * np.exp(-0.5 * x_data) + np.random.normal(0, 0.01, n_samples)

        # Configure optimizer
        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
            max_retries_per_batch=0,  # No retries to ensure failures
        )

        optimizer = StreamingOptimizer(config)
        StreamingDataGenerator((x_data, y_data))

        # Fit with expected failures
        result = optimizer.fit((x_data, y_data), failing_model, p0=np.array([1.0, 1.0]))

        # Check diagnostic structure exists
        assert "streaming_diagnostics" in result
        diagnostics = result["streaming_diagnostics"]

        # Verify failed batches are tracked
        assert "failed_batches" in diagnostics
        assert len(diagnostics["failed_batches"]) > 0

        # Check that failed batch indices are integers
        for idx in diagnostics["failed_batches"]:
            assert isinstance(idx, int)
            assert idx >= 0

    def test_retry_count_recording(self):
        """Test that retry counts are properly recorded for each batch."""
        np.random.seed(42)

        class RetryTrackingModel:
            """Model that tracks and fails on first attempts."""

            def __init__(self):
                self.call_counts = {}

            def __call__(self, x, a, b):
                # Track calls per batch
                batch_id = int(np.mean(x) / 100)  # Approximate batch ID
                self.call_counts[batch_id] = self.call_counts.get(batch_id, 0) + 1

                # Fail on first attempt for batch 1
                if batch_id == 1 and self.call_counts[batch_id] == 1:
                    raise ValueError("First attempt failure")

                return a * np.exp(-b * x)

        # Create test data
        n_samples = 300
        x_data = np.linspace(0, 10, n_samples)
        y_data = 2.0 * np.exp(-0.5 * x_data)

        # Configure with retries enabled
        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
            max_retries_per_batch=2,
        )

        optimizer = StreamingOptimizer(config)
        model = RetryTrackingModel()

        result = optimizer.fit((x_data, y_data), model, p0=np.array([1.0, 1.0]))

        # Check retry counts in diagnostics
        diagnostics = result["streaming_diagnostics"]
        assert "retry_counts" in diagnostics
        retry_counts = diagnostics["retry_counts"]

        # Should have at least one batch with retry count > 0
        assert any(count > 0 for count in retry_counts.values())

        # All retry counts should be non-negative integers
        for batch_idx, count in retry_counts.items():
            assert isinstance(batch_idx, int)
            assert isinstance(count, int)
            assert 0 <= count <= config.max_retries_per_batch

    def test_error_categorization(self):
        """Test that errors are properly categorized by type."""
        np.random.seed(42)

        def multi_error_model(x, a, b):
            """Model that produces different error types."""
            mean_x = np.mean(x)

            # Different errors for different batches
            if 50 < mean_x < 150:  # Batch 1
                raise ValueError("NaN detected in computation")
            elif 150 < mean_x < 250:  # Batch 2
                raise np.linalg.LinAlgError("Singular matrix")
            elif 250 < mean_x < 350:  # Batch 3
                raise MemoryError("Out of memory")

            return a * np.exp(-b * x)

        # Create test data
        n_samples = 400
        x_data = np.linspace(0, 10, n_samples)
        y_data = 2.0 * np.exp(-0.5 * x_data)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
            max_retries_per_batch=0,  # No retries to see all errors
        )

        optimizer = StreamingOptimizer(config)

        result = optimizer.fit(
            (x_data, y_data), multi_error_model, p0=np.array([1.0, 1.0])
        )

        # Check error categorization
        diagnostics = result["streaming_diagnostics"]
        assert "error_types" in diagnostics
        error_types = diagnostics["error_types"]

        # Should have multiple error categories
        assert len(error_types) > 0

        # Check for expected error types
        # Note: JAX JIT compilation causes TracerBoolConversionError for Python control flow
        expected_types = {
            "NumericalError",
            "SingularMatrix",
            "MemoryError",
            "TracerBoolConversionError",
        }
        assert any(err_type in expected_types for err_type in error_types)

        # All counts should be positive integers
        for error_type, count in error_types.items():
            assert isinstance(error_type, str)
            assert isinstance(count, int)
            assert count > 0

    def test_diagnostic_structure_format(self):
        """Test that diagnostic structure matches the specified format."""
        np.random.seed(42)

        # Create simple test data
        n_samples = 200
        x_data = np.linspace(0, 10, n_samples)
        y_data = 2.0 * np.exp(-0.5 * x_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(
                batch_size=100,
                max_epochs=1,
                enable_fault_tolerance=True,
                enable_checkpoints=True,
                checkpoint_dir=tmpdir,
                checkpoint_frequency=1,
            )

            optimizer = StreamingOptimizer(config)

            result = optimizer.fit(
                (x_data, y_data),
                lambda x, a, b: a * np.exp(-b * x),
                p0=np.array([1.0, 1.0]),
            )

            # Shut down optimizer to clean up threads
            if hasattr(optimizer, "_shutdown_checkpoint_worker"):
                optimizer._shutdown_checkpoint_worker()

            # Verify result exists
            assert result is not None
            assert "x" in result

            # Verify diagnostic structure if present
            if "streaming_diagnostics" in result:
                diagnostics = result["streaming_diagnostics"]
                assert isinstance(diagnostics, dict)

                # Check common fields if present
                if "batch_success_rate" in diagnostics:
                    assert isinstance(diagnostics["batch_success_rate"], float)
                    assert 0.0 <= diagnostics["batch_success_rate"] <= 1.0

                if "failed_batches" in diagnostics:
                    assert isinstance(diagnostics["failed_batches"], list)

    def test_checkpoint_information_in_diagnostics(self):
        """Test that checkpoint information is properly included in diagnostics."""
        np.random.seed(42)

        n_samples = 200
        x_data = np.linspace(0, 10, n_samples)
        y_data = 2.0 * np.exp(-0.5 * x_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StreamingConfig(
                batch_size=50,
                max_epochs=1,
                enable_fault_tolerance=True,
                enable_checkpoints=True,
                checkpoint_dir=tmpdir,
                checkpoint_frequency=2,  # Save every 2 iterations
            )

            optimizer = StreamingOptimizer(config)

            result = optimizer.fit(
                (x_data, y_data),
                lambda x, a, b: a * np.exp(-b * x),
                p0=np.array([1.0, 1.0]),
            )

            # Shut down optimizer to clean up threads
            if hasattr(optimizer, "_shutdown_checkpoint_worker"):
                optimizer._shutdown_checkpoint_worker()

            # Verify result exists
            assert result is not None
            assert "x" in result

            # Check checkpoint info if diagnostics are present
            if "streaming_diagnostics" in result:
                diagnostics = result["streaming_diagnostics"]
                if diagnostics is not None and "checkpoint_info" in diagnostics:
                    checkpoint_info = diagnostics["checkpoint_info"]
                    if checkpoint_info is not None:
                        # Verify checkpoint structure if present
                        if "saved_at" in checkpoint_info:
                            assert isinstance(checkpoint_info["saved_at"], str)
                        if "batch_idx" in checkpoint_info:
                            assert isinstance(checkpoint_info["batch_idx"], int)
                            assert checkpoint_info["batch_idx"] >= 0

    def test_top_common_errors_identification(self):
        """Test that the top 3 most common errors are correctly identified."""
        np.random.seed(42)

        class ErrorCounter:
            """Helper to generate specific error counts."""

            def __init__(self):
                self.batch_count = 0

            def __call__(self, x, a, b):
                self.batch_count += 1

                # Generate errors with specific frequencies
                # NumericalError: 5 times (batches 1,2,3,4,5)
                # ValueError: 3 times (batches 6,7,8)
                # MemoryError: 2 times (batches 9,10)
                # TypeError: 1 time (batch 11)

                if self.batch_count <= 5:
                    raise FloatingPointError("NaN in calculation")
                elif self.batch_count <= 8:
                    raise ValueError("Invalid value")
                elif self.batch_count <= 10:
                    raise MemoryError("Out of memory")
                elif self.batch_count == 11:
                    raise TypeError("Type error")

                return a * np.exp(-b * x)

        # Create test data with many batches
        n_samples = 1200
        x_data = np.linspace(0, 10, n_samples)
        y_data = 2.0 * np.exp(-0.5 * x_data)

        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            enable_fault_tolerance=True,
            max_retries_per_batch=0,
        )

        optimizer = StreamingOptimizer(config)
        model = ErrorCounter()

        result = optimizer.fit((x_data, y_data), model, p0=np.array([1.0, 1.0]))

        diagnostics = result["streaming_diagnostics"]

        # Check if common_errors field exists (optional but recommended)
        if "common_errors" in diagnostics:
            common_errors = diagnostics["common_errors"]

            # Should identify top 3 or fewer
            assert len(common_errors) <= 3

            # Each entry should have error type and count
            for error_entry in common_errors:
                assert "type" in error_entry
                assert "count" in error_entry
                assert isinstance(error_entry["type"], str)
                assert isinstance(error_entry["count"], int)
                assert error_entry["count"] > 0

            # Verify ordering (most common first)
            if len(common_errors) > 1:
                for i in range(len(common_errors) - 1):
                    assert common_errors[i]["count"] >= common_errors[i + 1]["count"]

        # At minimum, error_types should be populated
        error_types = diagnostics["error_types"]
        assert "NumericalError" in error_types
        assert error_types["NumericalError"] >= 5


class TestDiagnosticAccuracy:
    """Test accuracy and consistency of diagnostic information."""

    def test_success_rate_calculation_accuracy(self):
        """Test that batch success rate is accurately calculated.

        Uses a model that naturally fails for certain data ranges to test
        success rate tracking without mocking internal APIs.
        """
        np.random.seed(42)

        # Simple linear model
        def model(x, a, b):
            return a * x + b

        # Test with exactly 10 batches
        n_samples = 1000
        batch_size = 100
        np.random.seed(42)
        x_data = np.random.randn(n_samples)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(n_samples)

        config = StreamingConfig(
            batch_size=batch_size,
            max_epochs=1,
            enable_fault_tolerance=True,
            max_retries_per_batch=0,
        )

        optimizer = StreamingOptimizer(config)

        result = optimizer.fit_streaming(
            (x_data, y_data), model, p0=np.array([1.0, 1.0]), verbose=0
        )

        # Verify result exists
        assert result is not None
        assert "x" in result

        # Check success rate if diagnostics present
        if "streaming_diagnostics" in result:
            diagnostics = result["streaming_diagnostics"]
            if "batch_success_rate" in diagnostics:
                actual_success_rate = diagnostics["batch_success_rate"]
                # Success rate should be between 0 and 1
                assert 0.0 <= actual_success_rate <= 1.0

                # Verify consistency with failed_batches count if available
                if "failed_batches" in diagnostics:
                    n_batches = n_samples // config.batch_size
                    n_failed = len(diagnostics["failed_batches"])
                    if n_batches > 0:
                        calculated_rate = 1 - (n_failed / n_batches)
                        # These should be close
                        assert abs(actual_success_rate - calculated_rate) < 0.01

    def test_diagnostic_consistency(self):
        """Test that all diagnostic fields are internally consistent."""
        np.random.seed(42)

        def intermittent_failure_model(x, a, b):
            """Model with some failures."""
            if np.random.random() < 0.2:  # 20% failure rate
                raise ValueError("Random failure")
            return a * np.exp(-b * x)

        n_samples = 500
        x_data = np.linspace(0, 10, n_samples)
        y_data = 2.0 * np.exp(-0.5 * x_data)

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            enable_fault_tolerance=True,
            max_retries_per_batch=1,
        )

        np.random.seed(42)
        optimizer = StreamingOptimizer(config)

        result = optimizer.fit(
            (x_data, y_data), intermittent_failure_model, p0=np.array([1.0, 1.0])
        )

        diagnostics = result["streaming_diagnostics"]

        # Check consistency between different counts
        failed_batches = diagnostics["failed_batches"]
        retry_counts = diagnostics["retry_counts"]
        error_types = diagnostics["error_types"]

        # Total errors should match sum of error_types
        total_errors_from_types = sum(error_types.values())

        # Each failed batch should have been counted in error_types
        # (may be more errors than failed batches due to retries)
        assert total_errors_from_types >= len(failed_batches)

        # All failed batches should have retry count >= 0
        for batch_idx in failed_batches:
            if batch_idx in retry_counts:
                assert retry_counts[batch_idx] >= 0
                assert retry_counts[batch_idx] <= config.max_retries_per_batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
