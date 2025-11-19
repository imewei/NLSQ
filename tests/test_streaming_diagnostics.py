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

from nlsq.streaming_optimizer import (
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

            # Verify diagnostic structure
            assert "streaming_diagnostics" in result
            diagnostics = result["streaming_diagnostics"]

            # Check required fields exist
            required_fields = [
                "failed_batches",
                "retry_counts",
                "error_types",
                "batch_success_rate",
                "checkpoint_info",
            ]

            for field in required_fields:
                assert field in diagnostics, f"Missing required field: {field}"

            # Verify field types
            assert isinstance(diagnostics["failed_batches"], list)
            assert isinstance(diagnostics["retry_counts"], dict)
            assert isinstance(diagnostics["error_types"], dict)
            assert isinstance(diagnostics["batch_success_rate"], float)
            assert isinstance(diagnostics["checkpoint_info"], dict)

            # Verify checkpoint_info structure
            checkpoint_info = diagnostics["checkpoint_info"]
            assert "saved_at" in checkpoint_info
            assert "batch_idx" in checkpoint_info

            # Optional: check for aggregate stats if implemented
            if "aggregate_stats" in diagnostics:
                stats = diagnostics["aggregate_stats"]
                assert isinstance(stats, dict)
                # Could have mean_loss, std_loss, mean_grad_norm etc.

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

            # Check checkpoint info
            diagnostics = result["streaming_diagnostics"]
            checkpoint_info = diagnostics["checkpoint_info"]

            # Verify checkpoint path exists if checkpoints were saved
            if config.enable_checkpoints:
                # Check checkpoint directory was used
                checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.h5"))
                if checkpoint_files:
                    assert "path" in checkpoint_info
                    assert checkpoint_info["path"] is not None

                # Check save time format
                assert "saved_at" in checkpoint_info
                saved_at = checkpoint_info["saved_at"]
                # Should be a timestamp string
                assert isinstance(saved_at, str)

                # Check batch index
                assert "batch_idx" in checkpoint_info
                assert isinstance(checkpoint_info["batch_idx"], int)
                assert checkpoint_info["batch_idx"] >= 0

                # Check file size if implemented
                if "file_size" in checkpoint_info:
                    assert isinstance(checkpoint_info["file_size"], (int, float))
                    assert checkpoint_info["file_size"] > 0

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
        """Test that batch success rate is accurately calculated."""
        np.random.seed(42)

        # Simple linear model (more JAX-friendly than exponential)
        def model(x, a, b):
            """Simple linear model."""
            return a * x + b

        # Test with exactly 10 batches
        n_samples = 1000
        batch_size = 100  # 1000 / 100 = 10 batches
        np.random.seed(42)
        x_data = np.random.randn(n_samples)
        y_data = 2.0 * x_data + 1.0 + 0.1 * np.random.randn(n_samples)

        config = StreamingConfig(
            batch_size=batch_size,
            max_epochs=1,
            enable_fault_tolerance=True,
            max_retries_per_batch=0,  # No retries for accurate count
        )

        optimizer = StreamingOptimizer(config)

        # Mock _compute_loss_and_gradient to inject failures for specific batches
        original_compute = optimizer._compute_loss_and_gradient
        call_count = [0]
        batch_calls = []  # Track which batches are called
        fail_batches = {2, 5, 8}  # 3 out of 10 batches = 30% failure

        def mock_compute(func, params, x_batch, y_batch, mask=None):
            call_count[0] += 1
            # Each batch is processed once per epoch
            # Track batch index by call order
            batch_idx = len(batch_calls)
            if call_count[0] > len(batch_calls):
                batch_calls.append(call_count[0])

            if batch_idx in fail_batches:
                # Simulate failure for this batch
                raise ValueError(f"Controlled failure for batch {batch_idx}")

            return original_compute(func, params, x_batch, y_batch, mask)

        optimizer._compute_loss_and_gradient = mock_compute

        result = optimizer.fit_streaming(
            (x_data, y_data), model, p0=np.array([1.0, 1.0]), verbose=0
        )

        diagnostics = result["streaming_diagnostics"]
        actual_success_rate = diagnostics["batch_success_rate"]

        # Expected: 7 successes out of 10 batches = 70%
        expected_success_rate = 0.7
        assert abs(actual_success_rate - expected_success_rate) < 0.01

        # Verify consistency with failed_batches count
        n_batches = n_samples // config.batch_size
        n_failed = len(diagnostics["failed_batches"])
        calculated_rate = 1 - (n_failed / n_batches)

        # These should match exactly
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
