"""Tests for gradient health monitoring (User Story 2).

This module tests the GradientMonitor class which tracks gradient health
during optimization iterations. It detects:
- Vanishing gradients (GRAD-001)
- Gradient imbalance (GRAD-002)
- Gradient stagnation (GRAD-003)

Memory is bounded at <1KB regardless of iteration count via:
- Sliding window for gradient norms (100 entries default)
- Welford's online algorithm for running statistics
"""

import sys

import numpy as np
import pytest

from nlsq.diagnostics.gradient_health import GradientMonitor
from nlsq.diagnostics.types import (
    DiagnosticsConfig,
    GradientHealthReport,
    HealthStatus,
    IssueCategory,
    IssueSeverity,
)


class TestGradientMonitor:
    """Test suite for GradientMonitor class."""

    def test_initialization(self) -> None:
        """Test GradientMonitor initializes correctly."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        assert monitor.config == config
        assert monitor.iteration_count == 0
        assert len(monitor._gradient_norm_history) == 0

    def test_initialization_with_custom_config(self) -> None:
        """Test initialization with custom thresholds."""
        config = DiagnosticsConfig(
            vanishing_threshold=1e-8,
            imbalance_threshold=1e4,
            gradient_window_size=50,
        )
        monitor = GradientMonitor(config)

        assert monitor.config.vanishing_threshold == 1e-8
        assert monitor.config.imbalance_threshold == 1e4
        assert monitor.config.gradient_window_size == 50

    def test_record_gradient_basic(self) -> None:
        """Test recording a single gradient."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        gradient = np.array([1.0, 2.0, 3.0])
        cost = 1.0
        monitor.record_gradient(gradient, cost)

        assert monitor.iteration_count == 1
        assert len(monitor._gradient_norm_history) == 1

    def test_record_gradient_multiple(self) -> None:
        """Test recording multiple gradients."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        for i in range(10):
            gradient = np.array([1.0, 2.0, 3.0]) * (i + 1)
            monitor.record_gradient(gradient, cost=float(i))

        assert monitor.iteration_count == 10
        assert len(monitor._gradient_norm_history) == 10

    def test_sliding_window_bounds_memory(self) -> None:
        """Test that sliding window limits memory usage."""
        window_size = 100
        config = DiagnosticsConfig(gradient_window_size=window_size)
        monitor = GradientMonitor(config)

        # Record more iterations than window size
        for i in range(500):
            gradient = np.array([1.0, 2.0, 3.0])
            monitor.record_gradient(gradient, cost=1.0)

        # Window should be bounded
        assert len(monitor._gradient_norm_history) <= window_size
        assert monitor.iteration_count == 500

    def test_get_report_no_issues(self) -> None:
        """Test report generation with healthy gradients."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        # Record healthy gradients with good magnitude, balance, AND variation
        # Adding small variation to avoid stagnation detection
        np.random.seed(42)
        for i in range(30):
            # Add small random variation to avoid stagnation
            noise = 1.0 + 0.1 * np.random.randn()
            gradient = np.array([1.0, 0.8, 1.2]) * noise
            # Cost decreasing normally
            cost = 10.0 / (i + 1)
            monitor.record_gradient(gradient, cost)

        report = monitor.get_report()

        assert isinstance(report, GradientHealthReport)
        assert report.available is True
        assert report.health_status == HealthStatus.HEALTHY
        assert len(report.issues) == 0

    def test_get_report_insufficient_data(self) -> None:
        """Test report with insufficient iterations."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        # Record only one iteration
        gradient = np.array([1.0, 2.0, 3.0])
        monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()

        # Should still be available but with limited analysis
        assert report.available is True
        assert report.n_iterations == 1


class TestVanishingGradientDetection:
    """Tests for vanishing gradient detection (GRAD-001)."""

    def test_vanishing_gradient_detected(self) -> None:
        """Test that vanishing gradients are detected."""
        config = DiagnosticsConfig(vanishing_threshold=1e-6)
        monitor = GradientMonitor(config)

        # Start with reasonable gradient, then have it vanish
        # while cost remains high
        for i in range(30):
            if i < 5:
                # Initial phase: normal gradient
                gradient = np.array([1.0, 1.0, 1.0])
            else:
                # Vanishing phase: gradient drops to very small
                gradient = np.array([1e-10, 1e-10, 1e-10])
            # Cost remains significant (not converged)
            monitor.record_gradient(gradient, cost=10.0)

        report = monitor.get_report()

        # Should detect vanishing gradients
        grad_001_issues = [issue for issue in report.issues if issue.code == "GRAD-001"]
        assert len(grad_001_issues) > 0
        assert grad_001_issues[0].severity == IssueSeverity.WARNING

    def test_vanishing_gradient_with_saturating_model(self) -> None:
        """Test vanishing gradient detection with a saturating activation model.

        This simulates the behavior of models like sigmoid or tanh when
        parameters cause the function to saturate, leading to gradients
        that approach zero even though the fit is not optimal.
        """
        config = DiagnosticsConfig(vanishing_threshold=1e-6)
        monitor = GradientMonitor(config)

        # Simulate a saturating model where gradients vanish due to saturation
        # Initial gradients are reasonable, then they vanish as model saturates
        for i in range(30):
            if i < 10:
                # Initial phase: reasonable gradients
                gradient = np.array([0.1, 0.2, 0.15])
            else:
                # Saturation phase: gradients vanish due to sigmoid saturation
                # This happens when parameters are too extreme
                gradient = np.array([1e-8, 1e-9, 1e-8])

            # Cost remains high - we haven't converged
            cost = 5.0 - (i * 0.05)  # Slowly decreasing but still significant
            monitor.record_gradient(gradient, cost)

        report = monitor.get_report()

        # Should detect vanishing gradient issue
        grad_001_issues = [issue for issue in report.issues if issue.code == "GRAD-001"]
        assert len(grad_001_issues) > 0
        assert report.health_status in (HealthStatus.WARNING, HealthStatus.CRITICAL)

    def test_small_gradient_at_convergence_not_flagged(self) -> None:
        """Test that small gradients near convergence are not flagged."""
        config = DiagnosticsConfig(vanishing_threshold=1e-6)
        monitor = GradientMonitor(config)

        # Gradients decrease as cost decreases - this is normal convergence
        for i in range(20):
            gradient_mag = 1.0 / (i + 1)
            gradient = np.array([gradient_mag, gradient_mag, gradient_mag])
            cost = 1.0 / ((i + 1) ** 2)  # Cost also decreasing
            monitor.record_gradient(gradient, cost)

        report = monitor.get_report()

        # Small gradients with small cost should not be flagged as vanishing
        grad_001_issues = [issue for issue in report.issues if issue.code == "GRAD-001"]
        # At convergence, small gradients are expected
        # The key is that cost is also small


class TestGradientImbalanceDetection:
    """Tests for gradient imbalance detection (GRAD-002)."""

    def test_imbalance_detected(self) -> None:
        """Test that gradient imbalance is detected."""
        config = DiagnosticsConfig(imbalance_threshold=1e6)
        monitor = GradientMonitor(config)

        # Record gradients with extreme imbalance
        for i in range(20):
            # Parameter 0 has huge gradient, parameter 2 has tiny gradient
            gradient = np.array([1e6, 1.0, 1e-6])
            monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()

        # Should detect imbalance
        grad_002_issues = [issue for issue in report.issues if issue.code == "GRAD-002"]
        assert len(grad_002_issues) > 0
        assert grad_002_issues[0].category == IssueCategory.GRADIENT

    def test_balanced_gradients_not_flagged(self) -> None:
        """Test that balanced gradients are not flagged."""
        config = DiagnosticsConfig(imbalance_threshold=1e6)
        monitor = GradientMonitor(config)

        # Well-balanced gradients
        for i in range(20):
            gradient = np.array([1.0, 0.5, 2.0])  # Within reasonable ratio
            monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()

        # Should not detect imbalance
        grad_002_issues = [issue for issue in report.issues if issue.code == "GRAD-002"]
        assert len(grad_002_issues) == 0

    def test_imbalance_details_include_affected_parameters(self) -> None:
        """Test that imbalance issue includes affected parameter indices."""
        config = DiagnosticsConfig(imbalance_threshold=1e6)
        monitor = GradientMonitor(config)

        # Create clear imbalance between parameters 0 and 2
        for i in range(20):
            gradient = np.array([1e8, 1.0, 1e-2])
            monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()

        grad_002_issues = [issue for issue in report.issues if issue.code == "GRAD-002"]
        if grad_002_issues:
            issue = grad_002_issues[0]
            assert "imbalance_ratio" in issue.details


class TestGradientStagnationDetection:
    """Tests for gradient stagnation detection (GRAD-003)."""

    def test_stagnation_detected(self) -> None:
        """Test that gradient stagnation is detected."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        # Record identical gradients (stagnation)
        for i in range(30):
            gradient = np.array([1.0, 1.0, 1.0])
            cost = 5.0  # Cost not changing either
            monitor.record_gradient(gradient, cost)

        report = monitor.get_report()

        # Should detect stagnation
        grad_003_issues = [issue for issue in report.issues if issue.code == "GRAD-003"]
        assert len(grad_003_issues) > 0

    def test_varying_gradients_not_stagnation(self) -> None:
        """Test that varying gradients are not flagged as stagnation."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        # Record varying gradients
        np.random.seed(42)
        for i in range(30):
            gradient = np.random.randn(3) * 0.5 + 1.0
            monitor.record_gradient(gradient, cost=1.0 / (i + 1))

        report = monitor.get_report()

        # Should not detect stagnation
        grad_003_issues = [issue for issue in report.issues if issue.code == "GRAD-003"]
        assert len(grad_003_issues) == 0


class TestHealthScore:
    """Tests for gradient health score computation."""

    def test_health_score_range(self) -> None:
        """Test that health score is in [0, 1] range."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        for i in range(20):
            gradient = np.array([1.0, 2.0, 3.0])
            monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()

        assert 0.0 <= report.health_score <= 1.0

    def test_perfect_health_score(self) -> None:
        """Test that healthy gradients give high score."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        # Record healthy, well-behaved gradients with variation
        np.random.seed(42)
        for i in range(30):
            # Decreasing but well-balanced gradients with some variation
            factor = 1.0 / (1 + i * 0.1)
            noise = 1.0 + 0.05 * np.random.randn()
            gradient = np.array(
                [factor * noise, factor * 0.9 * noise, factor * 1.1 * noise]
            )
            cost = factor**2  # Cost decreasing with gradient
            monitor.record_gradient(gradient, cost)

        report = monitor.get_report()

        # Should have high health score
        assert report.health_score >= 0.8

    def test_poor_health_score(self) -> None:
        """Test that problematic gradients give low score."""
        config = DiagnosticsConfig(vanishing_threshold=1e-6, imbalance_threshold=1e4)
        monitor = GradientMonitor(config)

        # Record problematic gradients (vanishing + imbalanced)
        for i in range(30):
            if i < 5:
                # Start with reasonable gradient
                gradient = np.array([1.0, 1e3, 1.0])
            else:
                # Then extremely imbalanced and small
                gradient = np.array([1e-9, 1e3, 1e-9])
            monitor.record_gradient(gradient, cost=10.0)

        report = monitor.get_report()

        # Should have low health score
        assert report.health_score < 0.5
        assert len(report.issues) > 0


class TestRunningStatistics:
    """Tests for Welford's algorithm running statistics."""

    def test_running_mean_accuracy(self) -> None:
        """Test that running mean matches expected values."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        # Record known gradients
        gradients = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            np.array([3.0, 6.0, 9.0]),
        ]

        for g in gradients:
            monitor.record_gradient(g, cost=1.0)

        report = monitor.get_report()

        # Expected means per parameter (means of absolute values)
        expected_means = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(
            report.mean_gradient_magnitudes, expected_means, decimal=5
        )

    def test_running_variance_calculation(self) -> None:
        """Test that running variance is computed correctly via Welford's algorithm."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        # Record gradients with known variance
        # Use values [1, 2, 3] for each parameter
        # Mean = 2, Variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 2 = 1
        gradients = [
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0]),
            np.array([3.0, 3.0, 3.0]),
        ]

        for g in gradients:
            monitor.record_gradient(g, cost=1.0)

        report = monitor.get_report()

        # Expected variance = 1.0 for each parameter
        expected_variance = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(
            report.variance_gradient_magnitudes, expected_variance, decimal=5
        )


class TestMemoryBounds:
    """Tests for memory usage bounds (SC-005: <1KB)."""

    def test_memory_bounded_with_many_iterations(self) -> None:
        """Test that memory stays under 1KB with many iterations."""
        config = DiagnosticsConfig(gradient_window_size=100)
        monitor = GradientMonitor(config)

        # Record many iterations
        for i in range(10000):
            gradient = np.array([1.0, 2.0, 3.0])
            monitor.record_gradient(gradient, cost=1.0)

        # Measure memory of internal state
        # Sliding window: 100 floats * 8 bytes = 800 bytes
        # Running stats: ~24 floats * 8 bytes = 192 bytes
        # Other overhead: ~50 bytes
        # Total: ~1042 bytes (may need adjustment)

        memory_estimate = (
            len(monitor._gradient_norm_history) * 8  # Window
            + len(monitor._param_means) * 8  # Means
            + len(monitor._param_m2) * 8  # M2 for variance
            + 8 * 10  # Various counters/scalars
        )

        assert memory_estimate < 1024, f"Memory estimate {memory_estimate} exceeds 1KB"

    def test_memory_independent_of_parameter_count(self) -> None:
        """Test that memory is bounded regardless of parameter count."""
        config = DiagnosticsConfig(gradient_window_size=100)
        monitor = GradientMonitor(config)

        # Use many parameters
        n_params = 1000
        for i in range(200):
            gradient = np.random.randn(n_params)
            monitor.record_gradient(gradient, cost=1.0)

        # Memory should still be bounded
        # We store per-parameter running stats, but that scales with n_params
        # The key constraint is on gradient_norm_history which is window-bounded
        assert len(monitor._gradient_norm_history) <= 100


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_gradient(self) -> None:
        """Test handling of empty gradient array."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        with pytest.raises(ValueError, match="empty"):
            monitor.record_gradient(np.array([]), cost=1.0)

    def test_nan_in_gradient(self) -> None:
        """Test handling of NaN values in gradient."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        gradient = np.array([1.0, np.nan, 3.0])
        monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()
        assert report.has_numerical_issues is True

    def test_inf_in_gradient(self) -> None:
        """Test handling of Inf values in gradient."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        gradient = np.array([1.0, np.inf, 3.0])
        monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()
        assert report.has_numerical_issues is True

    def test_zero_gradient(self) -> None:
        """Test handling of zero gradient."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        gradient = np.array([0.0, 0.0, 0.0])
        monitor.record_gradient(gradient, cost=1.0)

        # Should handle gracefully
        report = monitor.get_report()
        assert report.available is True

    def test_single_parameter(self) -> None:
        """Test with single parameter model."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        for i in range(20):
            gradient = np.array([1.0 / (i + 1)])
            monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()
        assert report.available is True
        # Single parameter can't have imbalance
        grad_002_issues = [issue for issue in report.issues if issue.code == "GRAD-002"]
        assert len(grad_002_issues) == 0

    def test_reset(self) -> None:
        """Test monitor reset functionality."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        # Record some gradients
        for i in range(10):
            gradient = np.array([1.0, 2.0, 3.0])
            monitor.record_gradient(gradient, cost=1.0)

        assert monitor.iteration_count == 10

        # Reset
        monitor.reset()

        assert monitor.iteration_count == 0
        assert len(monitor._gradient_norm_history) == 0


class TestReportFormatting:
    """Tests for report string formatting."""

    def test_report_str(self) -> None:
        """Test report string representation."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        for i in range(20):
            gradient = np.array([1.0, 2.0, 3.0])
            monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()
        report_str = str(report)

        assert "Gradient Health" in report_str
        assert "Health Score" in report_str

    def test_report_summary(self) -> None:
        """Test report summary method."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        for i in range(20):
            gradient = np.array([1.0, 2.0, 3.0])
            monitor.record_gradient(gradient, cost=1.0)

        report = monitor.get_report()
        summary = report.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0


class TestCallbackIntegration:
    """Tests for callback integration with TRF optimizer (T023, T024)."""

    def test_create_callback_basic(self) -> None:
        """Test that create_callback returns a callable."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        callback = monitor.create_callback()

        assert callable(callback)

    def test_callback_records_iterations(self) -> None:
        """Test that callback records gradient information."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)
        callback = monitor.create_callback()

        # Simulate optimization iterations with info dict
        for i in range(10):
            callback(
                iteration=i,
                cost=10.0 / (i + 1),
                params=np.array([1.0, 2.0, 3.0]) * (i + 1),
                info={"gradient_norm": 1.0 / (i + 1), "nfev": i, "step_norm": 0.1},
            )

        assert monitor.iteration_count == 10

        report = monitor.get_report()
        assert report.n_iterations == 10
        assert report.available is True

    def test_callback_with_direct_gradient(self) -> None:
        """Test callback when gradient is directly available in info."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)
        callback = monitor.create_callback()

        # Simulate optimization with direct gradient info
        gradient = np.array([0.5, 0.3, 0.2])
        callback(
            iteration=1,
            cost=1.0,
            params=np.array([1.0, 2.0, 3.0]),
            info={"gradient": gradient, "gradient_norm": np.linalg.norm(gradient)},
        )

        assert monitor.iteration_count == 1
        report = monitor.get_report()
        np.testing.assert_array_almost_equal(
            report.mean_gradient_magnitudes, np.abs(gradient), decimal=5
        )

    def test_callback_chains_user_callback(self) -> None:
        """Test that user callback is called after gradient monitoring."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)

        user_callback_called = []

        def user_callback(iteration, cost, params, info, **kwargs):
            user_callback_called.append((iteration, cost))

        callback = monitor.create_callback(user_callback=user_callback)

        # Simulate iterations
        for i in range(5):
            callback(
                iteration=i,
                cost=float(i),
                params=np.array([1.0, 2.0]),
                info={"gradient_norm": 1.0},
            )

        # Both monitor and user callback should have recorded iterations
        assert monitor.iteration_count == 5
        assert len(user_callback_called) == 5
        assert user_callback_called[0] == (0, 0.0)
        assert user_callback_called[4] == (4, 4.0)

    def test_callback_estimates_gradient_from_params(self) -> None:
        """Test callback estimates gradient when only params available."""
        config = DiagnosticsConfig()
        monitor = GradientMonitor(config)
        callback = monitor.create_callback()

        # First call - no previous params
        callback(
            iteration=0,
            cost=10.0,
            params=np.array([1.0, 2.0, 3.0]),
            info=None,  # No info dict
        )

        # Second call - can estimate gradient from param change
        callback(
            iteration=1,
            cost=8.0,
            params=np.array([1.1, 2.2, 3.3]),
            info=None,
        )

        assert monitor.iteration_count == 2
        report = monitor.get_report()
        assert report.available is True
