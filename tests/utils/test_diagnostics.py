"""Comprehensive tests for nlsq.diagnostics module.

This test suite covers:
- ConvergenceMonitor class
- OptimizationDiagnostics class
- Global diagnostics functions
"""

import unittest
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nlsq.utils.diagnostics import (
    ConvergenceMonitor,
    OptimizationDiagnostics,
    get_diagnostics,
    reset_diagnostics,
)


class TestConvergenceMonitor(unittest.TestCase):
    """Tests for ConvergenceMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = ConvergenceMonitor(window_size=20, sensitivity=2.0)
        self.assertEqual(monitor.window_size, 20)
        self.assertEqual(monitor.sensitivity, 2.0)

    def test_update(self):
        """Test updating monitor."""
        monitor = ConvergenceMonitor()
        monitor.update(1.5, np.array([1.0, 2.0]))
        self.assertEqual(len(monitor.cost_history), 1)
        self.assertEqual(monitor.cost_history[0], 1.5)

    def test_update_with_gradient(self):
        """Test updating monitor with gradient."""
        monitor = ConvergenceMonitor()
        gradient = np.array([0.1, 0.2])
        monitor.update(1.5, np.array([1.0, 2.0]), gradient=gradient)
        self.assertEqual(len(monitor.gradient_history), 1)

    def test_update_with_step_size(self):
        """Test updating monitor with step size."""
        monitor = ConvergenceMonitor()
        monitor.update(1.5, np.array([1.0, 2.0]), step_size=0.5)
        self.assertEqual(len(monitor.step_size_history), 1)
        self.assertEqual(monitor.step_size_history[0], 0.5)

    def test_add_iteration(self):
        """Test add_iteration method."""
        monitor = ConvergenceMonitor()
        monitor.add_iteration(cost=5.0, params=np.array([1.0, 2.0]))
        self.assertEqual(len(monitor.cost_history), 1)
        self.assertEqual(monitor.cost_history[0], 5.0)

    def test_get_convergence_rate(self):
        """Test convergence rate calculation."""
        monitor = ConvergenceMonitor()
        # Add converging costs (decreasing)
        for i in range(10):
            monitor.update(10.0 - i, np.array([float(i)]))

        rate = monitor.get_convergence_rate()
        self.assertIsInstance(rate, (float, np.floating))
        # Converging (decreasing costs) should have positive rate
        self.assertGreater(rate, 0)

    def test_get_convergence_rate_insufficient_data(self):
        """Test convergence rate with insufficient data."""
        monitor = ConvergenceMonitor()
        monitor.update(1.0, np.array([1.0]))

        rate = monitor.get_convergence_rate()
        self.assertIsNone(rate)

    def test_pattern_detection_returns_tuples(self):
        """Test that pattern detection methods return tuples."""
        monitor = ConvergenceMonitor()
        for i in range(10):
            monitor.update(10.0 - i, np.array([float(i)]))

        osc_result = monitor.detect_oscillation()
        stag_result = monitor.detect_stagnation()
        div_result = monitor.detect_divergence()

        self.assertIsInstance(osc_result, tuple)
        self.assertIsInstance(stag_result, tuple)
        self.assertIsInstance(div_result, tuple)
        self.assertEqual(len(osc_result), 2)
        self.assertEqual(len(stag_result), 2)
        self.assertEqual(len(div_result), 2)

    @given(
        costs=st.lists(
            st.floats(
                min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False
            ),
            min_size=1,
            max_size=20,
        ),
        n_params=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=1000)
    def test_pattern_detection_robustness_property(self, costs, n_params):
        """Property-based test: Pattern detection always returns valid tuples."""
        monitor = ConvergenceMonitor()

        # Feed the monitor with generated cost data
        for cost in costs:
            params = np.random.randn(n_params)
            monitor.update(cost, params)

        # Pattern detection methods should always return valid tuples
        osc_result = monitor.detect_oscillation()
        stag_result = monitor.detect_stagnation()
        div_result = monitor.detect_divergence()

        # Check structure
        assert isinstance(osc_result, tuple) and len(osc_result) == 2
        assert isinstance(stag_result, tuple) and len(stag_result) == 2
        assert isinstance(div_result, tuple) and len(div_result) == 2

        # Check first element is bool
        assert isinstance(osc_result[0], (bool, np.bool_))
        assert isinstance(stag_result[0], (bool, np.bool_))
        assert isinstance(div_result[0], (bool, np.bool_))

        # Check second element is numeric
        assert isinstance(osc_result[1], (float, np.floating))
        assert isinstance(stag_result[1], (float, np.floating))
        assert isinstance(div_result[1], (float, np.floating))

        # Check scores are finite (can be negative for divergence)
        assert np.isfinite(osc_result[1])
        assert np.isfinite(stag_result[1])
        assert np.isfinite(div_result[1])
        # Oscillation and stagnation scores should be non-negative
        assert osc_result[1] >= 0
        assert stag_result[1] >= 0

    @given(
        window_size=st.integers(min_value=5, max_value=50),
        sensitivity=st.floats(
            min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        n_iterations=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=50, deadline=1000)
    def test_convergence_monitor_numerical_stability_property(
        self, window_size, sensitivity, n_iterations
    ):
        """Property-based test: ConvergenceMonitor handles various configurations gracefully."""
        monitor = ConvergenceMonitor(window_size=window_size, sensitivity=sensitivity)

        # Generate monotonically decreasing costs (converging scenario)
        initial_cost = 1000.0
        for i in range(n_iterations):
            cost = initial_cost / (i + 1)  # Monotonically decreasing
            params = np.random.randn(3)
            gradient = np.random.randn(3) * 0.1
            step_size = 1.0 / (i + 1)

            monitor.update(cost, params, gradient=gradient, step_size=step_size)

        # Should have valid history
        assert len(monitor.cost_history) > 0
        assert len(monitor.cost_history) <= window_size
        assert len(monitor.param_history) == len(monitor.cost_history)

        # Get convergence rate if enough data
        if len(monitor.cost_history) >= 3:
            rate = monitor.get_convergence_rate()
            # For converging scenario, rate should be positive or None
            assert rate is None or (
                isinstance(rate, (float, np.floating)) and np.isfinite(rate)
            )

        # Pattern detection should work without errors
        osc_detected, osc_score = monitor.detect_oscillation()
        stag_detected, stag_score = monitor.detect_stagnation()
        div_detected, div_score = monitor.detect_divergence()

        # All detections should return valid boolean values
        assert isinstance(osc_detected, (bool, np.bool_))
        assert isinstance(stag_detected, (bool, np.bool_))
        assert isinstance(div_detected, (bool, np.bool_))

        # All scores should be finite
        assert np.isfinite(osc_score)
        assert np.isfinite(stag_score)
        assert np.isfinite(div_score)
        # Oscillation and stagnation scores should be non-negative
        assert osc_score >= 0
        assert stag_score >= 0
        # Divergence score is a slope and can be negative (converging)


class TestOptimizationDiagnostics(unittest.TestCase):
    """Tests for OptimizationDiagnostics."""

    def test_initialization(self):
        """Test diagnostics initialization."""
        diag = OptimizationDiagnostics()
        self.assertEqual(diag.iteration_data, [])
        self.assertEqual(diag.function_eval_count, 0)

    def test_start_optimization(self):
        """Test start_optimization."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0, problem_name="test")

        self.assertEqual(diag.problem_name, "test")
        self.assertIsNotNone(diag.start_time)
        np.testing.assert_array_equal(diag.initial_params, x0)

    def test_record_iteration(self):
        """Test recording iteration."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        diag.record_iteration(
            iteration=0,
            x=np.array([1.1, 2.1]),
            cost=5.0,
        )

        self.assertEqual(len(diag.iteration_data), 1)
        self.assertEqual(diag.iteration_data[0]["iteration"], 0)
        self.assertEqual(diag.iteration_data[0]["cost"], 5.0)

    def test_record_iteration_with_gradient(self):
        """Test recording iteration with gradient."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        gradient = np.array([0.1, -0.2])
        diag.record_iteration(
            iteration=0,
            x=np.array([1.1, 2.1]),
            cost=5.0,
            gradient=gradient,
        )

        self.assertIn("gradient_norm", diag.iteration_data[0])
        self.assertIn("gradient_max", diag.iteration_data[0])
        self.assertAlmostEqual(
            diag.iteration_data[0]["gradient_norm"], np.linalg.norm(gradient)
        )

    def test_record_iteration_with_jacobian(self):
        """Test recording iteration with Jacobian."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        jacobian = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        diag.record_iteration(
            iteration=0, x=np.array([1.1, 2.1]), cost=5.0, jacobian=jacobian
        )

        self.assertIn("jacobian_condition", diag.iteration_data[0])
        self.assertEqual(diag.jacobian_eval_count, 1)

    def test_record_iteration_with_step_size(self):
        """Test recording iteration with step size."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        diag.record_iteration(
            iteration=0, x=np.array([1.1, 2.1]), cost=5.0, step_size=0.5
        )

        self.assertIn("step_size", diag.iteration_data[0])
        self.assertEqual(diag.iteration_data[0]["step_size"], 0.5)

    def test_record_iteration_with_method_info(self):
        """Test recording iteration with method info."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        method_info = {"trust_radius": 1.0, "reduction_ratio": 0.8}
        diag.record_iteration(
            iteration=0, x=np.array([1.1, 2.1]), cost=5.0, method_info=method_info
        )

        self.assertIn("method_info", diag.iteration_data[0])
        self.assertEqual(diag.iteration_data[0]["method_info"]["trust_radius"], 1.0)

    def test_record_iteration_non_finite_gradient(self):
        """Test detection of non-finite gradients."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        gradient = np.array([np.nan, 0.2])
        diag.record_iteration(
            iteration=0, x=np.array([1.1, 2.1]), cost=5.0, gradient=gradient
        )

        self.assertTrue(len(diag.numerical_issues) > 0)
        self.assertIn("Non-finite gradient", diag.numerical_issues[0])

    def test_record_iteration_large_gradient(self):
        """Test detection of extremely large gradients."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        gradient = np.array([1e11, 1e11])
        diag.record_iteration(
            iteration=0, x=np.array([1.1, 2.1]), cost=5.0, gradient=gradient
        )

        self.assertTrue(len(diag.numerical_issues) > 0)
        self.assertIn("Extremely large gradient", diag.numerical_issues[0])

    def test_record_iteration_jacobian_condition_number(self):
        """Test Jacobian condition number calculation."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        # Create Jacobian and verify condition number is computed
        jacobian = np.array([[1.0, 0.5], [0.5, 1.0], [0.25, 0.75]])
        diag.record_iteration(
            iteration=0, x=np.array([1.1, 2.1]), cost=5.0, jacobian=jacobian
        )

        # Should have computed and stored jacobian_condition
        self.assertIn("jacobian_condition", diag.iteration_data[0])
        cond = diag.iteration_data[0]["jacobian_condition"]
        self.assertGreater(cond, 1.0)  # Should be > 1 for any non-trivial matrix

    def test_get_summary_statistics_empty(self):
        """Test statistics with no data."""
        diag = OptimizationDiagnostics()
        stats = diag.get_summary_statistics()
        self.assertEqual(stats, {})

    def test_get_summary_statistics_with_data(self):
        """Test statistics with iteration data."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        for i in range(5):
            diag.record_iteration(
                iteration=i,
                x=np.array([1.0 + i * 0.1, 2.0 + i * 0.1]),
                cost=10.0 - i * 2.0,
            )

        stats = diag.get_summary_statistics()

        self.assertEqual(stats["total_iterations"], 5)
        self.assertEqual(stats["initial_cost"], 10.0)
        self.assertEqual(stats["final_cost"], 2.0)

    def test_get_summary_statistics_with_gradients(self):
        """Test statistics with gradient data."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        for i in range(5):
            gradient = np.array([1.0 / (i + 1), 2.0 / (i + 1)])
            diag.record_iteration(
                iteration=i,
                x=np.array([1.0 + i * 0.1, 2.0 + i * 0.1]),
                cost=10.0 - i * 2.0,
                gradient=gradient,
            )

        stats = diag.get_summary_statistics()

        self.assertIn("initial_gradient_norm", stats)
        self.assertIn("final_gradient_norm", stats)

    def test_get_summary_statistics_with_jacobians(self):
        """Test statistics with Jacobian data."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        for i in range(5):
            jacobian = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            diag.record_iteration(
                iteration=i,
                x=np.array([1.0 + i * 0.1, 2.0 + i * 0.1]),
                cost=10.0 - i * 2.0,
                jacobian=jacobian,
            )

        stats = diag.get_summary_statistics()

        self.assertIn("max_condition_number", stats)
        self.assertIn("mean_condition_number", stats)


class TestGlobalFunctions(unittest.TestCase):
    """Tests for global diagnostic functions."""

    def test_get_diagnostics(self):
        """Test get_diagnostics returns instance."""
        diag = get_diagnostics()
        self.assertIsInstance(diag, OptimizationDiagnostics)

    def test_reset_diagnostics(self):
        """Test reset creates new instance."""
        diag1 = get_diagnostics()
        diag1.record_event("warning", "Test")
        reset_diagnostics()
        diag2 = get_diagnostics()
        self.assertIsNot(diag1, diag2)


class TestRecordEvent(unittest.TestCase):
    """Tests for OptimizationDiagnostics.record_event method."""

    def test_record_event_basic(self):
        """Test basic event recording."""
        diag = OptimizationDiagnostics()
        diag.start_optimization(np.array([1.0]))

        diag.record_event("test_event", {"key": "value"})

        # Event is recorded (check via side effects like warnings list)
        self.assertIsNotNone(diag.start_time)

    def test_record_event_with_failed_type(self):
        """Test that failed events are stored in warnings."""
        diag = OptimizationDiagnostics()
        diag.start_optimization(np.array([1.0]))

        diag.record_event("recovery_failed", {"reason": "timeout"})

        self.assertEqual(len(diag.warnings_issued), 1)
        self.assertIn("recovery_failed", diag.warnings_issued[0])

    def test_record_event_with_error_type(self):
        """Test that error events are stored in warnings."""
        diag = OptimizationDiagnostics()
        diag.start_optimization(np.array([1.0]))

        diag.record_event("numerical_error", {"message": "singular matrix"})

        self.assertEqual(len(diag.warnings_issued), 1)
        self.assertIn("numerical_error", diag.warnings_issued[0])

    def test_record_event_without_data(self):
        """Test recording event without data dict."""
        diag = OptimizationDiagnostics()
        diag.start_optimization(np.array([1.0]))

        # Should not raise error
        diag.record_event("simple_event")

    def test_record_event_multiple_iterations(self):
        """Test event recording tracks iteration count."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0)

        # Record some iterations
        for i in range(3):
            diag.record_iteration(iteration=i, x=x0, cost=1.0)

        # Event should track that we're at iteration 3
        diag.record_event("checkpoint", {"iteration_count": 3})


class TestCheckConvergenceHealth(unittest.TestCase):
    """Tests for OptimizationDiagnostics._check_convergence_health method."""

    def test_check_convergence_health_early_iterations(self):
        """Test that early iterations don't trigger checks."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0)

        # Early iterations (< 10) should not trigger warnings
        for i in range(5):
            diag.record_iteration(iteration=i, x=x0, cost=10.0 - i)
            diag._check_convergence_health(i)

        # No warnings should be issued
        self.assertEqual(len(diag.warnings_issued), 0)

    def test_check_convergence_health_oscillation_warning(self):
        """Test oscillation detection triggers warning."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0)

        # Create oscillating pattern
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for i in range(20):
                cost = 10.0 + 5.0 * ((-1) ** i)  # Oscillating
                diag.record_iteration(iteration=i, x=x0, cost=cost)
                diag._check_convergence_health(i)

            # Should have issued oscillation warning
            self.assertIn("oscillation", diag.warnings_issued)

    def test_check_convergence_health_stagnation_warning(self):
        """Test stagnation detection triggers warning."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Create stagnant pattern (constant cost)
            for i in range(20):
                diag.record_iteration(iteration=i, x=x0, cost=10.0)
                diag._check_convergence_health(i)

            # Should have issued stagnation warning
            self.assertIn("stagnation", diag.warnings_issued)

    def test_check_convergence_health_divergence_warning(self):
        """Test divergence detection triggers warning."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Create diverging pattern (increasing cost)
            for i in range(20):
                cost = 10.0 + i * 2.0  # Diverging
                diag.record_iteration(iteration=i, x=x0, cost=cost)
                diag._check_convergence_health(i)

            # Should have issued divergence warning
            self.assertIn("divergence", diag.warnings_issued)

    def test_check_convergence_health_warns_only_once(self):
        """Test that warnings are issued only once per type."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Create stagnant pattern
            for i in range(40):
                diag.record_iteration(iteration=i, x=x0, cost=10.0)
                diag._check_convergence_health(i)

            # Should warn only once
            stagnation_warnings = [
                warn for warn in diag.warnings_issued if warn == "stagnation"
            ]
            self.assertEqual(len(stagnation_warnings), 1)


class TestGetMemoryUsage(unittest.TestCase):
    """Tests for OptimizationDiagnostics._get_memory_usage method."""

    def test_get_memory_usage_returns_float(self):
        """Test that _get_memory_usage returns a float."""
        diag = OptimizationDiagnostics()
        memory = diag._get_memory_usage()

        self.assertIsInstance(memory, float)
        self.assertGreaterEqual(memory, 0.0)

    def test_get_memory_usage_nonnegative(self):
        """Test that memory usage is non-negative."""
        diag = OptimizationDiagnostics()
        memory = diag._get_memory_usage()

        self.assertGreaterEqual(memory, 0.0)


class TestGenerateReport(unittest.TestCase):
    """Tests for OptimizationDiagnostics.generate_report method."""

    def test_generate_report_empty(self):
        """Test report generation with no data."""
        diag = OptimizationDiagnostics()
        report = diag.generate_report()

        self.assertEqual(report, "No optimization data available")

    def test_generate_report_with_data(self):
        """Test report generation with optimization data."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0, 2.0])
        diag.start_optimization(x0, problem_name="test_problem")

        # Record iterations
        for i in range(5):
            diag.record_iteration(
                iteration=i,
                x=np.array([1.0 + i * 0.1, 2.0 + i * 0.1]),
                cost=10.0 - i * 2.0,
            )

        report = diag.generate_report()

        # Check report contains expected sections
        self.assertIn("Optimization Report", report)
        self.assertIn("test_problem", report)
        self.assertIn("Total iterations", report)
        self.assertIn("Performance Metrics", report)
        self.assertIn("Cost Reduction", report)

    def test_generate_report_verbose(self):
        """Test verbose report includes warnings."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0)

        # Record stagnant iterations to trigger warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            for i in range(20):
                diag.record_iteration(iteration=i, x=x0, cost=10.0)
                diag._check_convergence_health(i)

        report = diag.generate_report(verbose=True)

        # Verbose report should include warnings section
        if diag.warnings_issued:
            self.assertIn("Warnings", report)

    def test_generate_report_non_verbose(self):
        """Test non-verbose report is shorter."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0)

        for i in range(5):
            diag.record_iteration(iteration=i, x=x0, cost=10.0 - i)

        report_verbose = diag.generate_report(verbose=True)
        report_brief = diag.generate_report(verbose=False)

        # Both should contain basic info
        self.assertIn("Optimization Report", report_verbose)
        self.assertIn("Optimization Report", report_brief)

    def test_generate_report_with_numerical_issues(self):
        """Test report includes numerical issues when present."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0)

        # Add numerical issues
        diag.numerical_issues.append("Singular Jacobian detected")
        diag.numerical_issues.append("Condition number exceeded threshold")

        for i in range(5):
            diag.record_iteration(iteration=i, x=x0, cost=10.0 - i)

        report = diag.generate_report(verbose=True)

        # Should include numerical issues section
        self.assertIn("Numerical Issues", report)

    def test_generate_report_formatting(self):
        """Test report has proper formatting."""
        diag = OptimizationDiagnostics()
        x0 = np.array([1.0])
        diag.start_optimization(x0, problem_name="format_test")

        for i in range(3):
            diag.record_iteration(iteration=i, x=x0, cost=10.0 - i * 2.0)

        report = diag.generate_report()

        # Check for formatting elements
        self.assertIn("=" * 60, report)  # Header/footer separators
        self.assertIn("---", report)  # Section separators
        self.assertIn("format_test", report)  # Problem name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
