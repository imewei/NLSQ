"""Edge case tests for Model Health Diagnostics System (Phase 8).

This module contains tests for edge cases identified in the spec:
1. Singular Jacobian (all zeros)
2. NaN/Inf in residuals
3. All parameters at bounds
4. No covariance matrix available
5. Plugin raises exception

Tests verify that the diagnostics system handles these edge cases gracefully,
providing partial information where possible and clear error messages.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from nlsq.diagnostics import (
    DiagnosticsConfig,
    GradientMonitor,
    IdentifiabilityAnalyzer,
    PluginRegistry,
    PluginResult,
    create_health_report,
    run_plugins,
)
from nlsq.diagnostics.types import (
    GradientHealthReport,
    HealthStatus,
    IdentifiabilityReport,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> DiagnosticsConfig:
    """Create default diagnostics configuration with verbose and warnings disabled."""
    return DiagnosticsConfig(verbose=False, emit_warnings=False)


@pytest.fixture(autouse=True)
def clear_plugin_registry() -> None:
    """Clear the plugin registry before and after each test."""
    PluginRegistry.clear()
    yield
    PluginRegistry.clear()


# =============================================================================
# Edge Case 1: Singular Jacobian (all zeros)
# =============================================================================


@pytest.mark.diagnostics
class TestSingularJacobian:
    """Tests for handling singular (all-zero) Jacobian matrices."""

    def test_all_zero_jacobian_detects_structural_unidentifiability(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that all-zero Jacobian is detected as structurally unidentifiable."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)
        J = np.zeros((100, 3))  # 100 data points, 3 parameters, all zeros

        report = analyzer.analyze(J)

        # Should complete without crashing
        assert report.available is True
        # Should detect that rank is 0 (structurally unidentifiable)
        assert report.numerical_rank == 0
        # Condition number should be infinite
        assert report.condition_number == float("inf") or report.condition_number > 1e15

    def test_all_zero_jacobian_generates_ident_001_issue(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that all-zero Jacobian generates IDENT-001 structural unidentifiability issue."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)
        J = np.zeros((50, 4))  # All zeros

        report = analyzer.analyze(J)

        # Should have IDENT-001 issue for structural unidentifiability
        issue_codes = [issue.code for issue in report.issues]
        assert "IDENT-001" in issue_codes

        # Verify the issue details
        ident_001 = next(i for i in report.issues if i.code == "IDENT-001")
        assert ident_001.severity == IssueSeverity.CRITICAL
        assert ident_001.category == IssueCategory.IDENTIFIABILITY
        assert "0" in ident_001.message  # Should mention rank is 0
        assert ident_001.details["numerical_rank"] == 0
        assert ident_001.details["n_params"] == 4

    def test_all_zero_jacobian_health_status_critical(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that all-zero Jacobian results in CRITICAL health status."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)
        J = np.zeros((20, 2))

        report = analyzer.analyze(J)

        assert report.health_status == HealthStatus.CRITICAL

    def test_near_zero_jacobian_handled_gracefully(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that near-zero Jacobian (very small values) is handled gracefully.

        Note: With float64 precision, very small but non-zero values are still
        well-conditioned relative to each other. This test verifies the system
        handles such cases without crashing, even if it determines them to be
        healthy (which is mathematically correct).
        """
        analyzer = IdentifiabilityAnalyzer(config=default_config)
        # Very small values - still well-conditioned relative to each other
        np.random.seed(42)
        J = np.random.randn(100, 3) * 1e-100

        report = analyzer.analyze(J)

        # Should complete without crashing - this is the key verification
        assert report.available is True
        # The report should have valid metrics
        assert report.n_params == 3
        assert report.numerical_rank >= 0

    def test_all_zero_jacobian_correlation_matrix_none_or_empty(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that all-zero Jacobian results in None correlation matrix."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)
        J = np.zeros((30, 3))

        report = analyzer.analyze(J)

        # Correlation matrix should be None (can't compute from singular FIM)
        # or have undefined correlations
        # Either None or diagonal zeros indicates inability to compute correlations
        if report.correlation_matrix is not None:
            # If computed, diagonal should be zero (no variance)
            assert np.all(np.diag(report.correlation_matrix) <= 0) or np.all(
                np.isnan(report.correlation_matrix)
            )


# =============================================================================
# Edge Case 2: NaN/Inf in residuals
# =============================================================================


@pytest.mark.diagnostics
class TestNaNInfResiduals:
    """Tests for handling NaN and Inf values in residuals and gradients."""

    def test_nan_residuals_in_gradient_monitor(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that GradientMonitor handles NaN values in gradients."""
        monitor = GradientMonitor(config=default_config)

        # Record some normal gradients first
        for i in range(5):
            gradient = np.array([0.1, 0.08, 0.12]) / (i + 1)
            monitor.record_gradient(gradient, cost=1.0 / (i + 1))

        # Record gradient with NaN
        nan_gradient = np.array([np.nan, 0.05, 0.03])
        monitor.record_gradient(nan_gradient, cost=0.5)

        # Should complete without crashing
        report = monitor.get_report()

        # Report should be available with partial information
        assert report.available is True
        assert report.n_iterations == 6
        # Should flag numerical issues
        assert report.has_numerical_issues is True

    def test_inf_residuals_in_gradient_monitor(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that GradientMonitor handles Inf values in gradients."""
        monitor = GradientMonitor(config=default_config)

        # Record gradient with positive infinity
        inf_gradient = np.array([np.inf, 0.05, 0.03])
        monitor.record_gradient(inf_gradient, cost=1.0)

        # Record gradient with negative infinity
        neg_inf_gradient = np.array([0.1, -np.inf, 0.03])
        monitor.record_gradient(neg_inf_gradient, cost=0.9)

        # Should complete without crashing
        report = monitor.get_report()

        assert report.available is True
        assert report.has_numerical_issues is True

    def test_all_nan_gradient_handled(self, default_config: DiagnosticsConfig) -> None:
        """Test that all-NaN gradient is handled gracefully."""
        monitor = GradientMonitor(config=default_config)

        all_nan_gradient = np.array([np.nan, np.nan, np.nan])
        monitor.record_gradient(all_nan_gradient, cost=1.0)

        report = monitor.get_report()

        assert report.available is True
        assert report.has_numerical_issues is True

    def test_nan_jacobian_returns_unavailable_report(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that Jacobian with NaN returns unavailable identifiability report."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)
        J = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])

        report = analyzer.analyze(J)

        assert report.available is False
        assert report.error_message is not None
        assert "nan" in report.error_message.lower() or "NaN" in report.error_message

    def test_inf_jacobian_returns_unavailable_report(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that Jacobian with Inf returns unavailable identifiability report."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)
        J = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])

        report = analyzer.analyze(J)

        assert report.available is False
        assert report.error_message is not None
        assert "inf" in report.error_message.lower() or "Inf" in report.error_message

    def test_mixed_nan_inf_gradient_health_score_reduced(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that NaN/Inf gradients reduce health score."""
        monitor = GradientMonitor(config=default_config)

        # Normal gradients
        for i in range(10):
            monitor.record_gradient(np.array([0.1, 0.1, 0.1]), cost=0.5)

        # Add problematic gradient
        monitor.record_gradient(np.array([np.nan, np.inf, 0.1]), cost=0.4)

        report = monitor.get_report()

        # Health score should be reduced due to numerical issues
        assert report.health_score < 1.0


# =============================================================================
# Edge Case 3: All parameters at bounds
# =============================================================================


@pytest.mark.diagnostics
class TestParametersAtBounds:
    """Tests for handling parameters at their bounds.

    Note: This is typically tested via curve_fit integration. These tests
    simulate the scenario and verify warning generation.
    """

    def test_parameters_at_bounds_warning_in_health_report(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that parameters at bounds generates appropriate warning.

        This simulates the scenario where an optimizer hits bounds by creating
        an identifiability report that would result from such a fit.
        """
        # Create a Jacobian that would be typical when parameters hit bounds
        # (reduced sensitivity in bounded direction)
        np.random.seed(42)
        n_data = 100
        n_params = 3

        J = np.random.randn(n_data, n_params)
        # Simulate parameter 0 hitting a bound (near-zero sensitivity)
        J[:, 0] = J[:, 0] * 1e-10  # Nearly zero gradient

        analyzer = IdentifiabilityAnalyzer(config=default_config)
        report = analyzer.analyze(J)

        # Should detect conditioning issues due to bound-hitting behavior
        assert report.available is True
        # Either high condition number or rank deficiency indicates problem
        has_conditioning_issue = (
            report.condition_number > default_config.condition_threshold
            or report.numerical_rank < n_params
        )
        assert has_conditioning_issue

    def test_single_parameter_at_bound_gradient_imbalance(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that single parameter at bound creates gradient imbalance."""
        monitor = GradientMonitor(config=default_config)

        # Simulate gradients where one parameter is essentially stuck at bound
        for i in range(20):
            # Parameter 0 has near-zero gradient (at bound)
            # Parameters 1, 2 have normal gradients
            gradient = np.array([1e-12, 1.0, 0.8])
            monitor.record_gradient(gradient, cost=1.0 / (i + 1))

        report = monitor.get_report()

        assert report.available is True
        # Should detect gradient imbalance (ratio > 1e6 default threshold)
        assert report.max_imbalance_ratio > 1e6
        # Use equality check for numpy boolean (not identity check)
        assert report.imbalance_detected == True  # noqa: E712

    def test_all_parameters_at_bounds_creates_issues(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that all parameters at bounds creates identifiability issues.

        When all parameters hit bounds, the Jacobian becomes singular or
        near-singular. This test verifies the system handles this case.
        """
        # When all parameters hit bounds, Jacobian becomes singular
        # Use exactly zero to guarantee detection
        J = np.zeros((50, 3))

        analyzer = IdentifiabilityAnalyzer(config=default_config)
        report = analyzer.analyze(J)

        # Should indicate severe problems
        assert report.available is True
        # All-zero Jacobian should be detected as rank-deficient
        assert report.numerical_rank == 0
        assert report.health_status == HealthStatus.CRITICAL


# =============================================================================
# Edge Case 4: No covariance matrix available
# =============================================================================


@pytest.mark.diagnostics
class TestNoCovarianceMatrix:
    """Tests for handling cases where covariance matrix is unavailable or contains infinities."""

    def test_identifiability_uses_jacobian_only_when_fim_singular(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that identifiability analysis works with Jacobian-only mode."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)

        # Create a valid Jacobian (covariance would come from inverting FIM)
        np.random.seed(42)
        J = np.random.randn(100, 3)

        report = analyzer.analyze(J)

        # Analysis should work even though we didn't provide pcov
        assert report.available is True
        assert report.n_params == 3
        assert report.numerical_rank > 0

    def test_analyze_from_singular_fim(self, default_config: DiagnosticsConfig) -> None:
        """Test analyze_from_fim handles singular FIM gracefully."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)

        # Create singular FIM (rank deficient)
        fim = np.array(
            [
                [1.0, 0.5, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 1.0],
            ]
        )
        # Make it singular by setting last row/col to linear combination
        fim[2, :] = fim[0, :] + fim[1, :]
        fim[:, 2] = fim[:, 0] + fim[:, 1]

        report = analyzer.analyze_from_fim(fim)

        assert report.available is True
        # Should detect the rank deficiency
        assert report.numerical_rank < 3 or report.condition_number > 1e10

    def test_health_report_with_none_identifiability(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test health report generation when identifiability is None."""
        # Create report with no identifiability analysis
        report = create_health_report(
            identifiability=None,
            gradient_health=GradientHealthReport(
                available=True,
                n_iterations=10,
                health_score=0.9,
                issues=[],
                health_status=HealthStatus.HEALTHY,
            ),
            config=default_config,
        )

        # Should still produce a valid report
        assert report.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
        assert report.identifiability is None

    def test_health_report_with_unavailable_identifiability(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test health report when identifiability analysis failed."""
        # Create identifiability report that indicates failure
        failed_ident = IdentifiabilityReport(
            available=False,
            error_message="Covariance matrix contains infinities",
            condition_number=float("inf"),
            numerical_rank=0,
            n_params=3,
            correlation_matrix=None,
            highly_correlated_pairs=[],
            issues=[],
            health_status=HealthStatus.CRITICAL,
        )

        report = create_health_report(
            identifiability=failed_ident,
            gradient_health=GradientHealthReport(
                available=True,
                n_iterations=50,
                health_score=0.8,
                issues=[],
                health_status=HealthStatus.HEALTHY,
            ),
            config=default_config,
        )

        # Report should note the limitation
        assert report.identifiability is not None
        assert report.identifiability.available is False
        assert (
            "infinite" in report.identifiability.error_message.lower()
            or "covariance" in report.identifiability.error_message.lower()
        )

    def test_correlation_matrix_none_handled_gracefully(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that None correlation matrix doesn't cause issues in reporting."""
        ident_report = IdentifiabilityReport(
            available=True,
            condition_number=1e5,
            numerical_rank=3,
            n_params=3,
            correlation_matrix=None,  # Explicitly None
            highly_correlated_pairs=[],
            issues=[],
            health_status=HealthStatus.HEALTHY,
        )

        report = create_health_report(
            identifiability=ident_report,
            config=default_config,
        )

        # Should produce valid report
        assert report.status == HealthStatus.HEALTHY
        # Summary should work without crashing
        summary = report.summary(verbose=True)
        assert isinstance(summary, str)


# =============================================================================
# Edge Case 5: Plugin raises exception
# =============================================================================


@pytest.mark.diagnostics
class TestPluginExceptionHandling:
    """Tests for handling plugin exceptions without affecting other diagnostics."""

    def test_failing_plugin_does_not_affect_identifiability(self) -> None:
        """Test that failing plugin doesn't prevent identifiability analysis."""

        # Register a failing plugin
        class FailingPlugin:
            @property
            def name(self) -> str:
                return "failing-plugin"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise RuntimeError("Plugin crashed intentionally")

        PluginRegistry.register(FailingPlugin())

        # Run identifiability analysis separately
        config = DiagnosticsConfig(verbose=False, emit_warnings=False)
        analyzer = IdentifiabilityAnalyzer(config=config)
        J = np.random.randn(100, 3)

        # Identifiability should work regardless of registered plugins
        report = analyzer.analyze(J)
        assert report.available is True
        assert report.n_params == 3

    def test_failing_plugin_does_not_affect_gradient_health(self) -> None:
        """Test that failing plugin doesn't prevent gradient health monitoring."""

        class FailingPlugin:
            @property
            def name(self) -> str:
                return "crashing-plugin"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise ValueError("Intentional crash")

        PluginRegistry.register(FailingPlugin())

        # Gradient monitor should work regardless
        config = DiagnosticsConfig(verbose=False, emit_warnings=False)
        monitor = GradientMonitor(config=config)

        for i in range(10):
            monitor.record_gradient(np.array([0.1, 0.08, 0.12]), cost=1.0 / (i + 1))

        report = monitor.get_report()
        assert report.available is True
        assert report.n_iterations == 10

    def test_failing_plugin_result_shows_unavailable_with_error(self) -> None:
        """Test that failed plugin result shows available=False with error message."""

        class ExceptionPlugin:
            @property
            def name(self) -> str:
                return "exception-raiser"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise TypeError("Type error in plugin")

        PluginRegistry.register(ExceptionPlugin())

        J = np.random.randn(50, 2)
        params = np.array([1.0, 2.0])
        residuals = np.array([0.1, -0.1, 0.05])

        results = run_plugins(J, params, residuals)

        assert "exception-raiser" in results
        result = results["exception-raiser"]
        assert result.available is False
        assert result.error_message is not None
        assert "Type error" in result.error_message

    def test_multiple_plugins_one_fails_others_succeed(self) -> None:
        """Test that one failing plugin doesn't affect other plugins."""

        class GoodPlugin:
            @property
            def name(self) -> str:
                return "good-plugin"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                return PluginResult(
                    plugin_name=self.name,
                    data={"computed": True},
                    issues=[],
                )

        class BadPlugin:
            @property
            def name(self) -> str:
                return "bad-plugin"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise Exception("Generic exception")

        PluginRegistry.register(GoodPlugin())
        PluginRegistry.register(BadPlugin())

        J = np.random.randn(30, 2)
        params = np.array([1.0, 2.0])
        residuals = np.array([0.1, 0.2, 0.3])

        results = run_plugins(J, params, residuals)

        # Both plugins should have results
        assert len(results) == 2

        # Good plugin should succeed
        assert results["good-plugin"].available is True
        assert results["good-plugin"].data["computed"] is True

        # Bad plugin should show failure
        assert results["bad-plugin"].available is False
        assert "Generic exception" in results["bad-plugin"].error_message  # type: ignore[operator]

    def test_failing_plugin_emits_warning(self) -> None:
        """Test that failing plugin emits a Python warning."""

        class WarningPlugin:
            @property
            def name(self) -> str:
                return "warning-trigger"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise RuntimeError("This should trigger a warning")

        PluginRegistry.register(WarningPlugin())

        J = np.random.randn(20, 2)
        params = np.array([1.0, 2.0])
        residuals = np.array([0.1, 0.2])

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            run_plugins(J, params, residuals)

            # Should have emitted a warning about the plugin failure
            plugin_warnings = [
                w for w in caught_warnings if "warning-trigger" in str(w.message)
            ]
            assert len(plugin_warnings) >= 1

    def test_health_report_includes_failed_plugin_results(self) -> None:
        """Test that health report includes failed plugin results."""

        class FailPlugin:
            @property
            def name(self) -> str:
                return "fail-in-report"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise KeyError("Missing key")

        PluginRegistry.register(FailPlugin())

        J = np.random.randn(50, 3)
        params = np.array([1.0, 2.0, 3.0])
        residuals = np.array([0.1, -0.1, 0.05])

        plugin_results = run_plugins(J, params, residuals)

        config = DiagnosticsConfig(verbose=False, emit_warnings=False)
        report = create_health_report(
            identifiability=None,
            gradient_health=None,
            plugin_results=plugin_results,
            config=config,
        )

        # Plugin results should be included
        assert "fail-in-report" in report.plugin_results
        assert report.plugin_results["fail-in-report"].available is False


# =============================================================================
# Combined Edge Cases
# =============================================================================


@pytest.mark.diagnostics
class TestCombinedEdgeCases:
    """Tests for combinations of edge cases occurring together."""

    def test_zero_jacobian_with_nan_residuals(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test handling of zero Jacobian with NaN residuals simultaneously."""
        # Zero Jacobian
        J = np.zeros((50, 3))

        analyzer = IdentifiabilityAnalyzer(config=default_config)
        ident_report = analyzer.analyze(J)

        # Gradient monitor with NaN
        monitor = GradientMonitor(config=default_config)
        nan_gradient = np.array([np.nan, 0.0, 0.0])
        monitor.record_gradient(nan_gradient, cost=np.nan)
        grad_report = monitor.get_report()

        # Both should complete
        assert ident_report.available is True
        assert grad_report.available is True

        # Both should indicate problems
        assert ident_report.health_status == HealthStatus.CRITICAL
        assert grad_report.has_numerical_issues is True

    def test_health_report_with_multiple_failures(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test health report when multiple components have issues."""
        # Failed identifiability
        failed_ident = IdentifiabilityReport(
            available=False,
            error_message="Jacobian contains NaN",
            n_params=3,
        )

        # Gradient with numerical issues
        grad_report = GradientHealthReport(
            available=True,
            n_iterations=10,
            health_score=0.5,
            has_numerical_issues=True,
            issues=[
                ModelHealthIssue(
                    category=IssueCategory.GRADIENT,
                    severity=IssueSeverity.WARNING,
                    code="GRAD-001",
                    message="Vanishing gradients detected",
                    affected_parameters=None,
                    details={},
                    recommendation="Check model parameterization",
                )
            ],
            health_status=HealthStatus.WARNING,
        )

        report = create_health_report(
            identifiability=failed_ident,
            gradient_health=grad_report,
            config=default_config,
        )

        # Should aggregate issues appropriately
        assert report.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]
        assert len(report.all_issues) >= 1

    def test_empty_inputs_handled_gracefully(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test that empty inputs are handled gracefully."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)

        # Empty Jacobian
        empty_J = np.array([]).reshape(0, 3)
        report = analyzer.analyze(empty_J)

        assert report.available is False
        assert report.error_message is not None
        assert "empty" in report.error_message.lower()

    def test_1d_jacobian_rejected(self, default_config: DiagnosticsConfig) -> None:
        """Test that 1D Jacobian is rejected with clear error."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)

        # 1D array instead of 2D matrix
        J_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        report = analyzer.analyze(J_1d)

        assert report.available is False
        assert report.error_message is not None
        assert "2D" in report.error_message or "2d" in report.error_message.lower()


# =============================================================================
# Stress Tests for Edge Cases
# =============================================================================


@pytest.mark.diagnostics
class TestEdgeCaseStress:
    """Stress tests for edge case handling."""

    def test_large_singular_jacobian(self, default_config: DiagnosticsConfig) -> None:
        """Test handling of large singular Jacobian."""
        analyzer = IdentifiabilityAnalyzer(config=default_config)

        # Large all-zero Jacobian
        J = np.zeros((10000, 20))

        report = analyzer.analyze(J)

        assert report.available is True
        assert report.numerical_rank == 0
        assert report.health_status == HealthStatus.CRITICAL

    def test_many_nan_values_in_sequence(
        self, default_config: DiagnosticsConfig
    ) -> None:
        """Test gradient monitor with many NaN values in sequence."""
        monitor = GradientMonitor(config=default_config)

        # 100 iterations with NaN values interspersed
        for i in range(100):
            if i % 5 == 0:
                gradient = np.array([np.nan, 0.1, 0.1])
            else:
                gradient = np.array([0.1, 0.1, 0.1]) / (i + 1)
            monitor.record_gradient(gradient, cost=1.0 / (i + 1))

        report = monitor.get_report()

        assert report.available is True
        assert report.has_numerical_issues is True
        assert report.n_iterations == 100

    def test_repeated_plugin_failures(self) -> None:
        """Test that repeated plugin failures don't cause memory issues."""

        class RepeatedFailPlugin:
            def __init__(self, fail_count: int = 0) -> None:
                self.fail_count = fail_count

            @property
            def name(self) -> str:
                return f"repeat-fail-{self.fail_count}"

            def analyze(
                self,
                jacobian: np.ndarray,
                parameters: np.ndarray,
                residuals: np.ndarray,
                **context: Any,
            ) -> PluginResult:
                raise RuntimeError(f"Failure {self.fail_count}")

        # Register many failing plugins
        for i in range(10):
            PluginRegistry.register(RepeatedFailPlugin(fail_count=i))

        J = np.random.randn(20, 3)
        params = np.array([1.0, 2.0, 3.0])
        residuals = np.array([0.1, 0.2, 0.3])

        # Should handle all failures gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = run_plugins(J, params, residuals)

        assert len(results) == 10
        assert all(not r.available for r in results.values())
