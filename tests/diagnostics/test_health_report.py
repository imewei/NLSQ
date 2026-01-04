"""Unit tests for ModelHealthReport aggregation and summary formatting (User Story 3).

Tests cover T027 and T028:
- T027: ModelHealthReport aggregation tests
  - Status determination (HEALTHY/WARNING/CRITICAL)
  - Health score computation with weighted contributions
  - Issue aggregation from all components
  - Partial component availability
  - All components unavailable

- T028: summary() output formatting tests
  - Healthy model output format (contract B4)
  - Issues present output format (contract B5)
  - Verbose vs non-verbose output
  - Recommendations section

Contract reference: /specs/005-model-health-diagnostics/contracts/health_report.md

This module is marked serial because it depends on PluginRegistry global state
through the diagnostics subsystem, causing race conditions in parallel execution.
"""

import numpy as np
import pytest

# Mark all tests in this module as serial to avoid PluginRegistry race conditions.
# The root conftest.py assigns serial tests to the same xdist worker group.
pytestmark = pytest.mark.serial

from nlsq.diagnostics.types import (
    DiagnosticLevel,
    DiagnosticsConfig,
    GradientHealthReport,
    HealthStatus,
    IdentifiabilityReport,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
)

# These imports will fail until implementation is complete:
# from nlsq.diagnostics.health_report import create_health_report, ModelHealthReport


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def healthy_identifiability_report() -> IdentifiabilityReport:
    """Create a healthy identifiability report with no issues."""
    return IdentifiabilityReport(
        condition_number=1.23e4,
        numerical_rank=3,
        n_params=3,
        correlation_matrix=np.eye(3),
        highly_correlated_pairs=[],
        issues=[],
        health_status=HealthStatus.HEALTHY,
        computation_time_ms=5.0,
    )


@pytest.fixture
def healthy_gradient_report() -> GradientHealthReport:
    """Create a healthy gradient health report with no issues."""
    return GradientHealthReport(
        n_iterations=100,
        health_score=1.0,
        mean_gradient_norm=0.01,
        final_gradient_norm=0.001,
        mean_gradient_magnitudes=np.array([0.01, 0.01, 0.01]),
        variance_gradient_magnitudes=np.array([0.001, 0.001, 0.001]),
        max_imbalance_ratio=1.5,
        has_numerical_issues=False,
        vanishing_detected=False,
        imbalance_detected=False,
        stagnation_detected=False,
        issues=[],
        health_status=HealthStatus.HEALTHY,
        computation_time_ms=2.0,
    )


@pytest.fixture
def warning_identifiability_report() -> IdentifiabilityReport:
    """Create an identifiability report with WARNING issues (practical unidentifiability)."""
    issue = ModelHealthIssue(
        category=IssueCategory.IDENTIFIABILITY,
        severity=IssueSeverity.WARNING,
        code="IDENT-002",
        message="Practical unidentifiability detected (condition number: 1.50e+10)",
        affected_parameters=None,
        details={"condition_number": 1.5e10},
        recommendation="Consider rescaling parameters or collecting more data",
    )
    return IdentifiabilityReport(
        condition_number=1.5e10,
        numerical_rank=3,
        n_params=3,
        correlation_matrix=np.eye(3),
        highly_correlated_pairs=[],
        issues=[issue],
        health_status=HealthStatus.WARNING,
        computation_time_ms=5.0,
    )


@pytest.fixture
def critical_identifiability_report() -> IdentifiabilityReport:
    """Create an identifiability report with CRITICAL issues (structural unidentifiability)."""
    issue = ModelHealthIssue(
        category=IssueCategory.IDENTIFIABILITY,
        severity=IssueSeverity.CRITICAL,
        code="IDENT-001",
        message="Structural unidentifiability detected (rank 2 < 3 parameters)",
        affected_parameters=(0, 1, 2),
        details={"numerical_rank": 2, "n_params": 3},
        recommendation="Consider reparameterizing the model",
    )
    return IdentifiabilityReport(
        condition_number=float("inf"),
        numerical_rank=2,
        n_params=3,
        correlation_matrix=None,
        highly_correlated_pairs=[],
        issues=[issue],
        health_status=HealthStatus.CRITICAL,
        computation_time_ms=5.0,
    )


@pytest.fixture
def correlation_warning_report() -> IdentifiabilityReport:
    """Create an identifiability report with highly correlated parameters."""
    issue = ModelHealthIssue(
        category=IssueCategory.CORRELATION,
        severity=IssueSeverity.WARNING,
        code="CORR-001",
        message="Parameters 0 and 2 are highly correlated (r=0.97)",
        affected_parameters=(0, 2),
        details={"correlation": 0.97},
        recommendation="Consider reparameterizing to combine correlated parameters",
    )
    return IdentifiabilityReport(
        condition_number=1e5,
        numerical_rank=3,
        n_params=3,
        correlation_matrix=np.array(
            [
                [1.0, 0.3, 0.97],
                [0.3, 1.0, 0.2],
                [0.97, 0.2, 1.0],
            ]
        ),
        highly_correlated_pairs=[(0, 2, 0.97)],
        issues=[issue],
        health_status=HealthStatus.WARNING,
        computation_time_ms=5.0,
    )


@pytest.fixture
def warning_gradient_report() -> GradientHealthReport:
    """Create a gradient health report with WARNING issues (imbalance)."""
    issue = ModelHealthIssue(
        category=IssueCategory.GRADIENT,
        severity=IssueSeverity.WARNING,
        code="GRAD-002",
        message="Gradient imbalance detected (ratio=1.2e+07)",
        affected_parameters=(0, 2),
        details={"imbalance_ratio": 1.2e7},
        recommendation="Consider rescaling parameters to similar magnitudes",
    )
    return GradientHealthReport(
        n_iterations=100,
        health_score=0.6,
        mean_gradient_norm=100.0,
        final_gradient_norm=50.0,
        mean_gradient_magnitudes=np.array([1e7, 1.0, 1e-1]),
        variance_gradient_magnitudes=np.array([1e6, 0.1, 0.01]),
        max_imbalance_ratio=1.2e7,
        has_numerical_issues=False,
        vanishing_detected=False,
        imbalance_detected=True,
        stagnation_detected=False,
        issues=[issue],
        health_status=HealthStatus.WARNING,
        computation_time_ms=2.0,
    )


@pytest.fixture
def critical_gradient_report() -> GradientHealthReport:
    """Create a gradient health report with CRITICAL issues (vanishing gradients)."""
    issue = ModelHealthIssue(
        category=IssueCategory.GRADIENT,
        severity=IssueSeverity.CRITICAL,
        code="GRAD-001",
        message="Vanishing gradients detected while cost function remains high",
        affected_parameters=None,
        details={"final_gradient_norm": 1e-12, "cost": 10.0},
        recommendation="Consider rescaling parameters or trying different initial guesses",
    )
    return GradientHealthReport(
        n_iterations=100,
        health_score=0.2,
        mean_gradient_norm=1e-10,
        final_gradient_norm=1e-12,
        mean_gradient_magnitudes=np.array([1e-12, 1e-11, 1e-12]),
        variance_gradient_magnitudes=np.array([1e-24, 1e-22, 1e-24]),
        max_imbalance_ratio=10.0,
        has_numerical_issues=False,
        vanishing_detected=True,
        imbalance_detected=False,
        stagnation_detected=False,
        issues=[issue],
        health_status=HealthStatus.CRITICAL,
        computation_time_ms=2.0,
    )


@pytest.fixture
def unavailable_identifiability_report() -> IdentifiabilityReport:
    """Create an unavailable identifiability report (SVD failed)."""
    return IdentifiabilityReport(
        available=False,
        error_message="SVD computation failed due to numerical issues",
        condition_number=float("inf"),
        numerical_rank=0,
        n_params=3,
        correlation_matrix=None,
        highly_correlated_pairs=[],
        issues=[],
        health_status=HealthStatus.HEALTHY,
        computation_time_ms=1.0,
    )


@pytest.fixture
def unavailable_gradient_report() -> GradientHealthReport:
    """Create an unavailable gradient health report."""
    return GradientHealthReport(
        available=False,
        error_message="No gradient information available",
        computation_time_ms=0.0,
    )


@pytest.fixture
def default_config() -> DiagnosticsConfig:
    """Create default diagnostics configuration."""
    return DiagnosticsConfig()


@pytest.fixture
def full_level_config() -> DiagnosticsConfig:
    """Create diagnostics configuration with FULL level."""
    return DiagnosticsConfig(level=DiagnosticLevel.FULL)


# =============================================================================
# T027: ModelHealthReport Aggregation Tests
# =============================================================================


@pytest.mark.diagnostics
class TestHealthReportStatusDetermination:
    """Tests for overall status determination (Contract B1)."""

    def test_status_healthy_when_no_issues(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test status is HEALTHY when all components report no issues."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        assert report.status.name == "HEALTHY"
        assert len(report.all_issues) == 0

    def test_status_warning_when_warning_issues_exist(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test status is WARNING when warning issues exist but no critical."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        assert report.status.name == "WARNING"
        assert any(
            issue.severity == IssueSeverity.WARNING for issue in report.all_issues
        )

    def test_status_critical_when_critical_issues_exist(
        self,
        critical_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test status is CRITICAL when any critical issue exists."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=critical_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        assert report.status.name == "CRITICAL"
        assert any(
            issue.severity == IssueSeverity.CRITICAL for issue in report.all_issues
        )

    def test_status_critical_overrides_warning(
        self,
        critical_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test CRITICAL status overrides WARNING when both exist."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=critical_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        assert report.status.name == "CRITICAL"
        # Should have both critical and warning issues
        severities = {issue.severity for issue in report.all_issues}
        assert IssueSeverity.CRITICAL in severities
        assert IssueSeverity.WARNING in severities

    def test_status_warning_from_gradient_only(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test WARNING status when only gradient issues exist."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        assert report.status.name == "WARNING"

    def test_status_critical_from_gradient_only(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        critical_gradient_report: GradientHealthReport,
    ) -> None:
        """Test CRITICAL status when only gradient issues exist."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=critical_gradient_report,
        )

        assert report.status.name == "CRITICAL"


@pytest.mark.diagnostics
class TestHealthScoreComputation:
    """Tests for health score computation (Contract B2)."""

    def test_health_score_perfect_when_healthy(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test health score is ~1.0 when all components healthy."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        # Should be very close to 1.0
        assert report.health_score >= 0.9
        assert report.health_score <= 1.0

    def test_health_score_reduced_by_structural_unidentifiability(
        self,
        critical_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test health score reduced by -0.4 for structural unidentifiability.

        Per contract B2:
        - Base: 1.0
        - After structural deduction: 1.0 - 0.4 = 0.6
        - Gradient contribution (0.3 weight): score = score * 0.7 + gradient * 0.3
        - With gradient = 1.0: 0.6 * 0.7 + 1.0 * 0.3 = 0.42 + 0.30 = 0.72
        """
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=critical_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        # Score should be around 0.72 (0.6 * 0.7 + 1.0 * 0.3)
        assert report.health_score <= 0.75
        assert report.health_score >= 0.65

    def test_health_score_reduced_by_practical_unidentifiability(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test health score reduced by -0.2 for practical unidentifiability.

        Per contract B2:
        - Base: 1.0
        - After practical deduction: 1.0 - 0.2 = 0.8
        - Gradient contribution (0.3 weight): score = score * 0.7 + gradient * 0.3
        - With gradient = 1.0: 0.8 * 0.7 + 1.0 * 0.3 = 0.56 + 0.30 = 0.86
        """
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        # Score should be around 0.86 (0.8 * 0.7 + 1.0 * 0.3)
        assert report.health_score <= 0.90
        assert report.health_score >= 0.80

    def test_health_score_reduced_by_correlated_pairs(
        self,
        correlation_warning_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test health score reduced by correlated pairs (-0.05 each, max -0.2)."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=correlation_warning_report,
            gradient_health=healthy_gradient_report,
        )

        # One correlated pair: 1.0 - 0.05 = 0.95 max
        assert report.health_score <= 0.98
        assert report.health_score >= 0.7

    def test_health_score_includes_gradient_weight(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test gradient health contributes with 0.3 weight."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        # warning_gradient_report has health_score=0.6
        # Contribution: 0.6 * 0.3 = 0.18, vs perfect 0.3
        # So overall should be reduced
        assert report.health_score < 1.0
        assert report.health_score >= 0.5

    def test_health_score_clamped_to_zero(
        self,
        critical_identifiability_report: IdentifiabilityReport,
        critical_gradient_report: GradientHealthReport,
    ) -> None:
        """Test health score is clamped to minimum 0.0."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=critical_identifiability_report,
            gradient_health=critical_gradient_report,
        )

        assert report.health_score >= 0.0

    def test_health_score_clamped_to_one(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test health score is clamped to maximum 1.0."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        assert report.health_score <= 1.0

    def test_health_score_range(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test health score always in [0, 1] range."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        assert 0.0 <= report.health_score <= 1.0


@pytest.mark.diagnostics
class TestIssueAggregation:
    """Tests for issue aggregation and sorting (Contract B3)."""

    def test_issues_aggregated_from_all_components(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test issues from all components are aggregated."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        # Should have issues from both components
        codes = {issue.code for issue in report.all_issues}
        assert "IDENT-002" in codes  # From identifiability
        assert "GRAD-002" in codes  # From gradient

    def test_issues_sorted_by_severity_critical_first(
        self,
        critical_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test issues are sorted with CRITICAL first."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=critical_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        # First issue should be CRITICAL
        assert report.all_issues[0].severity == IssueSeverity.CRITICAL
        # Last issue should be WARNING
        assert report.all_issues[-1].severity == IssueSeverity.WARNING

    def test_issues_sorted_by_code_within_severity(
        self,
        correlation_warning_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test issues within same severity are sorted by code."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=correlation_warning_report,
            gradient_health=warning_gradient_report,
        )

        # Both are WARNING, should be sorted by code
        warning_issues = [
            i for i in report.all_issues if i.severity == IssueSeverity.WARNING
        ]
        if len(warning_issues) >= 2:
            codes = [i.code for i in warning_issues]
            assert codes == sorted(codes)

    def test_issues_empty_when_all_healthy(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test all_issues is empty when no issues detected."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        assert len(report.all_issues) == 0

    def test_issues_from_multiple_identifiability_problems(self) -> None:
        """Test multiple issues from same component are aggregated."""
        from nlsq.diagnostics.health_report import create_health_report

        # Create report with both structural and correlation issues
        issue1 = ModelHealthIssue(
            category=IssueCategory.IDENTIFIABILITY,
            severity=IssueSeverity.CRITICAL,
            code="IDENT-001",
            message="Structural unidentifiability",
            affected_parameters=(0, 1),
            details={},
            recommendation="Reparameterize",
        )
        issue2 = ModelHealthIssue(
            category=IssueCategory.CORRELATION,
            severity=IssueSeverity.WARNING,
            code="CORR-001",
            message="High correlation",
            affected_parameters=(0, 1),
            details={},
            recommendation="Combine parameters",
        )

        ident_report = IdentifiabilityReport(
            condition_number=float("inf"),
            numerical_rank=2,
            n_params=3,
            correlation_matrix=None,
            highly_correlated_pairs=[(0, 1, 0.99)],
            issues=[issue1, issue2],
            health_status=HealthStatus.CRITICAL,
        )

        report = create_health_report(identifiability=ident_report)

        assert len(report.all_issues) == 2
        codes = {i.code for i in report.all_issues}
        assert "IDENT-001" in codes
        assert "CORR-001" in codes


@pytest.mark.diagnostics
class TestPartialComponentAvailability:
    """Tests for partial component availability (Error Handling)."""

    def test_identifiability_only(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
    ) -> None:
        """Test report creation with only identifiability component."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=None,
        )

        assert report.status.name == "HEALTHY"
        assert report.identifiability is not None
        assert report.gradient_health is None

    def test_gradient_health_only(
        self,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test report creation with only gradient health component."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=None,
            gradient_health=healthy_gradient_report,
        )

        assert report.status.name == "HEALTHY"
        assert report.identifiability is None
        assert report.gradient_health is not None

    def test_unavailable_identifiability_report(
        self,
        unavailable_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test handling of unavailable identifiability report."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=unavailable_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        # Should still create report with available component
        assert report.identifiability is not None
        assert report.identifiability.available is False
        assert report.gradient_health is not None

    def test_unavailable_gradient_report(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        unavailable_gradient_report: GradientHealthReport,
    ) -> None:
        """Test handling of unavailable gradient report."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=unavailable_gradient_report,
        )

        # Should still create report
        assert report.identifiability is not None
        assert report.gradient_health is not None
        assert report.gradient_health.available is False

    def test_all_components_unavailable(
        self,
        unavailable_identifiability_report: IdentifiabilityReport,
        unavailable_gradient_report: GradientHealthReport,
    ) -> None:
        """Test report when all components unavailable returns WARNING with note."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=unavailable_identifiability_report,
            gradient_health=unavailable_gradient_report,
        )

        # Per contract: all unavailable returns WARNING status
        assert report.status.name == "WARNING"

    def test_all_components_none(self) -> None:
        """Test report when all components are None returns WARNING."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=None,
            gradient_health=None,
        )

        # Per contract: all unavailable returns WARNING with note
        assert report.status.name == "WARNING"


# =============================================================================
# T028: summary() Output Formatting Tests
# =============================================================================


@pytest.mark.diagnostics
class TestSummaryFormatHealthy:
    """Tests for summary() format with healthy model (Contract B4)."""

    def test_summary_contains_header(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary contains proper header with separator lines."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        assert "=" * 50 in summary or "=" * 70 in summary
        assert "Model Health Report" in summary

    def test_summary_contains_status(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary contains status line."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        assert "Status:" in summary
        assert "HEALTHY" in summary

    def test_summary_contains_health_score(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary contains health score."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        assert "Health Score:" in summary
        # Score should be formatted as decimal (e.g., 0.95)
        assert any(c.isdigit() for c in summary)

    def test_summary_contains_identifiability_section(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary contains identifiability section."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        assert "Identifiability" in summary
        assert (
            "Structurally identifiable" in summary or "identifiable" in summary.lower()
        )

    def test_summary_contains_gradient_section(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary contains gradient health section."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        assert "Gradient" in summary

    def test_summary_healthy_shows_no_issues(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test healthy summary does not show issues section."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        # Should not have an Issues section with count
        assert "Issues (0)" not in summary or "No issues" in summary

    def test_summary_healthy_no_recommendations(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test healthy summary does not show recommendations section."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        # Recommendations section should not appear for healthy model
        assert "Recommendations" not in summary or len(report.all_issues) > 0


@pytest.mark.diagnostics
class TestSummaryFormatWithIssues:
    """Tests for summary() format with issues present (Contract B5)."""

    def test_summary_shows_warning_status(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary shows WARNING status."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        assert "WARNING" in summary

    def test_summary_shows_critical_status(
        self,
        critical_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary shows CRITICAL status."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=critical_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary()

        assert "CRITICAL" in summary

    def test_summary_shows_issues_section_with_count(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary shows Issues section with count."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        summary = report.summary()

        # Should have Issues section with count
        assert "Issues" in summary
        assert "(2)" in summary or "2" in summary

    def test_summary_shows_issue_codes(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary shows issue codes."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        summary = report.summary()

        assert "IDENT-002" in summary
        assert "GRAD-002" in summary

    def test_summary_shows_severity_tags(
        self,
        critical_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary shows severity tags like [WARNING] and [CRITICAL]."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=critical_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        summary = report.summary(verbose=True)

        assert "[CRITICAL]" in summary
        assert "[WARNING]" in summary

    def test_summary_shows_recommendations_section(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary shows recommendations section when issues exist."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        summary = report.summary(verbose=True)

        assert "Recommendations" in summary

    def test_summary_verbose_includes_descriptions(
        self,
        correlation_warning_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test verbose summary includes issue descriptions."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=correlation_warning_report,
            gradient_health=warning_gradient_report,
        )

        summary = report.summary(verbose=True)

        # Should include issue messages
        assert "correlated" in summary.lower() or "imbalance" in summary.lower()


@pytest.mark.diagnostics
class TestSummaryVerboseVsNonVerbose:
    """Tests for verbose vs non-verbose summary output."""

    def test_verbose_true_includes_recommendations(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test verbose=True includes recommendations."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        verbose_summary = report.summary(verbose=True)

        # Should have recommendations or suggestion text
        assert "Consider" in verbose_summary or "Recommendations" in verbose_summary

    def test_verbose_false_shorter_output(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        warning_gradient_report: GradientHealthReport,
    ) -> None:
        """Test verbose=False produces shorter output."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=warning_gradient_report,
        )

        verbose_summary = report.summary(verbose=True)
        non_verbose_summary = report.summary(verbose=False)

        # Non-verbose should be shorter or equal length
        assert len(non_verbose_summary) <= len(verbose_summary)

    def test_non_verbose_still_shows_status(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test non-verbose summary still shows status and score."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary(verbose=False)

        assert "Status:" in summary or "WARNING" in summary
        assert "Score" in summary or "score" in summary.lower()

    def test_non_verbose_shows_issue_codes(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test non-verbose summary still shows issue codes."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        summary = report.summary(verbose=False)

        # Should still show issue codes
        assert "IDENT-002" in summary


@pytest.mark.diagnostics
class TestSummaryDefault:
    """Tests for default summary() behavior."""

    def test_summary_default_is_verbose(
        self,
        warning_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test summary() with no args defaults to verbose=True."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=warning_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        default_summary = report.summary()
        verbose_summary = report.summary(verbose=True)

        # Default should be same as verbose=True
        assert default_summary == verbose_summary


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.diagnostics
class TestHealthReportEdgeCases:
    """Tests for edge cases in health report creation."""

    def test_empty_report(self) -> None:
        """Test creating report with no components."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report()

        assert report is not None
        # Use .name comparison to avoid enum identity issues across xdist workers
        assert report.status.name == "WARNING"
        # Summary should still work
        summary = report.summary()
        assert len(summary) > 0

    def test_report_with_config(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
        default_config: DiagnosticsConfig,
    ) -> None:
        """Test report creation with config parameter."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
            config=default_config,
        )

        assert report is not None
        # Use .name comparison to avoid enum identity issues across xdist workers
        assert report.status.name == "HEALTHY"

    def test_report_preserves_component_references(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test report preserves references to component reports."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        assert report.identifiability is healthy_identifiability_report
        assert report.gradient_health is healthy_gradient_report

    def test_many_issues_sorted_correctly(self) -> None:
        """Test sorting with many issues from multiple sources."""
        from nlsq.diagnostics.health_report import create_health_report

        # Create multiple issues with different severities
        issues_ident = [
            ModelHealthIssue(
                category=IssueCategory.IDENTIFIABILITY,
                severity=IssueSeverity.CRITICAL,
                code="IDENT-001",
                message="Critical ident issue",
                affected_parameters=None,
                details={},
                recommendation="Fix it",
            ),
            ModelHealthIssue(
                category=IssueCategory.CORRELATION,
                severity=IssueSeverity.WARNING,
                code="CORR-001",
                message="Warning corr issue",
                affected_parameters=(0, 1),
                details={},
                recommendation="Fix it",
            ),
            ModelHealthIssue(
                category=IssueCategory.IDENTIFIABILITY,
                severity=IssueSeverity.INFO,
                code="IDENT-INFO",
                message="Info ident issue",
                affected_parameters=None,
                details={},
                recommendation="Note it",
            ),
        ]

        issues_grad = [
            ModelHealthIssue(
                category=IssueCategory.GRADIENT,
                severity=IssueSeverity.WARNING,
                code="GRAD-002",
                message="Warning grad issue",
                affected_parameters=None,
                details={},
                recommendation="Fix it",
            ),
            ModelHealthIssue(
                category=IssueCategory.GRADIENT,
                severity=IssueSeverity.CRITICAL,
                code="GRAD-001",
                message="Critical grad issue",
                affected_parameters=None,
                details={},
                recommendation="Fix it",
            ),
        ]

        ident_report = IdentifiabilityReport(
            condition_number=float("inf"),
            numerical_rank=2,
            n_params=3,
            correlation_matrix=None,
            highly_correlated_pairs=[(0, 1, 0.99)],
            issues=issues_ident,
            health_status=HealthStatus.CRITICAL,
        )

        grad_report = GradientHealthReport(
            n_iterations=100,
            health_score=0.3,
            mean_gradient_norm=1e-10,
            final_gradient_norm=1e-12,
            mean_gradient_magnitudes=np.array([1e-10, 1e-10, 1e-10]),
            variance_gradient_magnitudes=np.array([1e-20, 1e-20, 1e-20]),
            max_imbalance_ratio=1.0,
            has_numerical_issues=False,
            vanishing_detected=True,
            imbalance_detected=False,
            stagnation_detected=False,
            issues=issues_grad,
            health_status=HealthStatus.CRITICAL,
        )

        report = create_health_report(
            identifiability=ident_report,
            gradient_health=grad_report,
        )

        # Should have all 5 issues
        assert len(report.all_issues) == 5

        # Check sorted by severity (CRITICAL > WARNING > INFO)
        severities = [i.severity for i in report.all_issues]

        # All CRITICAL should come before WARNING
        critical_indices = [
            idx for idx, s in enumerate(severities) if s == IssueSeverity.CRITICAL
        ]
        warning_indices = [
            idx for idx, s in enumerate(severities) if s == IssueSeverity.WARNING
        ]
        info_indices = [
            idx for idx, s in enumerate(severities) if s == IssueSeverity.INFO
        ]

        if critical_indices and warning_indices:
            assert max(critical_indices) < min(warning_indices)
        if warning_indices and info_indices:
            assert max(warning_indices) < min(info_indices)

    def test_summary_str_method(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test __str__ method returns summary."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        str_output = str(report)
        summary_output = report.summary()

        # str() should return same as summary()
        assert str_output == summary_output


@pytest.mark.diagnostics
class TestModelHealthReportDataclass:
    """Tests for ModelHealthReport dataclass structure."""

    def test_report_has_required_attributes(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test ModelHealthReport has all required attributes."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        # Required attributes per contract
        assert hasattr(report, "status")
        assert hasattr(report, "health_score")
        assert hasattr(report, "all_issues")
        assert hasattr(report, "identifiability")
        assert hasattr(report, "gradient_health")
        assert hasattr(report, "summary")

    def test_report_status_is_health_status(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test status attribute is HealthStatus enum."""
        from nlsq.diagnostics.health_report import create_health_report
        from nlsq.diagnostics.plugin import PluginRegistry

        # Clear plugin registry for isolation
        PluginRegistry.clear()

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        # Use .name comparison to avoid enum identity issues across module reloads
        assert report.status.name in ("HEALTHY", "WARNING", "CRITICAL")

    def test_report_all_issues_is_list(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test all_issues attribute is a list."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        assert isinstance(report.all_issues, list)

    def test_report_health_score_is_float(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
        healthy_gradient_report: GradientHealthReport,
    ) -> None:
        """Test health_score attribute is a float."""
        from nlsq.diagnostics.health_report import create_health_report

        report = create_health_report(
            identifiability=healthy_identifiability_report,
            gradient_health=healthy_gradient_report,
        )

        assert isinstance(report.health_score, float)


@pytest.mark.diagnostics
class TestCreateHealthReportFunction:
    """Tests for create_health_report() function signature and behavior."""

    def test_function_accepts_all_optional_parameters(self) -> None:
        """Test function accepts all parameters as optional."""
        from nlsq.diagnostics.health_report import create_health_report

        # Should not raise
        report = create_health_report(
            identifiability=None,
            gradient_health=None,
            sloppy_model=None,
            plugin_results=None,
            config=None,
        )

        assert report is not None

    def test_function_returns_model_health_report(
        self,
        healthy_identifiability_report: IdentifiabilityReport,
    ) -> None:
        """Test function returns ModelHealthReport instance."""
        from nlsq.diagnostics.health_report import create_health_report
        from nlsq.diagnostics.types import ModelHealthReport

        report = create_health_report(
            identifiability=healthy_identifiability_report,
        )

        assert isinstance(report, ModelHealthReport)
