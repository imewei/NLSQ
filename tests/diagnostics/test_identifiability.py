"""Unit tests for IdentifiabilityAnalyzer.

Tests cover:
- FIM computation from Jacobian (J.T @ J)
- SVD for condition number and rank
- Correlation matrix extraction
- Highly correlated pairs detection
- Issue generation (IDENT-001, IDENT-002, CORR-001)
- Graceful degradation for SVD failures
"""

import numpy as np
import pytest

from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer
from nlsq.diagnostics.types import (
    DiagnosticsConfig,
    HealthStatus,
    IdentifiabilityReport,
    IssueCategory,
    IssueSeverity,
)


class TestIdentifiabilityReport:
    """Tests for IdentifiabilityReport dataclass."""

    def test_identifiability_report_defaults(self) -> None:
        """Test IdentifiabilityReport with minimal required fields."""
        report = IdentifiabilityReport(
            condition_number=1.0,
            numerical_rank=3,
            n_params=3,
            correlation_matrix=np.eye(3),
            highly_correlated_pairs=[],
            issues=[],
            health_status=HealthStatus.HEALTHY,
        )
        assert report.available is True
        assert report.error_message is None
        assert report.condition_number == 1.0
        assert report.numerical_rank == 3
        assert report.n_params == 3
        assert len(report.issues) == 0

    def test_identifiability_report_with_issues(self) -> None:
        """Test IdentifiabilityReport with issues detected."""
        from nlsq.diagnostics.types import ModelHealthIssue

        issue = ModelHealthIssue(
            category=IssueCategory.IDENTIFIABILITY,
            severity=IssueSeverity.CRITICAL,
            code="IDENT-001",
            message="Structural unidentifiability",
            affected_parameters=(0, 1),
            details={"numerical_rank": 2, "n_params": 3},
            recommendation="Consider reparameterizing",
        )
        report = IdentifiabilityReport(
            condition_number=1e12,
            numerical_rank=2,
            n_params=3,
            correlation_matrix=np.eye(3),
            highly_correlated_pairs=[],
            issues=[issue],
            health_status=HealthStatus.CRITICAL,
        )
        assert len(report.issues) == 1
        assert report.issues[0].code == "IDENT-001"
        assert report.health_status == HealthStatus.CRITICAL

    def test_identifiability_report_unavailable(self) -> None:
        """Test IdentifiabilityReport when analysis failed."""
        report = IdentifiabilityReport(
            available=False,
            error_message="SVD computation failed",
            condition_number=float("inf"),
            numerical_rank=0,
            n_params=3,
            correlation_matrix=None,
            highly_correlated_pairs=[],
            issues=[],
            health_status=HealthStatus.CRITICAL,
        )
        assert report.available is False
        assert report.error_message == "SVD computation failed"

    def test_identifiability_report_correlated_pairs(self) -> None:
        """Test IdentifiabilityReport with correlated pairs."""
        report = IdentifiabilityReport(
            condition_number=100.0,
            numerical_rank=3,
            n_params=3,
            correlation_matrix=np.array(
                [
                    [1.0, 0.98, 0.1],
                    [0.98, 1.0, 0.2],
                    [0.1, 0.2, 1.0],
                ]
            ),
            highly_correlated_pairs=[(0, 1, 0.98)],
            issues=[],
            health_status=HealthStatus.WARNING,
        )
        assert len(report.highly_correlated_pairs) == 1
        assert report.highly_correlated_pairs[0] == (0, 1, 0.98)


class TestIdentifiabilityAnalyzer:
    """Tests for IdentifiabilityAnalyzer class."""

    @pytest.fixture
    def default_config(self) -> DiagnosticsConfig:
        """Create default diagnostics configuration."""
        return DiagnosticsConfig()

    @pytest.fixture
    def analyzer(self, default_config: DiagnosticsConfig) -> IdentifiabilityAnalyzer:
        """Create IdentifiabilityAnalyzer with default configuration."""
        return IdentifiabilityAnalyzer(config=default_config)

    @pytest.fixture
    def well_conditioned_jacobian(self) -> np.ndarray:
        """Create a well-conditioned Jacobian matrix."""
        # 100 data points, 3 parameters
        np.random.seed(42)
        J = np.random.randn(100, 3)
        # Make it well-conditioned by adding scaled identity-like structure
        J[:3, :] += 10 * np.eye(3)
        return J

    @pytest.fixture
    def ill_conditioned_jacobian(self) -> np.ndarray:
        """Create an ill-conditioned Jacobian matrix (practical unidentifiability)."""
        # Create a Jacobian with very different singular values
        np.random.seed(42)
        n, p = 100, 3
        U = np.linalg.qr(np.random.randn(n, p))[0]
        # Singular values spanning many orders of magnitude
        s = np.array([1e10, 1.0, 1e-5])
        V = np.linalg.qr(np.random.randn(p, p))[0]
        return U @ np.diag(s) @ V.T

    @pytest.fixture
    def rank_deficient_jacobian(self) -> np.ndarray:
        """Create a rank-deficient Jacobian (structural unidentifiability)."""
        np.random.seed(42)
        # Create a 100x3 matrix with rank 2
        J = np.zeros((100, 3))
        J[:, 0] = np.random.randn(100)
        J[:, 1] = np.random.randn(100)
        # Third column is a linear combination of first two
        J[:, 2] = 0.5 * J[:, 0] + 0.5 * J[:, 1]
        return J

    @pytest.fixture
    def correlated_jacobian(self) -> np.ndarray:
        """Create a Jacobian with highly correlated columns."""
        np.random.seed(42)
        n, p = 100, 3
        base = np.random.randn(n)
        J = np.zeros((n, p))
        J[:, 0] = base + 0.01 * np.random.randn(n)  # Very similar columns
        J[:, 1] = base + 0.02 * np.random.randn(n)
        J[:, 2] = np.random.randn(n)  # Independent column
        return J

    # Test FIM computation
    def test_compute_fim(self, analyzer: IdentifiabilityAnalyzer) -> None:
        """Test Fisher Information Matrix computation."""
        J = np.array([[1, 2], [3, 4], [5, 6]])
        fim = analyzer._compute_fim(J)
        expected = J.T @ J
        np.testing.assert_array_almost_equal(fim, expected)

    def test_compute_fim_shape(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test FIM has correct shape (n_params x n_params)."""
        fim = analyzer._compute_fim(well_conditioned_jacobian)
        n_params = well_conditioned_jacobian.shape[1]
        assert fim.shape == (n_params, n_params)

    def test_compute_fim_symmetric(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test FIM is symmetric."""
        fim = analyzer._compute_fim(well_conditioned_jacobian)
        np.testing.assert_array_almost_equal(fim, fim.T)

    # Test condition number and rank
    def test_condition_number_well_conditioned(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test condition number for well-conditioned Jacobian."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.available is True
        assert report.condition_number < 1e8  # Below threshold
        assert report.numerical_rank == well_conditioned_jacobian.shape[1]

    def test_condition_number_ill_conditioned(
        self, analyzer: IdentifiabilityAnalyzer, ill_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test condition number for ill-conditioned Jacobian."""
        report = analyzer.analyze(ill_conditioned_jacobian)
        assert report.available is True
        assert report.condition_number > 1e8  # Above threshold

    def test_numerical_rank_full_rank(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test numerical rank for full-rank Jacobian."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.numerical_rank == well_conditioned_jacobian.shape[1]

    def test_numerical_rank_deficient(
        self, analyzer: IdentifiabilityAnalyzer, rank_deficient_jacobian: np.ndarray
    ) -> None:
        """Test numerical rank for rank-deficient Jacobian."""
        report = analyzer.analyze(rank_deficient_jacobian)
        assert report.numerical_rank < rank_deficient_jacobian.shape[1]

    # Test correlation matrix
    def test_correlation_matrix_diagonal(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test correlation matrix has ones on diagonal."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.correlation_matrix is not None
        np.testing.assert_array_almost_equal(
            np.diag(report.correlation_matrix), np.ones(report.n_params)
        )

    def test_correlation_matrix_range(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test correlation values are in [-1, 1]."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.correlation_matrix is not None
        assert np.all(report.correlation_matrix >= -1.0 - 1e-10)
        assert np.all(report.correlation_matrix <= 1.0 + 1e-10)

    def test_correlation_matrix_symmetric(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test correlation matrix is symmetric."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.correlation_matrix is not None
        np.testing.assert_array_almost_equal(
            report.correlation_matrix, report.correlation_matrix.T
        )

    # Test highly correlated pairs detection
    def test_detect_highly_correlated_pairs(
        self, analyzer: IdentifiabilityAnalyzer, correlated_jacobian: np.ndarray
    ) -> None:
        """Test detection of highly correlated parameter pairs."""
        report = analyzer.analyze(correlated_jacobian)
        # Should detect correlation between columns 0 and 1
        corr_pairs = report.highly_correlated_pairs
        assert len(corr_pairs) >= 1
        # Check that (0, 1) pair is in the list
        pair_indices = [(p[0], p[1]) for p in corr_pairs]
        assert (0, 1) in pair_indices

    def test_no_highly_correlated_pairs_orthogonal(
        self, analyzer: IdentifiabilityAnalyzer
    ) -> None:
        """Test no correlated pairs for orthogonal Jacobian."""
        # Orthogonal columns
        J = np.eye(3)
        J = np.vstack([J, J, J])  # 9x3 with orthogonal columns
        report = analyzer.analyze(J)
        assert len(report.highly_correlated_pairs) == 0

    def test_correlation_threshold_respected(self) -> None:
        """Test custom correlation threshold is respected."""
        config = DiagnosticsConfig(correlation_threshold=0.5)
        analyzer = IdentifiabilityAnalyzer(config=config)

        # Create Jacobian with moderate correlation (0.7)
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        J = np.zeros((n, 3))
        J[:, 0] = base
        J[:, 1] = 0.7 * base + 0.3 * np.random.randn(n)  # ~0.7 correlation
        J[:, 2] = np.random.randn(n)

        report = analyzer.analyze(J)
        # With threshold 0.5, should detect correlation
        assert len(report.highly_correlated_pairs) >= 1

    # Test issue generation
    def test_issue_ident_001_structural(
        self, analyzer: IdentifiabilityAnalyzer, rank_deficient_jacobian: np.ndarray
    ) -> None:
        """Test IDENT-001 issue is generated for structural unidentifiability."""
        report = analyzer.analyze(rank_deficient_jacobian)
        issue_codes = [issue.code for issue in report.issues]
        assert "IDENT-001" in issue_codes

    def test_issue_ident_002_practical(
        self, analyzer: IdentifiabilityAnalyzer, ill_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test IDENT-002 issue is generated for practical unidentifiability."""
        report = analyzer.analyze(ill_conditioned_jacobian)
        issue_codes = [issue.code for issue in report.issues]
        assert "IDENT-002" in issue_codes

    def test_issue_corr_001_correlation(
        self, analyzer: IdentifiabilityAnalyzer, correlated_jacobian: np.ndarray
    ) -> None:
        """Test CORR-001 issue is generated for highly correlated parameters."""
        report = analyzer.analyze(correlated_jacobian)
        issue_codes = [issue.code for issue in report.issues]
        assert "CORR-001" in issue_codes

    def test_no_issues_healthy_jacobian(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test no issues for healthy Jacobian."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert len(report.issues) == 0
        assert report.health_status == HealthStatus.HEALTHY

    def test_issue_severity_structural(
        self, analyzer: IdentifiabilityAnalyzer, rank_deficient_jacobian: np.ndarray
    ) -> None:
        """Test IDENT-001 has CRITICAL severity."""
        report = analyzer.analyze(rank_deficient_jacobian)
        ident_001 = next(i for i in report.issues if i.code == "IDENT-001")
        assert ident_001.severity == IssueSeverity.CRITICAL

    def test_issue_severity_practical(
        self, analyzer: IdentifiabilityAnalyzer, ill_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test IDENT-002 has WARNING severity."""
        report = analyzer.analyze(ill_conditioned_jacobian)
        ident_002 = next(i for i in report.issues if i.code == "IDENT-002")
        assert ident_002.severity == IssueSeverity.WARNING

    def test_issue_affected_parameters(
        self, analyzer: IdentifiabilityAnalyzer, correlated_jacobian: np.ndarray
    ) -> None:
        """Test affected_parameters is set correctly."""
        report = analyzer.analyze(correlated_jacobian)
        corr_issue = next(i for i in report.issues if i.code == "CORR-001")
        assert corr_issue.affected_parameters is not None
        assert 0 in corr_issue.affected_parameters
        assert 1 in corr_issue.affected_parameters

    # Test health status
    def test_health_status_healthy(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test HEALTHY status for well-conditioned Jacobian."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.health_status == HealthStatus.HEALTHY

    def test_health_status_critical_structural(
        self, analyzer: IdentifiabilityAnalyzer, rank_deficient_jacobian: np.ndarray
    ) -> None:
        """Test CRITICAL status for structural unidentifiability."""
        report = analyzer.analyze(rank_deficient_jacobian)
        assert report.health_status == HealthStatus.CRITICAL

    def test_health_status_warning_practical(
        self, analyzer: IdentifiabilityAnalyzer, ill_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test WARNING status for practical unidentifiability."""
        report = analyzer.analyze(ill_conditioned_jacobian)
        # Could be WARNING or CRITICAL depending on severity
        assert report.health_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]

    # Test graceful degradation
    def test_graceful_degradation_empty_jacobian(
        self, analyzer: IdentifiabilityAnalyzer
    ) -> None:
        """Test graceful degradation for empty Jacobian."""
        J = np.array([]).reshape(0, 3)
        report = analyzer.analyze(J)
        assert report.available is False
        assert report.error_message is not None

    def test_graceful_degradation_nan_jacobian(
        self, analyzer: IdentifiabilityAnalyzer
    ) -> None:
        """Test graceful degradation for Jacobian with NaN."""
        J = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        report = analyzer.analyze(J)
        assert report.available is False
        assert "NaN" in report.error_message or "nan" in report.error_message.lower()

    def test_graceful_degradation_inf_jacobian(
        self, analyzer: IdentifiabilityAnalyzer
    ) -> None:
        """Test graceful degradation for Jacobian with Inf."""
        J = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
        report = analyzer.analyze(J)
        assert report.available is False
        assert "Inf" in report.error_message or "inf" in report.error_message.lower()

    def test_graceful_degradation_zero_jacobian(
        self, analyzer: IdentifiabilityAnalyzer
    ) -> None:
        """Test graceful degradation for all-zero Jacobian."""
        J = np.zeros((10, 3))
        report = analyzer.analyze(J)
        # Should still produce a result, but with issues
        assert report.available is True
        # Condition number should be infinite
        assert report.condition_number == float("inf") or report.condition_number > 1e15

    # Test computation time tracking
    def test_computation_time_tracked(
        self, analyzer: IdentifiabilityAnalyzer, well_conditioned_jacobian: np.ndarray
    ) -> None:
        """Test computation time is tracked."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.computation_time_ms >= 0.0

    # Test custom thresholds
    def test_custom_condition_threshold(self) -> None:
        """Test custom condition number threshold."""
        config = DiagnosticsConfig(condition_threshold=100.0)
        analyzer = IdentifiabilityAnalyzer(config=config)

        # Create moderately ill-conditioned Jacobian
        np.random.seed(42)
        n, p = 100, 3
        U = np.linalg.qr(np.random.randn(n, p))[0]
        s = np.array([1000.0, 10.0, 1.0])  # Condition number ~1000
        V = np.linalg.qr(np.random.randn(p, p))[0]
        J = U @ np.diag(s) @ V.T

        report = analyzer.analyze(J)
        # With threshold 100, condition number 1000 should trigger IDENT-002
        issue_codes = [issue.code for issue in report.issues]
        assert "IDENT-002" in issue_codes

    # Test analyze_from_fim
    def test_analyze_from_fim(self, analyzer: IdentifiabilityAnalyzer) -> None:
        """Test analysis from pre-computed FIM."""
        # Well-conditioned FIM
        fim = np.array([[10.0, 1.0, 0.5], [1.0, 8.0, 0.3], [0.5, 0.3, 6.0]])
        report = analyzer.analyze_from_fim(fim)
        assert report.available is True
        assert report.n_params == 3

    def test_analyze_from_fim_singular(self, analyzer: IdentifiabilityAnalyzer) -> None:
        """Test analysis from singular FIM."""
        # Singular FIM (rank 2)
        fim = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
        # Make it singular by setting last row/col
        fim[2, :] = fim[0, :] + fim[1, :]
        fim[:, 2] = fim[:, 0] + fim[:, 1]
        report = analyzer.analyze_from_fim(fim)
        # Should detect structural unidentifiability
        issue_codes = [issue.code for issue in report.issues]
        assert "IDENT-001" in issue_codes or report.numerical_rank < 3


class TestIdentifiabilityAnalyzerEdgeCases:
    """Edge case tests for IdentifiabilityAnalyzer."""

    def test_single_parameter(self) -> None:
        """Test analysis with single parameter."""
        config = DiagnosticsConfig()
        analyzer = IdentifiabilityAnalyzer(config=config)
        J = np.random.randn(50, 1)
        report = analyzer.analyze(J)
        assert report.available is True
        assert report.n_params == 1
        assert report.numerical_rank == 1

    def test_many_parameters(self) -> None:
        """Test analysis with many parameters."""
        config = DiagnosticsConfig()
        analyzer = IdentifiabilityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(1000, 20)
        report = analyzer.analyze(J)
        assert report.available is True
        assert report.n_params == 20

    def test_underdetermined_system(self) -> None:
        """Test analysis with more parameters than data points."""
        config = DiagnosticsConfig()
        analyzer = IdentifiabilityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(5, 10)  # 5 data points, 10 parameters
        report = analyzer.analyze(J)
        assert report.available is True
        # Should detect structural unidentifiability
        assert report.numerical_rank < 10

    def test_square_jacobian(self) -> None:
        """Test analysis with square Jacobian."""
        config = DiagnosticsConfig()
        analyzer = IdentifiabilityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(10, 10)
        report = analyzer.analyze(J)
        assert report.available is True
        assert report.n_params == 10

    def test_very_small_values(self) -> None:
        """Test analysis with very small Jacobian values."""
        config = DiagnosticsConfig()
        analyzer = IdentifiabilityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(50, 3) * 1e-15
        report = analyzer.analyze(J)
        # Should handle gracefully
        assert report.available is True

    def test_very_large_values(self) -> None:
        """Test analysis with very large Jacobian values."""
        config = DiagnosticsConfig()
        analyzer = IdentifiabilityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(50, 3) * 1e15
        report = analyzer.analyze(J)
        # Should handle gracefully
        assert report.available is True

    def test_mixed_scale_columns(self) -> None:
        """Test analysis with mixed scale columns."""
        config = DiagnosticsConfig()
        analyzer = IdentifiabilityAnalyzer(config=config)
        np.random.seed(42)
        J = np.zeros((100, 3))
        J[:, 0] = np.random.randn(100) * 1e-10
        J[:, 1] = np.random.randn(100) * 1.0
        J[:, 2] = np.random.randn(100) * 1e10
        report = analyzer.analyze(J)
        # Should detect conditioning issues
        assert report.available is True
        assert report.condition_number > 1e8
