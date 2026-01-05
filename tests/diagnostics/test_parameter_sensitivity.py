"""Unit tests for ParameterSensitivityAnalyzer.

Tests cover:
- Eigenvalue spectrum computation
- Stiff/sloppy direction classification
- Effective dimensionality computation
- get_sloppy_combinations() helper method
- Issue generation (SENS-001, SENS-002)
- Graceful degradation for SVD failures
- Multi-exponential model sensitivity detection
- Known sensitivity patterns from the literature
"""

import warnings

import numpy as np
import pytest

from nlsq.diagnostics.parameter_sensitivity import ParameterSensitivityAnalyzer
from nlsq.diagnostics.types import (
    DiagnosticsConfig,
    HealthStatus,
    IssueCategory,
    IssueSeverity,
    ParameterSensitivityReport,
)


@pytest.mark.diagnostics
class TestParameterSensitivityReport:
    """Tests for ParameterSensitivityReport dataclass."""

    def test_parameter_sensitivity_report_defaults(self) -> None:
        """Test ParameterSensitivityReport with minimal required fields."""
        report = ParameterSensitivityReport(
            is_sloppy=False,
            eigenvalues=np.array([1.0, 0.5, 0.1]),
            eigenvectors=np.eye(3),
            eigenvalue_range=1.0,
            effective_dimensionality=3.0,
            stiff_indices=[0, 1, 2],
            sloppy_indices=[],
            issues=[],
            health_status=HealthStatus.HEALTHY,
        )
        assert report.available is True
        assert report.error_message is None
        assert report.is_sloppy is False
        assert len(report.eigenvalues) == 3
        assert len(report.stiff_indices) == 3
        assert len(report.sloppy_indices) == 0

    def test_parameter_sensitivity_report_wide_spread(self) -> None:
        """Test ParameterSensitivityReport when wide sensitivity spread detected."""
        report = ParameterSensitivityReport(
            is_sloppy=True,
            eigenvalues=np.array([1e6, 1.0, 1e-6]),
            eigenvectors=np.eye(3),
            eigenvalue_range=12.0,  # log10(1e6 / 1e-6)
            effective_dimensionality=1.5,
            stiff_indices=[0],
            sloppy_indices=[2],
            issues=[],
            health_status=HealthStatus.WARNING,
        )
        assert report.is_sloppy is True
        assert report.eigenvalue_range == 12.0
        assert report.effective_dimensionality == 1.5
        assert len(report.stiff_indices) == 1
        assert len(report.sloppy_indices) == 1

    def test_parameter_sensitivity_report_unavailable(self) -> None:
        """Test ParameterSensitivityReport when analysis failed."""
        report = ParameterSensitivityReport(
            available=False,
            error_message="SVD computation failed",
            is_sloppy=False,
            eigenvalues=np.array([]),
            eigenvectors=None,
            eigenvalue_range=0.0,
            effective_dimensionality=0.0,
            stiff_indices=[],
            sloppy_indices=[],
            issues=[],
            health_status=HealthStatus.CRITICAL,
        )
        assert report.available is False
        assert report.error_message == "SVD computation failed"

    def test_get_sloppy_combinations_empty(self) -> None:
        """Test get_sloppy_combinations returns empty for well-conditioned model."""
        report = ParameterSensitivityReport(
            is_sloppy=False,
            eigenvalues=np.array([1.0, 0.5, 0.1]),
            eigenvectors=np.eye(3),
            eigenvalue_range=1.0,
            effective_dimensionality=3.0,
            stiff_indices=[0, 1, 2],
            sloppy_indices=[],
            issues=[],
            health_status=HealthStatus.HEALTHY,
        )
        combinations = report.get_sloppy_combinations()
        assert len(combinations) == 0

    def test_get_sloppy_combinations_returns_eigenvectors(self) -> None:
        """Test get_sloppy_combinations returns correct eigenvector-eigenvalue pairs."""
        eigenvectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        eigenvalues = np.array([1e6, 1.0, 1e-8])
        report = ParameterSensitivityReport(
            is_sloppy=True,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            eigenvalue_range=14.0,
            effective_dimensionality=1.5,
            stiff_indices=[0],
            sloppy_indices=[2],
            issues=[],
            health_status=HealthStatus.WARNING,
        )
        combinations = report.get_sloppy_combinations()
        assert len(combinations) == 1
        eigenvec, eigenval = combinations[0]
        np.testing.assert_array_equal(eigenvec, eigenvectors[:, 2])
        assert eigenval == eigenvalues[2]

    def test_get_sloppy_combinations_multiple_sloppy(self) -> None:
        """Test get_sloppy_combinations with multiple sloppy directions."""
        eigenvectors = np.eye(4)
        eigenvalues = np.array([1e6, 1.0, 1e-6, 1e-8])
        report = ParameterSensitivityReport(
            is_sloppy=True,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            eigenvalue_range=14.0,
            effective_dimensionality=1.5,
            stiff_indices=[0],
            sloppy_indices=[2, 3],
            issues=[],
            health_status=HealthStatus.WARNING,
        )
        combinations = report.get_sloppy_combinations()
        assert len(combinations) == 2
        # Verify correct eigenvectors are returned
        indices_returned = [
            idx
            for eigenvec, eigenval in combinations
            for idx in [2, 3]
            if np.allclose(eigenvec, eigenvectors[:, idx])
        ]
        assert set(indices_returned) == {2, 3}

    def test_get_sloppy_combinations_none_eigenvectors(self) -> None:
        """Test get_sloppy_combinations returns empty when eigenvectors is None."""
        report = ParameterSensitivityReport(
            is_sloppy=True,
            eigenvalues=np.array([1e6, 1.0, 1e-8]),
            eigenvectors=None,
            eigenvalue_range=14.0,
            effective_dimensionality=1.5,
            stiff_indices=[0],
            sloppy_indices=[2],
            issues=[],
            health_status=HealthStatus.WARNING,
        )
        combinations = report.get_sloppy_combinations()
        assert len(combinations) == 0


@pytest.mark.diagnostics
class TestParameterSensitivityAnalyzer:
    """Tests for ParameterSensitivityAnalyzer class."""

    @pytest.fixture
    def default_config(self) -> DiagnosticsConfig:
        """Create default diagnostics configuration."""
        return DiagnosticsConfig()

    @pytest.fixture
    def analyzer(
        self, default_config: DiagnosticsConfig
    ) -> ParameterSensitivityAnalyzer:
        """Create ParameterSensitivityAnalyzer with default configuration."""
        return ParameterSensitivityAnalyzer(config=default_config)

    @pytest.fixture
    def well_conditioned_jacobian(self) -> np.ndarray:
        """Create a well-conditioned Jacobian matrix (narrow sensitivity spread).

        Uses nearly equal singular values to achieve high effective dimensionality.
        """
        # 100 data points, 3 parameters
        np.random.seed(42)
        # Create Jacobian with nearly equal singular values
        n, p = 100, 3
        U = np.linalg.qr(np.random.randn(n, p))[0]
        # Nearly equal singular values for high effective dimensionality
        # s = [10, 9, 8] gives FIM eigenvalues [100, 81, 64]
        # effective_dim = (100+81+64)^2 / (100^2+81^2+64^2) = 60025/17297 = 2.92
        s = np.array([10.0, 9.0, 8.0])
        V = np.linalg.qr(np.random.randn(p, p))[0]
        return U @ np.diag(s) @ V.T

    @pytest.fixture
    def wide_spread_jacobian(self) -> np.ndarray:
        """Create a Jacobian with wide sensitivity spread (eigenvalue ratio > 1e6)."""
        # Create Jacobian with eigenvalue ratio spanning many orders of magnitude
        np.random.seed(42)
        n, p = 100, 3
        U = np.linalg.qr(np.random.randn(n, p))[0]
        # Singular values spanning many orders of magnitude
        # Note: FIM eigenvalues are squares of singular values
        # So singular values of [1e4, 1.0, 1e-2] give FIM eigenvalues [1e8, 1.0, 1e-4]
        s = np.array([1e4, 1.0, 1e-2])
        V = np.linalg.qr(np.random.randn(p, p))[0]
        return U @ np.diag(s) @ V.T

    @pytest.fixture
    def moderate_spread_jacobian(self) -> np.ndarray:
        """Create a Jacobian with moderate sensitivity spread."""
        np.random.seed(42)
        n, p = 100, 4
        U = np.linalg.qr(np.random.randn(n, p))[0]
        # Singular values spanning moderate range
        s = np.array([1e3, 10.0, 0.1, 0.01])
        V = np.linalg.qr(np.random.randn(p, p))[0]
        return U @ np.diag(s) @ V.T

    # Test basic analysis
    def test_analyze_returns_report(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test analyze returns a ParameterSensitivityReport."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert isinstance(report, ParameterSensitivityReport)
        assert report.available is True

    def test_analyze_eigenvalues_computed(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test eigenvalues are computed correctly."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert len(report.eigenvalues) == well_conditioned_jacobian.shape[1]
        # Eigenvalues should be non-negative (FIM is positive semi-definite)
        assert np.all(report.eigenvalues >= -1e-10)

    def test_analyze_eigenvectors_computed(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test eigenvectors are computed."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.eigenvectors is not None
        n_params = well_conditioned_jacobian.shape[1]
        assert report.eigenvectors.shape == (n_params, n_params)

    def test_analyze_eigenvectors_orthonormal(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test eigenvectors are orthonormal."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.eigenvectors is not None
        # V.T @ V should be identity
        product = report.eigenvectors.T @ report.eigenvectors
        np.testing.assert_array_almost_equal(
            product, np.eye(well_conditioned_jacobian.shape[1])
        )

    # Test sensitivity spread detection
    def test_well_conditioned_not_wide_spread(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test well-conditioned Jacobian returns is_sloppy=False."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.is_sloppy is False
        assert report.health_status == HealthStatus.HEALTHY

    def test_wide_spread_jacobian_detected(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test wide spread Jacobian (eigenvalue ratio > 1e6) returns is_sloppy=True."""
        report = analyzer.analyze(wide_spread_jacobian)
        assert report.is_sloppy is True
        # Eigenvalue range should be large
        assert report.eigenvalue_range > 6.0  # log10(1e6)

    def test_sensitivity_threshold_configurable(self) -> None:
        """Test sensitivity threshold is configurable."""
        # Use very strict threshold
        strict_config = DiagnosticsConfig(sloppy_threshold=0.1)  # 10x range
        strict_analyzer = ParameterSensitivityAnalyzer(config=strict_config)

        # Create Jacobian with 100x eigenvalue range
        np.random.seed(42)
        n, p = 100, 3
        U = np.linalg.qr(np.random.randn(n, p))[0]
        s = np.array([10.0, 1.0, 0.1])  # 100x range in singular values
        V = np.linalg.qr(np.random.randn(p, p))[0]
        J = U @ np.diag(s) @ V.T

        report = strict_analyzer.analyze(J)
        # With strict threshold, this should be considered wide spread
        assert report.is_sloppy is True

    # Test eigenvalue range computation
    def test_eigenvalue_range_computation(
        self, analyzer: ParameterSensitivityAnalyzer
    ) -> None:
        """Test eigenvalue range is computed as log10(max/min)."""
        np.random.seed(42)
        n, p = 100, 3
        U = np.linalg.qr(np.random.randn(n, p))[0]
        # Singular values: [100, 10, 1] -> FIM eigenvalues: [10000, 100, 1]
        s = np.array([100.0, 10.0, 1.0])
        V = np.linalg.qr(np.random.randn(p, p))[0]
        J = U @ np.diag(s) @ V.T

        report = analyzer.analyze(J)
        # Eigenvalue range should be log10(10000/1) = 4
        assert abs(report.eigenvalue_range - 4.0) < 0.5  # Allow some tolerance

    # Test stiff/sloppy direction classification
    def test_stiff_sloppy_classification(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test stiff and sloppy directions are classified correctly."""
        report = analyzer.analyze(wide_spread_jacobian)
        # Should have at least one stiff and one sloppy direction
        assert len(report.stiff_indices) >= 1
        assert len(report.sloppy_indices) >= 1
        # Total should equal number of parameters
        total = len(report.stiff_indices) + len(report.sloppy_indices)
        n_params = wide_spread_jacobian.shape[1]
        assert total <= n_params  # May have intermediate directions

    def test_stiff_indices_have_large_eigenvalues(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test stiff indices correspond to large eigenvalues."""
        report = analyzer.analyze(wide_spread_jacobian)
        if len(report.stiff_indices) > 0 and len(report.sloppy_indices) > 0:
            max_stiff = max(report.eigenvalues[i] for i in report.stiff_indices)
            min_sloppy = min(report.eigenvalues[i] for i in report.sloppy_indices)
            # Stiff eigenvalues should be larger than sloppy
            assert max_stiff > min_sloppy

    def test_sloppy_indices_have_small_eigenvalues(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test sloppy indices correspond to small eigenvalues."""
        report = analyzer.analyze(wide_spread_jacobian)
        if len(report.stiff_indices) > 0 and len(report.sloppy_indices) > 0:
            min_stiff = min(report.eigenvalues[i] for i in report.stiff_indices)
            max_sloppy = max(report.eigenvalues[i] for i in report.sloppy_indices)
            # Stiff eigenvalues should be larger than sloppy
            assert min_stiff > max_sloppy

    def test_all_stiff_when_well_conditioned(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test all directions are classified as stiff for well-conditioned model."""
        report = analyzer.analyze(well_conditioned_jacobian)
        n_params = well_conditioned_jacobian.shape[1]
        # For well-conditioned model, most or all should be stiff
        assert len(report.sloppy_indices) == 0 or len(report.stiff_indices) > 0

    # Test effective dimensionality computation
    def test_effective_dimensionality_range(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test effective dimensionality is in valid range."""
        report = analyzer.analyze(wide_spread_jacobian)
        n_params = wide_spread_jacobian.shape[1]
        # Effective dimensionality should be between 0 and n_params
        assert 0.0 <= report.effective_dimensionality <= n_params

    def test_effective_dimensionality_high_for_well_conditioned(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test effective dimensionality is high for well-conditioned model."""
        report = analyzer.analyze(well_conditioned_jacobian)
        n_params = well_conditioned_jacobian.shape[1]
        # Effective dimensionality should be close to n_params
        assert report.effective_dimensionality >= n_params * 0.8

    def test_effective_dimensionality_low_for_wide_spread(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test effective dimensionality is lower for wide spread model."""
        report = analyzer.analyze(wide_spread_jacobian)
        n_params = wide_spread_jacobian.shape[1]
        # Effective dimensionality should be less than n_params
        assert report.effective_dimensionality < n_params

    def test_effective_dimensionality_computation(
        self, analyzer: ParameterSensitivityAnalyzer
    ) -> None:
        """Test effective dimensionality is computed using participation ratio."""
        # Create known eigenvalue spectrum
        np.random.seed(42)
        n, p = 100, 4
        U = np.linalg.qr(np.random.randn(n, p))[0]
        # Equal singular values -> all eigenvalues equal -> effective_dim = p
        s = np.array([1.0, 1.0, 1.0, 1.0])
        V = np.linalg.qr(np.random.randn(p, p))[0]
        J = U @ np.diag(s) @ V.T

        report = analyzer.analyze(J)
        # For equal eigenvalues, effective dimensionality should be close to n_params
        assert abs(report.effective_dimensionality - 4.0) < 0.5

    # Test issue generation
    def test_issue_sens_001_generated(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test SENS-001 issue is generated for wide sensitivity spread model."""
        report = analyzer.analyze(wide_spread_jacobian)
        issue_codes = [issue.code for issue in report.issues]
        assert "SENS-001" in issue_codes

    def test_issue_sens_001_severity(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test SENS-001 has WARNING severity."""
        report = analyzer.analyze(wide_spread_jacobian)
        sens_001 = [i for i in report.issues if i.code == "SENS-001"]
        if sens_001:
            assert sens_001[0].severity == IssueSeverity.WARNING

    def test_issue_sens_001_category(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test SENS-001 has SENSITIVITY category."""
        report = analyzer.analyze(wide_spread_jacobian)
        sens_001 = [i for i in report.issues if i.code == "SENS-001"]
        if sens_001:
            assert sens_001[0].category == IssueCategory.SENSITIVITY

    def test_issue_sens_001_has_recommendation(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test SENS-001 has a recommendation."""
        report = analyzer.analyze(wide_spread_jacobian)
        sens_001 = [i for i in report.issues if i.code == "SENS-001"]
        if sens_001:
            assert sens_001[0].recommendation is not None
            assert len(sens_001[0].recommendation) > 0

    def test_issue_sens_002_low_effective_dim(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test SENS-002 is generated for low effective dimensionality."""
        report = analyzer.analyze(wide_spread_jacobian)
        n_params = wide_spread_jacobian.shape[1]

        # If effective dimensionality is less than half of n_params
        if report.effective_dimensionality < n_params * 0.5:
            issue_codes = [issue.code for issue in report.issues]
            assert "SENS-002" in issue_codes

    def test_issue_sens_002_severity(
        self, analyzer: ParameterSensitivityAnalyzer
    ) -> None:
        """Test SENS-002 has INFO severity."""
        # Create very wide spread Jacobian with low effective dimensionality
        np.random.seed(42)
        n, p = 100, 5
        U = np.linalg.qr(np.random.randn(n, p))[0]
        # One dominant direction, rest very small
        s = np.array([1e6, 1e-6, 1e-7, 1e-8, 1e-9])
        V = np.linalg.qr(np.random.randn(p, p))[0]
        J = U @ np.diag(s) @ V.T

        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        report = analyzer.analyze(J)

        sens_002 = [i for i in report.issues if i.code == "SENS-002"]
        if sens_002:
            assert sens_002[0].severity == IssueSeverity.INFO

    def test_no_issues_for_healthy_model(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test no issues for well-conditioned model."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert len(report.issues) == 0
        assert report.health_status == HealthStatus.HEALTHY

    # Test health status
    def test_health_status_healthy(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test HEALTHY status for well-conditioned Jacobian."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.health_status == HealthStatus.HEALTHY

    def test_health_status_warning_for_wide_spread(
        self, analyzer: ParameterSensitivityAnalyzer, wide_spread_jacobian: np.ndarray
    ) -> None:
        """Test WARNING status for wide spread model."""
        report = analyzer.analyze(wide_spread_jacobian)
        assert report.health_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]

    # Test graceful degradation
    def test_graceful_degradation_empty_jacobian(
        self, analyzer: ParameterSensitivityAnalyzer
    ) -> None:
        """Test graceful degradation for empty Jacobian."""
        J = np.array([]).reshape(0, 3)
        report = analyzer.analyze(J)
        assert report.available is False
        assert report.error_message is not None

    def test_graceful_degradation_nan_jacobian(
        self, analyzer: ParameterSensitivityAnalyzer
    ) -> None:
        """Test graceful degradation for Jacobian with NaN."""
        J = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        report = analyzer.analyze(J)
        assert report.available is False
        assert "NaN" in report.error_message or "nan" in report.error_message.lower()

    def test_graceful_degradation_inf_jacobian(
        self, analyzer: ParameterSensitivityAnalyzer
    ) -> None:
        """Test graceful degradation for Jacobian with Inf."""
        J = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
        report = analyzer.analyze(J)
        assert report.available is False
        assert "Inf" in report.error_message or "inf" in report.error_message.lower()

    def test_graceful_degradation_zero_jacobian(
        self, analyzer: ParameterSensitivityAnalyzer
    ) -> None:
        """Test graceful degradation for all-zero Jacobian."""
        J = np.zeros((10, 3))
        report = analyzer.analyze(J)
        # Should still produce a result, but mark as having wide spread
        assert report.available is True
        # All-zero Jacobian should have zero eigenvalues -> wide spread
        assert report.is_sloppy is True or report.eigenvalue_range == 0.0

    # Test computation time tracking
    def test_computation_time_tracked(
        self,
        analyzer: ParameterSensitivityAnalyzer,
        well_conditioned_jacobian: np.ndarray,
    ) -> None:
        """Test computation time is tracked."""
        report = analyzer.analyze(well_conditioned_jacobian)
        assert report.computation_time_ms >= 0.0

    # Test analyze_from_fim
    def test_analyze_from_fim(self, analyzer: ParameterSensitivityAnalyzer) -> None:
        """Test analysis from pre-computed FIM."""
        # Well-conditioned FIM
        fim = np.array([[10.0, 1.0, 0.5], [1.0, 8.0, 0.3], [0.5, 0.3, 6.0]])
        report = analyzer.analyze_from_fim(fim)
        assert report.available is True
        assert len(report.eigenvalues) == 3

    def test_analyze_from_fim_wide_spread(
        self, analyzer: ParameterSensitivityAnalyzer
    ) -> None:
        """Test analysis from wide spread FIM."""
        # Wide spread FIM with eigenvalues spanning many orders of magnitude
        eigenvalues = np.array([1e8, 1.0, 1e-6])
        V = np.eye(3)  # Simple eigenvectors
        fim = V @ np.diag(eigenvalues) @ V.T
        report = analyzer.analyze_from_fim(fim)
        assert report.available is True
        assert report.is_sloppy is True
        assert report.eigenvalue_range > 6.0


@pytest.mark.diagnostics
class TestParameterSensitivityAnalyzerEdgeCases:
    """Edge case tests for ParameterSensitivityAnalyzer."""

    def test_single_parameter(self) -> None:
        """Test analysis with single parameter."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(50, 1)
        report = analyzer.analyze(J)
        assert report.available is True
        assert len(report.eigenvalues) == 1
        # Single parameter cannot have wide spread
        assert report.is_sloppy is False

    def test_two_parameters(self) -> None:
        """Test analysis with two parameters."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        np.random.seed(42)
        n, p = 100, 2
        U = np.linalg.qr(np.random.randn(n, p))[0]
        s = np.array([1e4, 1e-4])  # Large ratio
        V = np.linalg.qr(np.random.randn(p, p))[0]
        J = U @ np.diag(s) @ V.T

        report = analyzer.analyze(J)
        assert report.available is True
        assert len(report.eigenvalues) == 2
        assert report.is_sloppy is True

    def test_many_parameters(self) -> None:
        """Test analysis with many parameters."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(1000, 20)
        report = analyzer.analyze(J)
        assert report.available is True
        assert len(report.eigenvalues) == 20

    def test_underdetermined_system(self) -> None:
        """Test analysis with more parameters than data points."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(5, 10)  # 5 data points, 10 parameters
        report = analyzer.analyze(J)
        assert report.available is True
        # Should detect wide spread behavior due to underdetermined system
        # Some eigenvalues will be zero or near-zero

    def test_square_jacobian(self) -> None:
        """Test analysis with square Jacobian."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(10, 10)
        report = analyzer.analyze(J)
        assert report.available is True
        assert len(report.eigenvalues) == 10

    def test_very_small_values(self) -> None:
        """Test analysis with very small Jacobian values."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(50, 3) * 1e-15
        report = analyzer.analyze(J)
        # Should handle gracefully
        assert report.available is True

    def test_very_large_values(self) -> None:
        """Test analysis with very large Jacobian values."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        np.random.seed(42)
        J = np.random.randn(50, 3) * 1e15
        report = analyzer.analyze(J)
        # Should handle gracefully
        assert report.available is True

    def test_mixed_scale_columns(self) -> None:
        """Test analysis with mixed scale columns."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        np.random.seed(42)
        J = np.zeros((100, 3))
        J[:, 0] = np.random.randn(100) * 1e-10
        J[:, 1] = np.random.randn(100) * 1.0
        J[:, 2] = np.random.randn(100) * 1e10
        report = analyzer.analyze(J)
        # Should detect wide spread behavior
        assert report.available is True
        assert report.is_sloppy is True

    def test_negative_eigenvalue_handling(self) -> None:
        """Test handling of near-zero/negative eigenvalues from numerical noise."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        # Create a rank-deficient Jacobian
        np.random.seed(42)
        J = np.zeros((100, 3))
        J[:, 0] = np.random.randn(100)
        J[:, 1] = np.random.randn(100)
        J[:, 2] = J[:, 0] + J[:, 1]  # Linear combination

        report = analyzer.analyze(J)
        # Should handle gracefully - eigenvalues should be non-negative
        assert report.available is True
        assert np.all(report.eigenvalues >= -1e-10)


@pytest.mark.diagnostics
class TestMultiExponentialSensitivity:
    """Tests for sensitivity detection with multi-exponential decay.

    Multi-exponential decay models are known to have wide sensitivity spread because:
    1. The amplitudes and rate constants are highly correlated
    2. The eigenvalue spectrum spans many orders of magnitude
    3. Only certain parameter combinations are well-determined

    This is a classic test case from the parameter sensitivity literature.
    """

    @pytest.fixture
    def multi_exponential_jacobian_2_exp(self) -> tuple[np.ndarray, np.ndarray]:
        """Create Jacobian for 2-exponential decay model.

        Model: y = A1 * exp(-k1 * t) + A2 * exp(-k2 * t)
        Parameters: [A1, k1, A2, k2]

        Returns tuple of (Jacobian, time_points)
        """
        np.random.seed(42)
        t = np.linspace(0.01, 10.0, 200)

        # True parameters: similar amplitudes, different rates
        A1, k1 = 1.0, 0.5
        A2, k2 = 0.8, 2.0

        # Compute Jacobian analytically
        # dy/dA1 = exp(-k1 * t)
        # dy/dk1 = -A1 * t * exp(-k1 * t)
        # dy/dA2 = exp(-k2 * t)
        # dy/dk2 = -A2 * t * exp(-k2 * t)
        J = np.zeros((len(t), 4))
        J[:, 0] = np.exp(-k1 * t)
        J[:, 1] = -A1 * t * np.exp(-k1 * t)
        J[:, 2] = np.exp(-k2 * t)
        J[:, 3] = -A2 * t * np.exp(-k2 * t)

        return J, t

    @pytest.fixture
    def multi_exponential_jacobian_3_exp(self) -> tuple[np.ndarray, np.ndarray]:
        """Create Jacobian for 3-exponential decay model.

        Model: y = A1 * exp(-k1 * t) + A2 * exp(-k2 * t) + A3 * exp(-k3 * t)
        Parameters: [A1, k1, A2, k2, A3, k3]

        This has even wider spread than the 2-exponential case.
        """
        np.random.seed(42)
        t = np.linspace(0.01, 10.0, 300)

        # True parameters
        A1, k1 = 1.0, 0.1
        A2, k2 = 0.5, 1.0
        A3, k3 = 0.3, 5.0

        J = np.zeros((len(t), 6))
        J[:, 0] = np.exp(-k1 * t)
        J[:, 1] = -A1 * t * np.exp(-k1 * t)
        J[:, 2] = np.exp(-k2 * t)
        J[:, 3] = -A2 * t * np.exp(-k2 * t)
        J[:, 4] = np.exp(-k3 * t)
        J[:, 5] = -A3 * t * np.exp(-k3 * t)

        return J, t

    @pytest.fixture
    def similar_rate_constants_jacobian(self) -> np.ndarray:
        """Create Jacobian for 2-exponential with similar rate constants (very wide spread)."""
        np.random.seed(42)
        t = np.linspace(0.01, 10.0, 200)

        # Very similar rate constants -> very wide spread
        A1, k1 = 1.0, 1.0
        A2, k2 = 0.8, 1.1  # k2 very close to k1

        J = np.zeros((len(t), 4))
        J[:, 0] = np.exp(-k1 * t)
        J[:, 1] = -A1 * t * np.exp(-k1 * t)
        J[:, 2] = np.exp(-k2 * t)
        J[:, 3] = -A2 * t * np.exp(-k2 * t)

        return J

    def test_2_exponential_has_wide_spread(
        self, multi_exponential_jacobian_2_exp: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test 2-exponential decay model is detected as having wide spread."""
        J, _t = multi_exponential_jacobian_2_exp
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        assert report.available is True
        assert report.is_sloppy is True

    def test_2_exponential_eigenvalue_range(
        self, multi_exponential_jacobian_2_exp: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test 2-exponential eigenvalue range spans multiple orders of magnitude."""
        J, _t = multi_exponential_jacobian_2_exp
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        # Multi-exponential models typically have eigenvalue ranges > 2-3 orders of magnitude
        assert report.eigenvalue_range >= 2.0

    def test_2_exponential_has_sensitivity_issue(
        self, multi_exponential_jacobian_2_exp: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test 2-exponential generates SENS-001 issue."""
        J, _t = multi_exponential_jacobian_2_exp
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        issue_codes = [issue.code for issue in report.issues]
        assert "SENS-001" in issue_codes

    def test_3_exponential_has_wide_spread(
        self, multi_exponential_jacobian_3_exp: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test 3-exponential decay model is detected as having wide spread."""
        J, _t = multi_exponential_jacobian_3_exp
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        assert report.available is True
        assert report.is_sloppy is True

    def test_3_exponential_wider_spread_than_2_exponential(
        self,
        multi_exponential_jacobian_2_exp: tuple[np.ndarray, np.ndarray],
        multi_exponential_jacobian_3_exp: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test 3-exponential has larger eigenvalue range than 2-exponential."""
        J_2exp, _ = multi_exponential_jacobian_2_exp
        J_3exp, _ = multi_exponential_jacobian_3_exp

        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report_2exp = analyzer.analyze(J_2exp)
        report_3exp = analyzer.analyze(J_3exp)

        # 3-exponential should have larger eigenvalue range
        assert report_3exp.eigenvalue_range >= report_2exp.eigenvalue_range

    def test_3_exponential_has_low_effective_dimensionality(
        self, multi_exponential_jacobian_3_exp: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test 3-exponential has low effective dimensionality relative to 6 params."""
        J, _t = multi_exponential_jacobian_3_exp
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        # 6 parameters but fewer effective degrees of freedom
        assert report.effective_dimensionality < 6.0

    def test_3_exponential_sloppy_directions(
        self, multi_exponential_jacobian_3_exp: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test 3-exponential has multiple sloppy directions."""
        J, _t = multi_exponential_jacobian_3_exp
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        # Should have at least one sloppy direction
        assert len(report.sloppy_indices) >= 1

    def test_similar_rates_very_wide_spread(
        self, similar_rate_constants_jacobian: np.ndarray
    ) -> None:
        """Test 2-exponential with similar rates has very wide spread."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(similar_rate_constants_jacobian)
        assert report.available is True
        assert report.is_sloppy is True
        # Similar rate constants should make it even more spread
        assert report.eigenvalue_range >= 3.0

    def test_similar_rates_low_effective_dim(
        self, similar_rate_constants_jacobian: np.ndarray
    ) -> None:
        """Test 2-exponential with similar rates has very low effective dimensionality."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(similar_rate_constants_jacobian)
        # With 4 parameters but near-degenerate rates, effective dim should be low
        assert report.effective_dimensionality < 3.0

    def test_sloppy_combinations_represent_parameter_mixtures(
        self, multi_exponential_jacobian_2_exp: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test sloppy combinations show which parameter mixtures are poorly determined."""
        J, _t = multi_exponential_jacobian_2_exp
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        combinations = report.get_sloppy_combinations()

        # Should have at least one sloppy combination
        assert len(combinations) >= 1

        # Each combination should be a valid eigenvector
        for eigenvec, eigenval in combinations:
            assert len(eigenvec) == 4  # 4 parameters
            # Eigenvector should be normalized
            norm = np.linalg.norm(eigenvec)
            assert abs(norm - 1.0) < 1e-6

    def test_stiff_directions_are_well_determined(
        self, multi_exponential_jacobian_2_exp: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test stiff directions correspond to well-determined parameter combinations."""
        J, _t = multi_exponential_jacobian_2_exp
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)

        # Stiff directions should have large eigenvalues
        if len(report.stiff_indices) > 0:
            stiff_eigenvalues = [report.eigenvalues[i] for i in report.stiff_indices]
            # Stiff eigenvalues should be significantly larger than sloppy
            if len(report.sloppy_indices) > 0:
                sloppy_eigenvalues = [
                    report.eigenvalues[i] for i in report.sloppy_indices
                ]
                assert min(stiff_eigenvalues) > max(sloppy_eigenvalues)


@pytest.mark.diagnostics
class TestKnownSensitivityPatterns:
    """Tests for known sensitivity patterns from the literature."""

    def test_power_law_decay_sensitivity(self) -> None:
        """Test power-law decay model sensitivity.

        Model: y = A * t^(-alpha)
        This is a known model with parameter correlations when fitting amplitude and exponent.
        """
        np.random.seed(42)
        t = np.linspace(0.1, 10.0, 100)

        # Parameters
        A, alpha = 1.0, 0.5

        # Jacobian
        # dy/dA = t^(-alpha)
        # dy/dalpha = -A * t^(-alpha) * log(t)
        J = np.zeros((len(t), 2))
        J[:, 0] = t ** (-alpha)
        J[:, 1] = -A * t ** (-alpha) * np.log(t)

        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        # Power-law is inherently sensitive due to log-dependence
        assert report.available is True
        # May or may not be classified as wide spread depending on threshold

    def test_michaelis_menten_kinetics(self) -> None:
        """Test Michaelis-Menten model for enzyme kinetics.

        Model: v = Vmax * S / (Km + S)
        Known to have correlated Vmax and Km parameters.
        """
        np.random.seed(42)
        S = np.logspace(-2, 2, 100)  # Substrate concentration

        # Parameters
        Vmax, Km = 1.0, 0.5

        # Jacobian
        # dv/dVmax = S / (Km + S)
        # dv/dKm = -Vmax * S / (Km + S)^2
        J = np.zeros((len(S), 2))
        J[:, 0] = S / (Km + S)
        J[:, 1] = -Vmax * S / (Km + S) ** 2

        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        assert report.available is True
        # Michaelis-Menten with good data coverage should be identifiable
        # but may show some correlation

    def test_sigmoid_model(self) -> None:
        """Test sigmoid/logistic model.

        Model: y = L / (1 + exp(-k * (x - x0)))
        Parameters: L (maximum), k (steepness), x0 (midpoint)
        """
        np.random.seed(42)
        x = np.linspace(-5, 5, 100)

        # Parameters
        L, k, x0 = 1.0, 1.0, 0.0

        # Jacobian
        exp_term = np.exp(-k * (x - x0))
        denom = 1 + exp_term
        y = L / denom

        J = np.zeros((len(x), 3))
        J[:, 0] = 1 / denom  # dy/dL
        J[:, 1] = L * (x - x0) * exp_term / denom**2  # dy/dk
        J[:, 2] = -L * k * exp_term / denom**2  # dy/dx0

        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        assert report.available is True
        # Sigmoid with good coverage should be reasonably well-conditioned

    def test_gaussian_peak_model(self) -> None:
        """Test Gaussian peak model.

        Model: y = A * exp(-((x - x0)^2) / (2 * sigma^2))
        Parameters: A (amplitude), x0 (center), sigma (width)
        """
        np.random.seed(42)
        x = np.linspace(-5, 5, 100)

        # Parameters
        A, x0, sigma = 1.0, 0.0, 1.0

        # Jacobian
        z = (x - x0) / sigma
        gauss = np.exp(-(z**2) / 2)

        J = np.zeros((len(x), 3))
        J[:, 0] = gauss  # dy/dA
        J[:, 1] = A * gauss * z / sigma  # dy/dx0
        J[:, 2] = A * gauss * z**2 / sigma  # dy/dsigma

        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(J)
        assert report.available is True
        # Gaussian peak with good coverage should be identifiable
