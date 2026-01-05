"""Unit tests for the diagnostics types module.

Tests cover:
- Enumerations (HealthStatus, IssueSeverity, IssueCategory, DiagnosticLevel)
- ModelHealthIssue dataclass
- AnalysisResult dataclass
- DiagnosticsConfig dataclass
"""

import pytest

from nlsq.diagnostics.types import (
    AnalysisResult,
    DiagnosticLevel,
    DiagnosticsConfig,
    HealthStatus,
    IssueCategory,
    IssueSeverity,
    ModelHealthIssue,
)


class TestHealthStatus:
    """Tests for HealthStatus enumeration."""

    def test_health_status_values(self) -> None:
        """Test HealthStatus has expected values."""
        assert hasattr(HealthStatus, "HEALTHY")
        assert hasattr(HealthStatus, "WARNING")
        assert hasattr(HealthStatus, "CRITICAL")

    def test_health_status_is_enum(self) -> None:
        """Test HealthStatus is an Enum."""
        assert HealthStatus.HEALTHY.value is not None
        assert HealthStatus.WARNING.value is not None
        assert HealthStatus.CRITICAL.value is not None

    def test_health_status_unique_values(self) -> None:
        """Test HealthStatus values are unique."""
        values = [status.value for status in HealthStatus]
        assert len(values) == len(set(values))

    def test_health_status_membership(self) -> None:
        """Test HealthStatus membership checks."""
        assert HealthStatus.HEALTHY in HealthStatus
        assert HealthStatus.WARNING in HealthStatus
        assert HealthStatus.CRITICAL in HealthStatus


class TestIssueSeverity:
    """Tests for IssueSeverity enumeration."""

    def test_issue_severity_values(self) -> None:
        """Test IssueSeverity has expected values."""
        assert hasattr(IssueSeverity, "INFO")
        assert hasattr(IssueSeverity, "WARNING")
        assert hasattr(IssueSeverity, "CRITICAL")

    def test_issue_severity_is_enum(self) -> None:
        """Test IssueSeverity is an Enum."""
        assert IssueSeverity.INFO.value is not None
        assert IssueSeverity.WARNING.value is not None
        assert IssueSeverity.CRITICAL.value is not None

    def test_issue_severity_unique_values(self) -> None:
        """Test IssueSeverity values are unique."""
        values = [severity.value for severity in IssueSeverity]
        assert len(values) == len(set(values))


class TestIssueCategory:
    """Tests for IssueCategory enumeration."""

    def test_issue_category_values(self) -> None:
        """Test IssueCategory has expected values."""
        assert hasattr(IssueCategory, "IDENTIFIABILITY")
        assert hasattr(IssueCategory, "GRADIENT")
        assert hasattr(IssueCategory, "CORRELATION")
        assert hasattr(IssueCategory, "CONDITIONING")
        assert hasattr(IssueCategory, "CONVERGENCE")
        assert hasattr(IssueCategory, "SENSITIVITY")

    def test_issue_category_count(self) -> None:
        """Test IssueCategory has expected number of values."""
        assert len(IssueCategory) == 6

    def test_issue_category_unique_values(self) -> None:
        """Test IssueCategory values are unique."""
        values = [category.value for category in IssueCategory]
        assert len(values) == len(set(values))


class TestDiagnosticLevel:
    """Tests for DiagnosticLevel enumeration."""

    def test_diagnostic_level_values(self) -> None:
        """Test DiagnosticLevel has expected values."""
        assert hasattr(DiagnosticLevel, "BASIC")
        assert hasattr(DiagnosticLevel, "FULL")

    def test_diagnostic_level_count(self) -> None:
        """Test DiagnosticLevel has expected number of values."""
        assert len(DiagnosticLevel) == 2

    def test_diagnostic_level_unique_values(self) -> None:
        """Test DiagnosticLevel values are unique."""
        values = [level.value for level in DiagnosticLevel]
        assert len(values) == len(set(values))


class TestModelHealthIssue:
    """Tests for ModelHealthIssue dataclass."""

    @pytest.fixture
    def sample_issue(self) -> ModelHealthIssue:
        """Create a sample ModelHealthIssue for testing."""
        return ModelHealthIssue(
            category=IssueCategory.IDENTIFIABILITY,
            severity=IssueSeverity.CRITICAL,
            code="IDENT-001",
            message="Test issue message",
            affected_parameters=(0, 1),
            details={"test_key": "test_value"},
            recommendation="Test recommendation",
        )

    def test_model_health_issue_creation(self, sample_issue: ModelHealthIssue) -> None:
        """Test ModelHealthIssue can be created with valid data."""
        assert sample_issue.category == IssueCategory.IDENTIFIABILITY
        assert sample_issue.severity == IssueSeverity.CRITICAL
        assert sample_issue.code == "IDENT-001"
        assert sample_issue.message == "Test issue message"
        assert sample_issue.affected_parameters == (0, 1)
        assert sample_issue.details == {"test_key": "test_value"}
        assert sample_issue.recommendation == "Test recommendation"

    def test_model_health_issue_frozen(self, sample_issue: ModelHealthIssue) -> None:
        """Test ModelHealthIssue is immutable (frozen)."""
        with pytest.raises(AttributeError):
            sample_issue.code = "NEW-001"  # type: ignore[misc]

    def test_model_health_issue_slots(self) -> None:
        """Test ModelHealthIssue uses __slots__ for memory efficiency."""
        assert hasattr(ModelHealthIssue, "__slots__")

    def test_model_health_issue_none_affected_parameters(self) -> None:
        """Test ModelHealthIssue with None affected_parameters."""
        issue = ModelHealthIssue(
            category=IssueCategory.GRADIENT,
            severity=IssueSeverity.WARNING,
            code="GRAD-001",
            message="Gradient issue",
            affected_parameters=None,
            details={},
            recommendation="Fix gradient",
        )
        assert issue.affected_parameters is None

    def test_model_health_issue_empty_details(self) -> None:
        """Test ModelHealthIssue with empty details dict."""
        issue = ModelHealthIssue(
            category=IssueCategory.CONDITIONING,
            severity=IssueSeverity.INFO,
            code="COND-001",
            message="Conditioning issue",
            affected_parameters=None,
            details={},
            recommendation="Check conditioning",
        )
        assert issue.details == {}

    def test_model_health_issue_empty_code_raises(self) -> None:
        """Test ModelHealthIssue raises ValueError for empty code."""
        with pytest.raises(ValueError, match="Issue code cannot be empty"):
            ModelHealthIssue(
                category=IssueCategory.IDENTIFIABILITY,
                severity=IssueSeverity.CRITICAL,
                code="",
                message="Test message",
                affected_parameters=None,
                details={},
                recommendation="Test recommendation",
            )

    def test_model_health_issue_empty_message_raises(self) -> None:
        """Test ModelHealthIssue raises ValueError for empty message."""
        with pytest.raises(ValueError, match="Issue message cannot be empty"):
            ModelHealthIssue(
                category=IssueCategory.IDENTIFIABILITY,
                severity=IssueSeverity.CRITICAL,
                code="IDENT-001",
                message="",
                affected_parameters=None,
                details={},
                recommendation="Test recommendation",
            )

    def test_model_health_issue_equality(self) -> None:
        """Test ModelHealthIssue equality comparison."""
        issue1 = ModelHealthIssue(
            category=IssueCategory.IDENTIFIABILITY,
            severity=IssueSeverity.CRITICAL,
            code="IDENT-001",
            message="Test",
            affected_parameters=(0,),
            details={"key": "value"},
            recommendation="Rec",
        )
        issue2 = ModelHealthIssue(
            category=IssueCategory.IDENTIFIABILITY,
            severity=IssueSeverity.CRITICAL,
            code="IDENT-001",
            message="Test",
            affected_parameters=(0,),
            details={"key": "value"},
            recommendation="Rec",
        )
        assert issue1 == issue2

    def test_model_health_issue_not_hashable(
        self, sample_issue: ModelHealthIssue
    ) -> None:
        """Test ModelHealthIssue is not hashable due to dict field.

        Even though the dataclass is frozen, the details field is a dict
        which is not hashable, making the entire object unhashable.
        This is expected behavior.
        """
        with pytest.raises(TypeError, match="unhashable type"):
            hash(sample_issue)

    def test_model_health_issue_can_be_stored_in_list(
        self, sample_issue: ModelHealthIssue
    ) -> None:
        """Test ModelHealthIssue can be stored in lists."""
        issues = [sample_issue]
        assert sample_issue in issues
        assert len(issues) == 1


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_analysis_result_defaults(self) -> None:
        """Test AnalysisResult default values."""
        result = AnalysisResult()
        assert result.available is True
        assert result.error_message is None
        assert result.computation_time_ms == 0.0

    def test_analysis_result_available_false(self) -> None:
        """Test AnalysisResult with available=False."""
        result = AnalysisResult(
            available=False,
            error_message="SVD computation failed",
            computation_time_ms=5.2,
        )
        assert result.available is False
        assert result.error_message == "SVD computation failed"
        assert result.computation_time_ms == 5.2

    def test_analysis_result_slots(self) -> None:
        """Test AnalysisResult uses __slots__ for memory efficiency."""
        assert hasattr(AnalysisResult, "__slots__")

    def test_analysis_result_mutable(self) -> None:
        """Test AnalysisResult is mutable (not frozen)."""
        result = AnalysisResult()
        result.available = False
        result.error_message = "Updated error"
        result.computation_time_ms = 10.5
        assert result.available is False
        assert result.error_message == "Updated error"
        assert result.computation_time_ms == 10.5


class TestDiagnosticsConfig:
    """Tests for DiagnosticsConfig dataclass."""

    def test_diagnostics_config_defaults(self) -> None:
        """Test DiagnosticsConfig default values."""
        config = DiagnosticsConfig()
        assert config.level == DiagnosticLevel.BASIC
        assert config.condition_threshold == 1e8
        assert config.correlation_threshold == 0.95
        assert config.imbalance_threshold == 1e6
        assert config.vanishing_threshold == 1e-6
        assert config.sloppy_threshold == 1e-6
        assert config.gradient_window_size == 100
        assert config.verbose is True
        assert config.emit_warnings is True

    def test_diagnostics_config_custom_values(self) -> None:
        """Test DiagnosticsConfig with custom values."""
        config = DiagnosticsConfig(
            level=DiagnosticLevel.FULL,
            condition_threshold=1e10,
            correlation_threshold=0.99,
            imbalance_threshold=1e8,
            vanishing_threshold=1e-8,
            sloppy_threshold=1e-8,
            gradient_window_size=50,
            verbose=False,
            emit_warnings=False,
        )
        assert config.level == DiagnosticLevel.FULL
        assert config.condition_threshold == 1e10
        assert config.correlation_threshold == 0.99
        assert config.imbalance_threshold == 1e8
        assert config.vanishing_threshold == 1e-8
        assert config.sloppy_threshold == 1e-8
        assert config.gradient_window_size == 50
        assert config.verbose is False
        assert config.emit_warnings is False

    def test_diagnostics_config_frozen(self) -> None:
        """Test DiagnosticsConfig is immutable (frozen)."""
        config = DiagnosticsConfig()
        with pytest.raises(AttributeError):
            config.verbose = False  # type: ignore[misc]

    def test_diagnostics_config_slots(self) -> None:
        """Test DiagnosticsConfig uses __slots__ for memory efficiency."""
        assert hasattr(DiagnosticsConfig, "__slots__")

    def test_diagnostics_config_invalid_condition_threshold(self) -> None:
        """Test DiagnosticsConfig rejects non-positive condition_threshold."""
        with pytest.raises(ValueError, match="condition_threshold must be positive"):
            DiagnosticsConfig(condition_threshold=0)
        with pytest.raises(ValueError, match="condition_threshold must be positive"):
            DiagnosticsConfig(condition_threshold=-1)

    def test_diagnostics_config_invalid_correlation_threshold(self) -> None:
        """Test DiagnosticsConfig rejects invalid correlation_threshold."""
        with pytest.raises(ValueError, match="correlation_threshold must be in"):
            DiagnosticsConfig(correlation_threshold=0)
        with pytest.raises(ValueError, match="correlation_threshold must be in"):
            DiagnosticsConfig(correlation_threshold=-0.5)
        with pytest.raises(ValueError, match="correlation_threshold must be in"):
            DiagnosticsConfig(correlation_threshold=1.5)

    def test_diagnostics_config_correlation_threshold_boundary(self) -> None:
        """Test DiagnosticsConfig accepts correlation_threshold=1.0."""
        config = DiagnosticsConfig(correlation_threshold=1.0)
        assert config.correlation_threshold == 1.0

    def test_diagnostics_config_invalid_imbalance_threshold(self) -> None:
        """Test DiagnosticsConfig rejects non-positive imbalance_threshold."""
        with pytest.raises(ValueError, match="imbalance_threshold must be positive"):
            DiagnosticsConfig(imbalance_threshold=0)
        with pytest.raises(ValueError, match="imbalance_threshold must be positive"):
            DiagnosticsConfig(imbalance_threshold=-100)

    def test_diagnostics_config_invalid_vanishing_threshold(self) -> None:
        """Test DiagnosticsConfig rejects non-positive vanishing_threshold."""
        with pytest.raises(ValueError, match="vanishing_threshold must be positive"):
            DiagnosticsConfig(vanishing_threshold=0)
        with pytest.raises(ValueError, match="vanishing_threshold must be positive"):
            DiagnosticsConfig(vanishing_threshold=-1e-6)

    def test_diagnostics_config_invalid_sloppy_threshold(self) -> None:
        """Test DiagnosticsConfig rejects non-positive sloppy_threshold."""
        with pytest.raises(ValueError, match="sloppy_threshold must be positive"):
            DiagnosticsConfig(sloppy_threshold=0)
        with pytest.raises(ValueError, match="sloppy_threshold must be positive"):
            DiagnosticsConfig(sloppy_threshold=-1e-6)

    def test_diagnostics_config_invalid_gradient_window_size(self) -> None:
        """Test DiagnosticsConfig rejects non-positive gradient_window_size."""
        with pytest.raises(ValueError, match="gradient_window_size must be positive"):
            DiagnosticsConfig(gradient_window_size=0)
        with pytest.raises(ValueError, match="gradient_window_size must be positive"):
            DiagnosticsConfig(gradient_window_size=-10)

    def test_diagnostics_config_hashing(self) -> None:
        """Test DiagnosticsConfig is hashable (for caching)."""
        config = DiagnosticsConfig()
        config_set = {config}
        assert config in config_set

    def test_diagnostics_config_equality(self) -> None:
        """Test DiagnosticsConfig equality comparison."""
        config1 = DiagnosticsConfig(condition_threshold=1e9)
        config2 = DiagnosticsConfig(condition_threshold=1e9)
        config3 = DiagnosticsConfig(condition_threshold=1e10)
        assert config1 == config2
        assert config1 != config3


class TestRecommendationsModule:
    """Tests for the recommendations module."""

    def test_recommendations_import(self) -> None:
        """Test RECOMMENDATIONS can be imported."""
        from nlsq.diagnostics.recommendations import RECOMMENDATIONS

        assert isinstance(RECOMMENDATIONS, dict)

    def test_recommendations_has_expected_codes(self) -> None:
        """Test RECOMMENDATIONS contains expected issue codes.

        Note: Issue codes were renamed from SLOPPY-* to SENS-* in the
        domain-agnostic refactoring.
        """
        from nlsq.diagnostics.recommendations import RECOMMENDATIONS

        expected_codes = [
            "IDENT-001",
            "IDENT-002",
            "CORR-001",
            "GRAD-001",
            "GRAD-002",
            "GRAD-003",
            "COND-001",
            "SENS-001",
            "SENS-002",
        ]
        for code in expected_codes:
            assert code in RECOMMENDATIONS, f"Missing recommendation for {code}"

    def test_recommendations_values_are_strings(self) -> None:
        """Test all recommendation values are non-empty strings."""
        from nlsq.diagnostics.recommendations import RECOMMENDATIONS

        for code, recommendation in RECOMMENDATIONS.items():
            assert isinstance(recommendation, str), f"{code} value is not a string"
            assert len(recommendation) > 0, f"{code} has empty recommendation"

    def test_get_recommendation_known_code(self) -> None:
        """Test get_recommendation returns correct text for known code."""
        from nlsq.diagnostics.recommendations import get_recommendation

        recommendation = get_recommendation("IDENT-001")
        assert "Structural unidentifiability" in recommendation

    def test_get_recommendation_unknown_code(self) -> None:
        """Test get_recommendation returns default for unknown code."""
        from nlsq.diagnostics.recommendations import get_recommendation

        recommendation = get_recommendation("UNKNOWN-999")
        assert "No specific recommendation" in recommendation


class TestModuleExports:
    """Tests for module-level exports and imports."""

    def test_diagnostics_init_exports(self) -> None:
        """Test diagnostics __init__ exports expected names."""
        from nlsq import diagnostics

        # Enumerations
        assert hasattr(diagnostics, "HealthStatus")
        assert hasattr(diagnostics, "IssueSeverity")
        assert hasattr(diagnostics, "IssueCategory")
        assert hasattr(diagnostics, "DiagnosticLevel")

        # Core types
        assert hasattr(diagnostics, "ModelHealthIssue")
        assert hasattr(diagnostics, "AnalysisResult")
        assert hasattr(diagnostics, "DiagnosticsConfig")

        # Recommendations
        assert hasattr(diagnostics, "RECOMMENDATIONS")

    def test_diagnostics_types_direct_import(self) -> None:
        """Test types can be imported directly from diagnostics.types."""
        from nlsq.diagnostics.types import (
            AnalysisResult,
            DiagnosticLevel,
            DiagnosticsConfig,
            HealthStatus,
            IssueCategory,
            IssueSeverity,
            ModelHealthIssue,
        )

        # Verify imports work
        assert HealthStatus is not None
        assert IssueSeverity is not None
        assert IssueCategory is not None
        assert DiagnosticLevel is not None
        assert ModelHealthIssue is not None
        assert AnalysisResult is not None
        assert DiagnosticsConfig is not None
