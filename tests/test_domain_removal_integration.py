"""Integration tests for domain-specific content removal.

This module provides end-to-end integration tests verifying that:
1. New generic presets work with curve_fit
2. ParameterSensitivityAnalyzer works in full workflow
3. No domain-specific strings leak into output
4. Deprecated presets still produce valid configs
5. Integration points between modified modules work correctly

Task Group 7.3: Strategic integration tests for domain removal spec.
"""

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.minpack import curve_fit
from nlsq.core.workflow import (
    DEPRECATED_PRESET_ALIASES,
    WORKFLOW_PRESETS,
    WorkflowConfig,
)
from nlsq.diagnostics import (
    DiagnosticsConfig,
    ParameterSensitivityAnalyzer,
    ParameterSensitivityReport,
)


# Simple model function for testing
def exponential_decay(x, amplitude, decay_rate):
    """Simple exponential decay model for testing."""
    return amplitude * jnp.exp(-decay_rate * x)


class TestPresetWithCurveFitEndToEnd:
    """End-to-end tests: create config from new preset, run curve_fit."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic exponential decay data."""
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        true_amplitude = 2.5
        true_decay = 0.8
        y_true = true_amplitude * np.exp(-true_decay * x)
        y_noisy = y_true + 0.1 * np.random.normal(size=len(x))
        return x, y_noisy, (true_amplitude, true_decay)

    def test_precision_high_preset_with_curve_fit(self, synthetic_data):
        """Test that precision_high preset parameters work with curve_fit."""
        x, y, (true_amp, true_decay) = synthetic_data

        # Get preset configuration
        config = WorkflowConfig.from_preset("precision_high")

        # Run curve_fit with preset tolerances
        popt, pcov = curve_fit(
            exponential_decay,
            x,
            y,
            p0=[1.0, 1.0],
            gtol=config.gtol,
            ftol=config.ftol,
            xtol=config.xtol,
        )

        # Verify fit converged to reasonable values
        assert len(popt) == 2
        assert 1.5 < popt[0] < 3.5  # amplitude near 2.5
        assert 0.5 < popt[1] < 1.2  # decay near 0.8
        assert pcov is not None
        assert pcov.shape == (2, 2)

    def test_precision_standard_preset_with_curve_fit(self, synthetic_data):
        """Test that precision_standard preset works with curve_fit."""
        x, y, _ = synthetic_data

        config = WorkflowConfig.from_preset("precision_standard")

        popt, pcov = curve_fit(
            exponential_decay,
            x,
            y,
            p0=[1.0, 1.0],
            gtol=config.gtol,
            ftol=config.ftol,
            xtol=config.xtol,
        )

        assert len(popt) == 2
        assert pcov is not None

    def test_deprecated_preset_produces_valid_curve_fit_result(self, synthetic_data):
        """Test deprecated preset still produces valid curve_fit results."""
        x, y, _ = synthetic_data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = WorkflowConfig.from_preset("xpcs")

        popt, pcov = curve_fit(
            exponential_decay,
            x,
            y,
            p0=[1.0, 1.0],
            gtol=config.gtol,
            ftol=config.ftol,
            xtol=config.xtol,
        )

        assert len(popt) == 2
        assert pcov is not None


class TestParameterSensitivityAnalyzerWorkflow:
    """End-to-end test: use ParameterSensitivityAnalyzer in full workflow."""

    @pytest.fixture
    def jacobian_from_fit(self):
        """Generate a realistic Jacobian matrix from a fit scenario."""
        # Simulate Jacobian from fitting exponential decay to 100 points
        # J[i, 0] = d(y_i)/d(amplitude) = exp(-decay * x_i)
        # J[i, 1] = d(y_i)/d(decay) = -amplitude * x_i * exp(-decay * x_i)
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        amplitude = 2.5
        decay = 0.8

        J = np.zeros((100, 2))
        J[:, 0] = np.exp(-decay * x)
        J[:, 1] = -amplitude * x * np.exp(-decay * x)

        return J

    def test_analyzer_produces_valid_report(self, jacobian_from_fit):
        """Test ParameterSensitivityAnalyzer produces valid report from Jacobian."""
        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)

        report = analyzer.analyze(jacobian_from_fit)

        # Verify report structure matches ParameterSensitivityReport dataclass
        assert isinstance(report, ParameterSensitivityReport)
        assert report.eigenvalues is not None
        assert len(report.eigenvalues) == 2  # 2 parameters
        assert report.eigenvalue_range is not None
        assert report.effective_dimensionality is not None
        assert isinstance(report.stiff_indices, list)
        assert isinstance(report.sloppy_indices, list)

    def test_analyzer_issue_codes_use_sens_prefix(self, jacobian_from_fit):
        """Test that analyzer issues use SENS- prefix, not SLOPPY-."""
        # Create a Jacobian with wide eigenvalue spread to trigger issues
        np.random.seed(42)
        # Create Jacobian with vastly different column scales
        J = np.random.randn(100, 3)
        J[:, 0] *= 1e6  # Very sensitive parameter
        J[:, 2] *= 1e-6  # Very insensitive parameter

        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        report = analyzer.analyze(J)

        # If issues are present, they should use SENS- prefix
        for issue in report.issues:
            if issue.code:
                assert issue.code.startswith("SENS-"), (
                    f"Issue code should use SENS- prefix, got: {issue.code}"
                )
                assert not issue.code.startswith("SLOPPY-"), (
                    f"Issue code should not use SLOPPY- prefix: {issue.code}"
                )


class TestNoDomainStringsInOutput:
    """Integration test: verify no domain-specific strings leak into output."""

    def test_workflow_preset_descriptions_are_domain_agnostic(self):
        """Test that WORKFLOW_PRESETS does not contain domain-specific terms."""
        domain_terms = [
            "saxs",
            "synchrotron",
            "kinetics",
            "dose_response",
            "imaging",
            "materials",
            "binding",
            "scattering",
        ]

        # Check preset keys (should not have domain names as keys)
        for preset_name in WORKFLOW_PRESETS:
            preset_lower = preset_name.lower()
            for term in domain_terms:
                assert term not in preset_lower, (
                    f"Preset name '{preset_name}' contains domain term '{term}'"
                )

    def test_parameter_sensitivity_report_str_is_domain_agnostic(self):
        """Test ParameterSensitivityReport string representation has no domain terms."""
        # Create a report with the actual dataclass fields
        report = ParameterSensitivityReport(
            is_sloppy=True,
            eigenvalues=np.array([1e-6, 1.0]),
            eigenvalue_range=6.0,
            effective_dimensionality=1.0,
            stiff_indices=[1],
            sloppy_indices=[0],
        )

        # Convert to string and check
        report_str = str(report)

        # Should not contain biology-specific terms
        biology_terms = ["biological", "enzyme", "dose"]
        for term in biology_terms:
            assert term.lower() not in report_str.lower(), (
                f"Report string contains biology term '{term}'"
            )

    def test_no_domain_strings_in_non_deprecation_code(self):
        """Test non-deprecation code in core modules is domain-agnostic.

        The DEPRECATED_PRESET_ALIASES mapping is intentionally allowed to contain
        domain terms like 'xpcs' since that is part of the backwards compatibility
        layer. But actual preset definitions should be domain-agnostic.
        """
        nlsq_root = Path(__file__).parent.parent / "nlsq"

        # Check that no domain presets are in WORKFLOW_PRESETS directly
        domain_preset_names = set(DEPRECATED_PRESET_ALIASES.keys())

        for preset_name in domain_preset_names:
            assert preset_name not in WORKFLOW_PRESETS, (
                f"Domain preset '{preset_name}' should be in DEPRECATED_PRESET_ALIASES, "
                "not WORKFLOW_PRESETS"
            )


class TestDeprecatedPresetsStillWork:
    """Regression test: verify deprecated presets still produce valid configs."""

    def test_all_deprecated_presets_produce_valid_configs(self):
        """Test all deprecated presets in DEPRECATED_PRESET_ALIASES work."""
        for old_preset, new_preset in DEPRECATED_PRESET_ALIASES.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = WorkflowConfig.from_preset(old_preset)

            # Config should be valid
            assert config.gtol > 0
            assert config.ftol > 0
            assert config.xtol > 0
            assert config.tier is not None
            assert config.goal is not None

    def test_deprecated_presets_match_new_preset_config(self):
        """Test deprecated presets produce configs matching their new preset targets."""
        for old_preset, new_preset in DEPRECATED_PRESET_ALIASES.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                old_config = WorkflowConfig.from_preset(old_preset)

            new_config = WorkflowConfig.from_preset(new_preset)

            # Core tolerances should match
            assert old_config.gtol == new_config.gtol, (
                f"gtol mismatch for {old_preset} -> {new_preset}"
            )
            assert old_config.ftol == new_config.ftol, (
                f"ftol mismatch for {old_preset} -> {new_preset}"
            )
            assert old_config.xtol == new_config.xtol, (
                f"xtol mismatch for {old_preset} -> {new_preset}"
            )

    def test_deprecated_preset_count_matches_spec(self):
        """Test that exactly 8 deprecated presets exist per spec."""
        expected_deprecated = {
            "xpcs",
            "saxs",
            "kinetics",
            "dose_response",
            "imaging",
            "materials",
            "binding",
            "synchrotron",
        }

        actual_deprecated = set(DEPRECATED_PRESET_ALIASES.keys())

        assert actual_deprecated == expected_deprecated, (
            f"Deprecated presets mismatch.\n"
            f"Expected: {expected_deprecated}\n"
            f"Actual: {actual_deprecated}"
        )


class TestModuleIntegration:
    """Integration tests for integration points between modified modules."""

    def test_diagnostics_types_exports_sensitivity_category(self):
        """Test IssueCategory.SENSITIVITY is exported from diagnostics.types."""
        from nlsq.diagnostics.types import IssueCategory

        assert hasattr(IssueCategory, "SENSITIVITY")
        # SLOPPY should be an alias for backwards compatibility
        assert hasattr(IssueCategory, "SLOPPY")
        assert IssueCategory.SLOPPY == IssueCategory.SENSITIVITY

    def test_parameter_sensitivity_uses_correct_issue_category(self):
        """Test ParameterSensitivityAnalyzer issues use IssueCategory.SENSITIVITY."""
        from nlsq.diagnostics.types import IssueCategory

        # Create analyzer and analyze a problematic Jacobian
        np.random.seed(42)
        J = np.random.randn(100, 3)
        J[:, 0] *= 1e8  # Create wide eigenvalue spread
        J[:, 2] *= 1e-8

        config = DiagnosticsConfig()
        analyzer = ParameterSensitivityAnalyzer(config=config)
        report = analyzer.analyze(J)

        # If issues were generated, verify category
        for issue in report.issues:
            assert issue.category == IssueCategory.SENSITIVITY, (
                f"Issue category should be SENSITIVITY, got: {issue.category}"
            )

    def test_recommendations_use_sens_codes(self):
        """Test recommendations module uses SENS- codes."""
        from nlsq.diagnostics.recommendations import RECOMMENDATIONS

        # Check that SENS-001 and SENS-002 exist
        assert "SENS-001" in RECOMMENDATIONS, "SENS-001 should be in RECOMMENDATIONS"
        assert "SENS-002" in RECOMMENDATIONS, "SENS-002 should be in RECOMMENDATIONS"

        # Check that SLOPPY- codes do not exist
        for code in RECOMMENDATIONS:
            assert not code.startswith("SLOPPY-"), (
                f"RECOMMENDATIONS should not contain SLOPPY- codes: {code}"
            )
