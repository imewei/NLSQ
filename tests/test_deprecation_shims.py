"""Tests for deprecation shims in NLSQ v0.5.0.

This module tests the deprecation warnings and aliases for:
- SloppyModelAnalyzer -> ParameterSensitivityAnalyzer
- SloppyModelReport -> ParameterSensitivityReport
- Domain-specific preset aliases (xpcs, synchrotron, etc.)
- IssueCategory.SLOPPY -> IssueCategory.SENSITIVITY

All deprecation warnings should specify removal version v0.6.0.
"""

import warnings

import pytest


class TestDiagnosticsClassDeprecationShims:
    """Test deprecation warnings for renamed diagnostics classes."""

    def test_sloppy_model_analyzer_import_raises_deprecation_warning(self):
        """Test that importing SloppyModelAnalyzer raises DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from nlsq.diagnostics import SloppyModelAnalyzer

            # Check that a DeprecationWarning was raised
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

            # Check the warning message contains expected information
            warning_message = str(deprecation_warnings[0].message)
            assert "SloppyModelAnalyzer" in warning_message
            assert "ParameterSensitivityAnalyzer" in warning_message

    def test_sloppy_model_analyzer_is_aliased_to_parameter_sensitivity_analyzer(self):
        """Test that SloppyModelAnalyzer is an alias for ParameterSensitivityAnalyzer."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from nlsq.diagnostics import (
                ParameterSensitivityAnalyzer,
                SloppyModelAnalyzer,
            )

            # Verify they are the same class
            assert SloppyModelAnalyzer is ParameterSensitivityAnalyzer

    def test_sloppy_model_report_import_raises_deprecation_warning(self):
        """Test that importing SloppyModelReport raises DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from nlsq.diagnostics import SloppyModelReport

            # Check that a DeprecationWarning was raised
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

            # Check the warning message contains expected information
            warning_message = str(deprecation_warnings[0].message)
            assert "SloppyModelReport" in warning_message
            assert "ParameterSensitivityReport" in warning_message

    def test_deprecation_warnings_include_removal_version(self):
        """Test that deprecation warnings include removal version v0.6.0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from nlsq.diagnostics import SloppyModelAnalyzer, SloppyModelReport

            # Suppress unused import warnings
            _ = SloppyModelAnalyzer
            _ = SloppyModelReport

            # Check all deprecation warnings include v0.6.0
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            for warning in deprecation_warnings:
                warning_message = str(warning.message)
                assert "v0.6.0" in warning_message, (
                    f"Warning message should include v0.6.0: {warning_message}"
                )


class TestWorkflowPresetDeprecationShims:
    """Test deprecation warnings for domain-specific workflow presets."""

    def test_xpcs_preset_maps_to_precision_standard_with_warning(self):
        """Test that 'xpcs' preset maps to 'precision_standard' with deprecation warning."""
        from nlsq.core.workflow import WorkflowConfig

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = WorkflowConfig.from_preset("xpcs")

            # Check that a DeprecationWarning was raised
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

            # Check the warning message
            warning_message = str(deprecation_warnings[0].message)
            assert "xpcs" in warning_message
            assert "precision_standard" in warning_message

            # Verify the config matches precision_standard
            reference_config = WorkflowConfig.from_preset("precision_standard")
            assert config.gtol == reference_config.gtol
            assert config.ftol == reference_config.ftol
            assert config.xtol == reference_config.xtol
            assert config.enable_multistart == reference_config.enable_multistart

    def test_synchrotron_preset_maps_to_streaming_large_with_warning(self):
        """Test that 'synchrotron' preset maps to 'streaming_large' with deprecation warning."""
        from nlsq.core.workflow import WorkflowConfig

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = WorkflowConfig.from_preset("synchrotron")

            # Check that a DeprecationWarning was raised
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

            # Check the warning message
            warning_message = str(deprecation_warnings[0].message)
            assert "synchrotron" in warning_message
            assert "streaming_large" in warning_message

            # Verify the config matches streaming_large
            reference_config = WorkflowConfig.from_preset("streaming_large")
            assert config.gtol == reference_config.gtol
            assert config.ftol == reference_config.ftol
            assert config.enable_checkpoints == reference_config.enable_checkpoints

    def test_preset_deprecation_warnings_include_removal_version(self):
        """Test that preset deprecation warnings include removal version v0.6.0."""
        from nlsq.core.workflow import WorkflowConfig

        deprecated_presets = ["xpcs", "synchrotron", "saxs", "kinetics"]

        for preset in deprecated_presets:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                WorkflowConfig.from_preset(preset)

                # Check all deprecation warnings include v0.6.0
                deprecation_warnings = [
                    x for x in w if issubclass(x.category, DeprecationWarning)
                ]
                assert len(deprecation_warnings) >= 1, (
                    f"No deprecation warning for preset '{preset}'"
                )

                warning_message = str(deprecation_warnings[0].message)
                assert "v0.6.0" in warning_message, (
                    f"Warning for preset '{preset}' should include v0.6.0: {warning_message}"
                )

    def test_all_deprecated_presets_work_with_warnings(self):
        """Test that all deprecated domain-specific presets work with warnings."""
        from nlsq.core.workflow import DEPRECATED_PRESET_ALIASES, WorkflowConfig

        for old_preset in DEPRECATED_PRESET_ALIASES:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config = WorkflowConfig.from_preset(old_preset)

                # Verify deprecation warning raised
                deprecation_warnings = [
                    x for x in w if issubclass(x.category, DeprecationWarning)
                ]
                assert len(deprecation_warnings) >= 1, (
                    f"No deprecation warning for preset '{old_preset}'"
                )

                # Verify config is valid
                assert config.gtol > 0
                assert config.ftol > 0
                assert config.xtol > 0


class TestIssueCategorySloppyAlias:
    """Test IssueCategory.SLOPPY deprecation alias."""

    def test_issue_category_sloppy_is_alias_for_sensitivity(self):
        """Test that IssueCategory.SLOPPY exists and equals SENSITIVITY."""
        from nlsq.diagnostics.types import IssueCategory

        # SLOPPY should be an alias for SENSITIVITY
        assert hasattr(IssueCategory, "SLOPPY")
        assert IssueCategory.SLOPPY == IssueCategory.SENSITIVITY
