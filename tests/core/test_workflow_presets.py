"""Tests for generic workflow presets (domain-agnostic).

This module tests the new generic presets that replace domain-specific presets:
- precision_high: tolerances 1e-10, STANDARD tier, QUALITY goal
- precision_standard: tolerances 1e-8, STANDARD tier, ROBUST goal
- streaming_large: STREAMING tier, checkpoints enabled, checkpoint_frequency=100
- global_multimodal: multi-start enabled, n_starts=30, sampler='sobol', GLOBAL goal

These tests verify the domain-agnostic preset configurations work correctly.
"""

import warnings

import pytest

from nlsq.core.workflow import (
    DEPRECATED_PRESET_ALIASES,
    WORKFLOW_PRESETS,
    OptimizationGoal,
    WorkflowConfig,
    WorkflowTier,
)


class TestPrecisionHighPreset:
    """Tests for 'precision_high' preset configuration."""

    def test_precision_high_has_tolerances_1e10(self):
        """Test precision_high preset has tolerances 1e-10."""
        config = WorkflowConfig.from_preset("precision_high")

        assert config.gtol == 1e-10
        assert config.ftol == 1e-10
        assert config.xtol == 1e-10

    def test_precision_high_has_standard_tier(self):
        """Test precision_high preset has STANDARD tier."""
        config = WorkflowConfig.from_preset("precision_high")

        assert config.tier == WorkflowTier.STANDARD

    def test_precision_high_has_quality_goal(self):
        """Test precision_high preset has QUALITY goal."""
        config = WorkflowConfig.from_preset("precision_high")

        assert config.goal == OptimizationGoal.QUALITY


class TestPrecisionStandardPreset:
    """Tests for 'precision_standard' preset configuration."""

    def test_precision_standard_has_tolerances_1e8(self):
        """Test precision_standard preset has tolerances 1e-8."""
        config = WorkflowConfig.from_preset("precision_standard")

        assert config.gtol == 1e-8
        assert config.ftol == 1e-8
        assert config.xtol == 1e-8

    def test_precision_standard_has_standard_tier(self):
        """Test precision_standard preset has STANDARD tier."""
        config = WorkflowConfig.from_preset("precision_standard")

        assert config.tier == WorkflowTier.STANDARD

    def test_precision_standard_has_robust_goal(self):
        """Test precision_standard preset has ROBUST goal."""
        config = WorkflowConfig.from_preset("precision_standard")

        assert config.goal == OptimizationGoal.ROBUST


class TestStreamingLargePreset:
    """Tests for 'streaming_large' preset configuration."""

    def test_streaming_large_has_streaming_tier(self):
        """Test streaming_large preset has STREAMING tier."""
        config = WorkflowConfig.from_preset("streaming_large")

        assert config.tier == WorkflowTier.STREAMING

    def test_streaming_large_has_checkpoints_enabled(self):
        """Test streaming_large preset has checkpoints enabled."""
        config = WorkflowConfig.from_preset("streaming_large")

        assert config.enable_checkpoints is True

    def test_streaming_large_preset_dict_has_checkpoint_frequency(self):
        """Test streaming_large preset dict has checkpoint_frequency=100."""
        preset = WORKFLOW_PRESETS["streaming_large"]

        assert preset.get("checkpoint_frequency") == 100


class TestGlobalMultimodalPreset:
    """Tests for 'global_multimodal' preset configuration."""

    def test_global_multimodal_has_multistart_enabled(self):
        """Test global_multimodal preset has multi-start enabled."""
        config = WorkflowConfig.from_preset("global_multimodal")

        assert config.enable_multistart is True

    def test_global_multimodal_has_30_starts(self):
        """Test global_multimodal preset has n_starts=30."""
        config = WorkflowConfig.from_preset("global_multimodal")

        assert config.n_starts == 30

    def test_global_multimodal_has_sobol_sampler(self):
        """Test global_multimodal preset has Sobol sampling."""
        config = WorkflowConfig.from_preset("global_multimodal")

        assert config.sampler == "sobol"

    def test_global_multimodal_has_global_goal(self):
        """Test global_multimodal preset has GLOBAL goal."""
        config = WorkflowConfig.from_preset("global_multimodal")

        assert config.goal == OptimizationGoal.GLOBAL


class TestPresetValidation:
    """Tests for preset validation behavior."""

    def test_unknown_preset_raises_valueerror(self):
        """Test preset validation rejects unknown preset names."""
        with pytest.raises(ValueError, match="Unknown preset"):
            WorkflowConfig.from_preset("unknown_preset_name")

    def test_deprecated_domain_presets_work_with_warning(self):
        """Test deprecated domain presets work with deprecation warnings.

        Per the spec, deprecated presets should still work but emit
        DeprecationWarning. They should NOT raise ValueError.
        """
        deprecated_presets = list(DEPRECATED_PRESET_ALIASES.keys())

        for preset_name in deprecated_presets:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                config = WorkflowConfig.from_preset(preset_name)

                # Should produce a valid config, not raise ValueError
                assert config is not None
                assert config.gtol > 0

                # Should emit deprecation warning
                deprecation_warnings = [
                    x for x in w if issubclass(x.category, DeprecationWarning)
                ]
                assert len(deprecation_warnings) >= 1, (
                    f"No deprecation warning for preset '{preset_name}'"
                )

    def test_deprecated_presets_not_in_workflow_presets(self):
        """Test deprecated presets are not in WORKFLOW_PRESETS dict directly.

        The old domain presets should be removed from WORKFLOW_PRESETS,
        but still accessible via DEPRECATED_PRESET_ALIASES.
        """
        deprecated_presets = [
            "xpcs",
            "saxs",
            "kinetics",
            "dose_response",
            "imaging",
            "materials",
            "binding",
            "synchrotron",
        ]

        for preset_name in deprecated_presets:
            assert preset_name not in WORKFLOW_PRESETS, (
                f"Deprecated preset '{preset_name}' should not be in WORKFLOW_PRESETS"
            )


class TestFromPresetReturnsWorkflowConfig:
    """Tests for from_preset() returning correct WorkflowConfig instances."""

    def test_from_preset_returns_workflowconfig_instance(self):
        """Test from_preset() returns correct WorkflowConfig instances."""
        new_presets = [
            "precision_high",
            "precision_standard",
            "streaming_large",
            "global_multimodal",
        ]

        for preset_name in new_presets:
            config = WorkflowConfig.from_preset(preset_name)

            assert isinstance(config, WorkflowConfig)
            assert config.preset == preset_name

    def test_new_presets_in_workflow_presets_dict(self):
        """Test new presets are in WORKFLOW_PRESETS dict."""
        new_presets = [
            "precision_high",
            "precision_standard",
            "streaming_large",
            "global_multimodal",
        ]

        for preset_name in new_presets:
            assert preset_name in WORKFLOW_PRESETS, f"Missing preset: {preset_name}"


class TestExistingGenericPresetsUnchanged:
    """Tests that existing generic presets remain unchanged."""

    def test_existing_generic_presets_still_exist(self):
        """Test existing generic presets still exist in WORKFLOW_PRESETS."""
        existing_generic_presets = [
            "standard",
            "quality",
            "fast",
            "large_robust",
            "streaming",
            "hpc_distributed",
            "memory_efficient",
            "spectroscopy",
            "timeseries",
            "multimodal",
        ]

        for preset_name in existing_generic_presets:
            assert preset_name in WORKFLOW_PRESETS, (
                f"Missing existing preset: {preset_name}"
            )

    def test_standard_preset_unchanged(self):
        """Test standard preset configuration is unchanged."""
        config = WorkflowConfig.from_preset("standard")

        assert config.tier == WorkflowTier.STANDARD
        assert config.goal == OptimizationGoal.ROBUST
        assert config.gtol == 1e-8
        assert config.enable_multistart is False
