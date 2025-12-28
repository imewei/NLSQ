"""Tests for workflow configuration infrastructure.

This module tests the workflow enums, dataclasses, and adaptive tolerance
calculation from nlsq/workflow.py (Task Group 1).

Tests cover:
- WorkflowTier enum values and comparison
- OptimizationGoal enum values
- Adaptive tolerance calculation by dataset size
- Tolerance tier shifting for quality/fast goals
- WorkflowConfig dataclass serialization (to_dict/from_dict)

Task Group 3 tests cover:
- WorkflowSelector class and auto_select_workflow function
- Workflow tier selection matrix
- Config class factory methods

Task Group 5 tests cover:
- WORKFLOW_PRESETS dict and from_preset() classmethod
- YAML configuration loading
- Environment variable overrides
- Custom workflow definitions in YAML

Task Group 7 tests cover:
- Automatic checkpoint directory creation with timestamp
- Quality goal enables multi-start
- Validation passes with perturbed parameters
- Validation result comparison and warnings
- precision='auto' integration for quality goal
"""

import os
import shutil
import tempfile
import warnings

import pytest

from nlsq.core.workflow import (
    WORKFLOW_PRESETS,
    DatasetSizeTier,
    MemoryTier,
    OptimizationGoal,
    WorkflowConfig,
    WorkflowSelector,
    WorkflowTier,
    auto_select_workflow,
    calculate_adaptive_tolerances,
    get_custom_workflow,
    get_env_overrides,
    load_config_with_overrides,
    load_yaml_config,
    validate_custom_workflow,
)


class TestWorkflowTier:
    """Tests for WorkflowTier enum."""

    def test_workflow_tier_values_exist(self):
        """Test that WorkflowTier has all expected values."""
        assert hasattr(WorkflowTier, "STANDARD")
        assert hasattr(WorkflowTier, "CHUNKED")
        assert hasattr(WorkflowTier, "STREAMING")
        assert hasattr(WorkflowTier, "STREAMING_CHECKPOINT")

    def test_workflow_tier_comparison(self):
        """Test that WorkflowTier values are distinct and comparable."""
        # Each tier should be unique
        tiers = [
            WorkflowTier.STANDARD,
            WorkflowTier.CHUNKED,
            WorkflowTier.STREAMING,
            WorkflowTier.STREAMING_CHECKPOINT,
        ]
        assert len(set(tiers)) == 4

        # Equality comparison
        assert WorkflowTier.STANDARD == WorkflowTier.STANDARD
        assert WorkflowTier.STANDARD != WorkflowTier.CHUNKED

    def test_workflow_tier_membership(self):
        """Test that WorkflowTier can be checked for membership."""
        assert WorkflowTier.STREAMING in WorkflowTier
        assert "STREAMING" not in WorkflowTier  # String should not be a member


class TestOptimizationGoal:
    """Tests for OptimizationGoal enum."""

    def test_optimization_goal_values_exist(self):
        """Test that OptimizationGoal has all expected values."""
        assert hasattr(OptimizationGoal, "FAST")
        assert hasattr(OptimizationGoal, "ROBUST")
        assert hasattr(OptimizationGoal, "GLOBAL")
        assert hasattr(OptimizationGoal, "MEMORY_EFFICIENT")
        assert hasattr(OptimizationGoal, "QUALITY")

    def test_global_is_alias_for_robust(self):
        """Test that GLOBAL normalizes to ROBUST behavior."""
        # GLOBAL and ROBUST should normalize to ROBUST
        normalized_global = OptimizationGoal.normalize(OptimizationGoal.GLOBAL)
        normalized_robust = OptimizationGoal.normalize(OptimizationGoal.ROBUST)

        assert normalized_global == OptimizationGoal.ROBUST
        assert normalized_robust == OptimizationGoal.ROBUST

    def test_other_goals_not_normalized(self):
        """Test that non-GLOBAL goals are not changed by normalize."""
        for goal in [
            OptimizationGoal.FAST,
            OptimizationGoal.QUALITY,
            OptimizationGoal.MEMORY_EFFICIENT,
        ]:
            assert OptimizationGoal.normalize(goal) == goal


class TestDatasetSizeTier:
    """Tests for DatasetSizeTier enum and from_n_points method."""

    def test_dataset_size_tier_values(self):
        """Test that DatasetSizeTier has correct threshold values."""
        assert DatasetSizeTier.TINY.max_points == 1_000
        assert DatasetSizeTier.SMALL.max_points == 10_000
        assert DatasetSizeTier.MEDIUM.max_points == 100_000
        assert DatasetSizeTier.LARGE.max_points == 1_000_000
        assert DatasetSizeTier.VERY_LARGE.max_points == 10_000_000
        assert DatasetSizeTier.HUGE.max_points == 100_000_000
        assert DatasetSizeTier.MASSIVE.max_points == float("inf")

    def test_dataset_size_tier_tolerances(self):
        """Test that DatasetSizeTier has correct tolerance mappings."""
        assert DatasetSizeTier.TINY.tolerance == 1e-12
        assert DatasetSizeTier.SMALL.tolerance == 1e-10
        assert DatasetSizeTier.MEDIUM.tolerance == 1e-9
        assert DatasetSizeTier.LARGE.tolerance == 1e-8
        assert DatasetSizeTier.VERY_LARGE.tolerance == 1e-7
        assert DatasetSizeTier.HUGE.tolerance == 1e-6
        assert DatasetSizeTier.MASSIVE.tolerance == 1e-5

    def test_from_n_points_boundaries(self):
        """Test from_n_points correctly classifies at boundaries."""
        # Just under each threshold
        assert DatasetSizeTier.from_n_points(999) == DatasetSizeTier.TINY
        assert DatasetSizeTier.from_n_points(9_999) == DatasetSizeTier.SMALL
        assert DatasetSizeTier.from_n_points(99_999) == DatasetSizeTier.MEDIUM
        assert DatasetSizeTier.from_n_points(999_999) == DatasetSizeTier.LARGE
        assert DatasetSizeTier.from_n_points(9_999_999) == DatasetSizeTier.VERY_LARGE
        assert DatasetSizeTier.from_n_points(99_999_999) == DatasetSizeTier.HUGE

        # At or above threshold moves to next tier
        assert DatasetSizeTier.from_n_points(1_000) == DatasetSizeTier.SMALL
        assert DatasetSizeTier.from_n_points(10_000) == DatasetSizeTier.MEDIUM
        assert DatasetSizeTier.from_n_points(100_000_000) == DatasetSizeTier.MASSIVE


class TestMemoryTier:
    """Tests for MemoryTier enum."""

    def test_memory_tier_thresholds(self):
        """Test that MemoryTier has correct threshold values."""
        assert MemoryTier.LOW.max_memory_gb == 16.0
        assert MemoryTier.MEDIUM.max_memory_gb == 64.0
        assert MemoryTier.HIGH.max_memory_gb == 128.0
        assert MemoryTier.VERY_HIGH.max_memory_gb == float("inf")

    def test_from_available_memory_classification(self):
        """Test from_available_memory_gb correctly classifies memory."""
        assert MemoryTier.from_available_memory_gb(8.0) == MemoryTier.LOW
        assert MemoryTier.from_available_memory_gb(15.9) == MemoryTier.LOW
        assert MemoryTier.from_available_memory_gb(16.0) == MemoryTier.MEDIUM
        assert MemoryTier.from_available_memory_gb(32.0) == MemoryTier.MEDIUM
        assert MemoryTier.from_available_memory_gb(64.0) == MemoryTier.HIGH
        assert MemoryTier.from_available_memory_gb(100.0) == MemoryTier.HIGH
        assert MemoryTier.from_available_memory_gb(128.0) == MemoryTier.VERY_HIGH
        assert MemoryTier.from_available_memory_gb(256.0) == MemoryTier.VERY_HIGH


class TestAdaptiveTolerances:
    """Tests for adaptive tolerance calculation."""

    def test_tolerances_by_dataset_size(self):
        """Test that tolerances decrease with dataset size."""
        # Tiny dataset
        tiny_tols = calculate_adaptive_tolerances(500)
        assert tiny_tols["gtol"] == 1e-12
        assert tiny_tols["ftol"] == 1e-12
        assert tiny_tols["xtol"] == 1e-12

        # Large dataset
        large_tols = calculate_adaptive_tolerances(500_000)
        assert large_tols["gtol"] == 1e-8
        assert large_tols["ftol"] == 1e-8
        assert large_tols["xtol"] == 1e-8

        # Very large dataset
        very_large_tols = calculate_adaptive_tolerances(5_000_000)
        assert very_large_tols["gtol"] == 1e-7
        assert very_large_tols["ftol"] == 1e-7
        assert very_large_tols["xtol"] == 1e-7

    def test_quality_goal_shifts_tighter(self):
        """Test that QUALITY goal uses one tier tighter tolerances."""
        # Base tolerance for VERY_LARGE is 1e-7
        base_tols = calculate_adaptive_tolerances(5_000_000)
        quality_tols = calculate_adaptive_tolerances(
            5_000_000, goal=OptimizationGoal.QUALITY
        )

        # Quality should use LARGE tier tolerance (1e-8) instead of VERY_LARGE (1e-7)
        assert quality_tols["gtol"] == 1e-8
        assert quality_tols["gtol"] < base_tols["gtol"]

    def test_fast_goal_shifts_looser(self):
        """Test that FAST goal uses one tier looser tolerances."""
        # Base tolerance for VERY_LARGE is 1e-7
        base_tols = calculate_adaptive_tolerances(5_000_000)
        fast_tols = calculate_adaptive_tolerances(5_000_000, goal=OptimizationGoal.FAST)

        # Fast should use HUGE tier tolerance (1e-6) instead of VERY_LARGE (1e-7)
        assert fast_tols["gtol"] == 1e-6
        assert fast_tols["gtol"] > base_tols["gtol"]

    def test_robust_and_global_use_base_tolerances(self):
        """Test that ROBUST and GLOBAL goals don't shift tolerances."""
        base_tols = calculate_adaptive_tolerances(5_000_000)
        robust_tols = calculate_adaptive_tolerances(
            5_000_000, goal=OptimizationGoal.ROBUST
        )
        global_tols = calculate_adaptive_tolerances(
            5_000_000, goal=OptimizationGoal.GLOBAL
        )

        assert robust_tols["gtol"] == base_tols["gtol"]
        assert global_tols["gtol"] == base_tols["gtol"]

    def test_tier_shift_clamping_at_boundaries(self):
        """Test that tier shifting doesn't go beyond bounds."""
        # TINY dataset with QUALITY goal - can't go tighter, stays at TINY
        tiny_quality = calculate_adaptive_tolerances(500, goal=OptimizationGoal.QUALITY)
        assert tiny_quality["gtol"] == 1e-12  # Stays at tightest

        # MASSIVE dataset with FAST goal - can't go looser, stays at MASSIVE
        massive_fast = calculate_adaptive_tolerances(
            500_000_000, goal=OptimizationGoal.FAST
        )
        assert massive_fast["gtol"] == 1e-5  # Stays at loosest


class TestWorkflowConfigSerialization:
    """Tests for WorkflowConfig to_dict/from_dict serialization."""

    def test_to_dict_converts_enums_to_strings(self):
        """Test that to_dict converts enum values to strings."""
        config = WorkflowConfig(
            tier=WorkflowTier.STREAMING,
            goal=OptimizationGoal.QUALITY,
        )
        d = config.to_dict()

        assert d["tier"] == "STREAMING"
        assert d["goal"] == "QUALITY"
        assert isinstance(d["tier"], str)
        assert isinstance(d["goal"], str)

    def test_from_dict_converts_strings_to_enums(self):
        """Test that from_dict converts string values to enums."""
        d = {
            "tier": "CHUNKED",
            "goal": "FAST",
            "n_starts": 20,
        }
        config = WorkflowConfig.from_dict(d)

        assert config.tier == WorkflowTier.CHUNKED
        assert config.goal == OptimizationGoal.FAST
        assert config.n_starts == 20

    def test_round_trip_serialization(self):
        """Test that to_dict -> from_dict preserves all values."""
        original = WorkflowConfig(
            tier=WorkflowTier.STREAMING_CHECKPOINT,
            goal=OptimizationGoal.MEMORY_EFFICIENT,
            gtol=1e-6,
            ftol=1e-6,
            xtol=1e-6,
            enable_multistart=True,
            n_starts=15,
            sampler="sobol",
            center_on_p0=False,
            scale_factor=0.5,
            memory_limit_gb=32.0,
            chunk_size=50000,
            enable_checkpoints=True,
            checkpoint_dir="/tmp/ckpt",
        )

        d = original.to_dict()
        restored = WorkflowConfig.from_dict(d)

        assert restored.tier == original.tier
        assert restored.goal == original.goal
        assert restored.gtol == original.gtol
        assert restored.ftol == original.ftol
        assert restored.xtol == original.xtol
        assert restored.enable_multistart == original.enable_multistart
        assert restored.n_starts == original.n_starts
        assert restored.sampler == original.sampler
        assert restored.center_on_p0 == original.center_on_p0
        assert restored.scale_factor == original.scale_factor
        assert restored.memory_limit_gb == original.memory_limit_gb
        assert restored.chunk_size == original.chunk_size
        assert restored.enable_checkpoints == original.enable_checkpoints
        assert restored.checkpoint_dir == original.checkpoint_dir

    def test_from_dict_with_defaults(self):
        """Test that from_dict uses defaults for missing keys."""
        d = {"tier": "STANDARD"}  # Minimal dict
        config = WorkflowConfig.from_dict(d)

        assert config.tier == WorkflowTier.STANDARD
        assert config.goal == OptimizationGoal.ROBUST  # Default
        assert config.gtol == 1e-8  # Default
        assert config.enable_multistart is False  # Default
        assert config.n_starts == 10  # Default

    def test_from_dict_accepts_enum_values(self):
        """Test that from_dict accepts enum instances directly."""
        d = {
            "tier": WorkflowTier.CHUNKED,
            "goal": OptimizationGoal.QUALITY,
        }
        config = WorkflowConfig.from_dict(d)

        assert config.tier == WorkflowTier.CHUNKED
        assert config.goal == OptimizationGoal.QUALITY


class TestWorkflowConfigValidation:
    """Tests for WorkflowConfig validation."""

    def test_invalid_sampler_raises(self):
        """Test that invalid sampler raises ValueError."""
        with pytest.raises(ValueError, match="sampler must be one of"):
            WorkflowConfig(sampler="invalid")

    def test_invalid_tolerance_raises(self):
        """Test that non-positive tolerances raise ValueError."""
        with pytest.raises(ValueError, match="gtol must be positive"):
            WorkflowConfig(gtol=-1e-8)

        with pytest.raises(ValueError, match="ftol must be positive"):
            WorkflowConfig(ftol=0)

        with pytest.raises(ValueError, match="xtol must be positive"):
            WorkflowConfig(xtol=-1)

    def test_invalid_n_starts_raises(self):
        """Test that negative n_starts raises ValueError."""
        with pytest.raises(ValueError, match="n_starts must be non-negative"):
            WorkflowConfig(n_starts=-1)

    def test_sampler_case_insensitive(self):
        """Test that sampler is normalized to lowercase."""
        config = WorkflowConfig(sampler="LHS")
        assert config.sampler == "lhs"

        config2 = WorkflowConfig(sampler="SOBOL")
        assert config2.sampler == "sobol"


class TestWorkflowConfigAdaptiveTolerance:
    """Tests for WorkflowConfig.with_adaptive_tolerances method."""

    def test_with_adaptive_tolerances_creates_new_config(self):
        """Test that with_adaptive_tolerances returns new instance."""
        original = WorkflowConfig(goal=OptimizationGoal.QUALITY)
        adapted = original.with_adaptive_tolerances(5_000_000)

        assert adapted is not original
        assert adapted.goal == original.goal

    def test_with_adaptive_tolerances_applies_goal_shift(self):
        """Test that with_adaptive_tolerances applies goal-based shifting."""
        # QUALITY goal with VERY_LARGE dataset should get LARGE tolerances
        config = WorkflowConfig(goal=OptimizationGoal.QUALITY)
        adapted = config.with_adaptive_tolerances(5_000_000)

        assert adapted.gtol == 1e-8  # LARGE tier due to QUALITY shift
        assert adapted.ftol == 1e-8
        assert adapted.xtol == 1e-8


# =============================================================================
# Task Group 3: WorkflowSelector and auto_select_workflow Tests
# =============================================================================


class TestWorkflowSelectorSmallDatasetLowMemory:
    """Test small dataset + low memory -> STANDARD tier."""

    def test_small_dataset_low_memory_returns_standard_tier(self):
        """Test small dataset + low memory -> STANDARD tier.

        Per the workflow matrix:
        | Small (<10K) | Low Memory (<16GB) -> standard
        """
        from nlsq.streaming.large_dataset import LDMemoryConfig

        # Use explicit memory limit to simulate low memory
        selector = WorkflowSelector(memory_limit_gb=8.0)
        config = selector.select(n_points=5_000, n_params=5)

        # Should return LDMemoryConfig (for STANDARD/CHUNKED tier)
        assert isinstance(config, LDMemoryConfig)


class TestWorkflowSelectorLargeDatasetLowMemory:
    """Test large dataset + low memory -> STREAMING tier."""

    def test_large_dataset_low_memory_returns_streaming_tier(self):
        """Test large dataset + low memory -> STREAMING tier.

        Per the workflow matrix:
        | Large (1M-10M) | Low Memory (<16GB) -> streaming
        """
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        # Use explicit memory limit to simulate low memory
        selector = WorkflowSelector(memory_limit_gb=8.0)
        config = selector.select(n_points=5_000_000, n_params=5)

        # Should return HybridStreamingConfig (for STREAMING tier)
        assert isinstance(config, HybridStreamingConfig)


class TestWorkflowSelectorMassiveDataset:
    """Test massive dataset -> STREAMING_CHECKPOINT tier."""

    def test_massive_dataset_returns_streaming_checkpoint_tier(self):
        """Test massive dataset (>100M) -> STREAMING_CHECKPOINT tier.

        Per the workflow matrix:
        | Massive (>100M) | Low/Medium Memory -> streaming+ckpt
        """
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        # Use explicit memory limit to simulate low/medium memory
        selector = WorkflowSelector(memory_limit_gb=32.0)
        config = selector.select(n_points=200_000_000, n_params=5)

        # Should return HybridStreamingConfig with checkpoints enabled
        assert isinstance(config, HybridStreamingConfig)
        assert config.enable_checkpoints is True


class TestWorkflowSelectorQualityGoal:
    """Test quality goal enables multi-start."""

    def test_quality_goal_enables_multistart(self):
        """Test QUALITY goal enables multi-start optimization.

        Per requirements:
        - "quality": One tier tighter tolerances, enable multi-start
        """
        from nlsq.global_optimization import GlobalOptimizationConfig

        # Small dataset + quality goal + high memory should enable multi-start
        selector = WorkflowSelector(memory_limit_gb=200.0)  # Very high memory
        config = selector.select(
            n_points=5_000,
            n_params=5,
            goal=OptimizationGoal.QUALITY,
        )

        # For small dataset with STANDARD tier and multi-start,
        # should return GlobalOptimizationConfig
        assert isinstance(config, GlobalOptimizationConfig)
        assert config.n_starts > 0  # Multi-start enabled


class TestWorkflowSelectorFastGoal:
    """Test fast goal skips multi-start."""

    def test_fast_goal_skips_multistart(self):
        """Test FAST goal skips multi-start optimization.

        Per requirements:
        - "fast": Local optimization only, skip multi-start
        """
        from nlsq.global_optimization import GlobalOptimizationConfig
        from nlsq.streaming.large_dataset import LDMemoryConfig

        # Even with high memory, FAST goal should skip multi-start
        selector = WorkflowSelector(memory_limit_gb=200.0)  # Very high memory
        config = selector.select(
            n_points=5_000,
            n_params=5,
            goal=OptimizationGoal.FAST,
        )

        # Should NOT return GlobalOptimizationConfig (no multi-start)
        assert not isinstance(config, GlobalOptimizationConfig)
        # Should return LDMemoryConfig for small dataset
        assert isinstance(config, LDMemoryConfig)


class TestWorkflowSelectorMemoryEfficientGoal:
    """Test memory_efficient goal prioritizes streaming."""

    def test_memory_efficient_goal_prioritizes_streaming(self):
        """Test MEMORY_EFFICIENT goal prioritizes streaming/chunking.

        Per requirements:
        - "memory_efficient": Minimize memory usage, prioritize streaming/chunking
        """
        from nlsq.streaming.large_dataset import LDMemoryConfig

        # Medium dataset with high memory but memory_efficient goal
        # should still use CHUNKED (not STANDARD)
        selector = WorkflowSelector(memory_limit_gb=100.0)  # High memory
        config = selector.select(
            n_points=500_000,  # Medium dataset
            n_params=5,
            goal=OptimizationGoal.MEMORY_EFFICIENT,
        )

        # Should use CHUNKED tier (returns LDMemoryConfig)
        assert isinstance(config, LDMemoryConfig)


class TestWorkflowSelectorMatrixCombinations:
    """Test workflow matrix selection (2-3 combinations)."""

    def test_medium_dataset_medium_memory_standard(self):
        """Test Medium (10K-1M) + Medium (16-64GB) -> STANDARD.

        Per the workflow matrix:
        | Medium (10K-1M) | Medium (16-64GB) -> standard
        """
        from nlsq.streaming.large_dataset import LDMemoryConfig

        selector = WorkflowSelector(memory_limit_gb=32.0)  # Medium memory
        config = selector.select(n_points=100_000, n_params=5)

        # Should return LDMemoryConfig (for STANDARD tier)
        assert isinstance(config, LDMemoryConfig)

    def test_huge_dataset_high_memory_chunked(self):
        """Test Huge (10M-100M) + High (64-128GB) -> CHUNKED.

        Per the workflow matrix:
        | Huge (10M-100M) | High (64-128GB) -> chunked
        """
        from nlsq.streaming.large_dataset import LDMemoryConfig

        selector = WorkflowSelector(memory_limit_gb=100.0)  # High memory
        config = selector.select(n_points=50_000_000, n_params=5)

        # Should return LDMemoryConfig (for CHUNKED tier)
        assert isinstance(config, LDMemoryConfig)

    def test_large_dataset_very_high_memory_with_multistart(self):
        """Test Large (1M-10M) + Very High (>128GB) -> CHUNKED + multistart.

        Per the workflow matrix:
        | Large (1M-10M) | Very High (>128GB) -> chunked+multistart
        """
        from nlsq.streaming.large_dataset import LDMemoryConfig

        selector = WorkflowSelector(memory_limit_gb=200.0)  # Very high memory
        config = selector.select(n_points=5_000_000, n_params=5)

        # Should return LDMemoryConfig for CHUNKED tier
        # (multi-start is handled at LargeDatasetFitter level for chunked)
        assert isinstance(config, LDMemoryConfig)


class TestWorkflowSelectorReturnTypes:
    """Test return type is correct config class."""

    def test_return_type_ldmemoryconfig_for_chunked(self):
        """Test that CHUNKED tier returns LDMemoryConfig."""
        from nlsq.streaming.large_dataset import LDMemoryConfig

        selector = WorkflowSelector(memory_limit_gb=8.0)  # Low memory
        config = selector.select(
            n_points=100_000,  # Medium dataset
            n_params=5,
        )

        assert isinstance(config, LDMemoryConfig)

    def test_return_type_hybridstreamingconfig_for_streaming(self):
        """Test that STREAMING tier returns HybridStreamingConfig."""
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        selector = WorkflowSelector(memory_limit_gb=8.0)  # Low memory
        config = selector.select(
            n_points=5_000_000,  # Large dataset
            n_params=5,
        )

        assert isinstance(config, HybridStreamingConfig)

    def test_return_type_globaloptimizationconfig_for_multistart(self):
        """Test that multi-start scenarios return GlobalOptimizationConfig."""
        from nlsq.global_optimization import GlobalOptimizationConfig

        selector = WorkflowSelector(memory_limit_gb=200.0)  # Very high memory
        config = selector.select(
            n_points=5_000,  # Small dataset
            n_params=5,
            goal=OptimizationGoal.QUALITY,
        )

        assert isinstance(config, GlobalOptimizationConfig)


class TestAutoSelectWorkflow:
    """Tests for auto_select_workflow convenience function."""

    def test_auto_select_workflow_basic(self):
        """Test auto_select_workflow returns config."""
        config = auto_select_workflow(n_points=5_000, n_params=5)

        # Should return some config object
        assert config is not None

    def test_auto_select_workflow_with_goal(self):
        """Test auto_select_workflow with goal parameter."""
        from nlsq.global_optimization import GlobalOptimizationConfig

        config = auto_select_workflow(
            n_points=5_000,
            n_params=5,
            goal=OptimizationGoal.QUALITY,
            memory_limit_gb=200.0,
        )

        # Quality goal with high memory should return GlobalOptimizationConfig
        assert isinstance(config, GlobalOptimizationConfig)

    def test_auto_select_workflow_with_memory_limit(self):
        """Test auto_select_workflow with memory_limit_gb parameter."""
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        config = auto_select_workflow(
            n_points=50_000_000,
            n_params=5,
            memory_limit_gb=8.0,  # Low memory
        )

        # Large dataset with low memory should return HybridStreamingConfig
        assert isinstance(config, HybridStreamingConfig)


# =============================================================================
# Task Group 5: WORKFLOW_PRESETS and YAML Configuration Tests
# =============================================================================


class TestWorkflowPresetsDict:
    """Tests for WORKFLOW_PRESETS dict contains all expected presets."""

    def test_workflow_presets_contains_all_expected_presets(self):
        """Test WORKFLOW_PRESETS dict contains all expected presets.

        Expected presets include core presets (standard, quality, fast, etc.)
        and scientific application presets (spectroscopy, xpcs, saxs, etc.).
        """
        expected_presets = [
            # Core presets
            "standard",
            "quality",
            "fast",
            "large_robust",
            "streaming",
            "hpc_distributed",
            "memory_efficient",
            # Scientific application presets
            "spectroscopy",
            "xpcs",
            "saxs",
            "kinetics",
            "dose_response",
            "imaging",
            "timeseries",
            "materials",
            "binding",
            "multimodal",
            "synchrotron",
        ]

        # Check all expected presets exist
        for preset_name in expected_presets:
            assert preset_name in WORKFLOW_PRESETS, f"Missing preset: {preset_name}"

        # Verify we have at least 18 presets (7 core + 11 scientific)
        assert len(WORKFLOW_PRESETS) >= 18
        assert len(WORKFLOW_PRESETS) <= 30  # Reasonable upper bound for future growth

    def test_each_preset_has_required_fields(self):
        """Test each preset has required configuration fields."""
        required_fields = ["tier", "goal"]

        for preset_name, preset_config in WORKFLOW_PRESETS.items():
            for field in required_fields:
                assert field in preset_config, (
                    f"Preset '{preset_name}' missing field '{field}'"
                )


class TestStandardPreset:
    """Tests for 'standard' preset returns correct config."""

    def test_standard_preset_returns_correct_config(self):
        """Test 'standard' preset returns correct configuration."""
        config = WorkflowConfig.from_preset("standard")

        # Standard preset should have:
        # - tier: STANDARD
        # - Default tolerances (1e-8)
        # - No multi-start
        assert config.tier == WorkflowTier.STANDARD
        assert config.gtol == 1e-8
        assert config.ftol == 1e-8
        assert config.xtol == 1e-8
        assert config.enable_multistart is False
        assert config.preset == "standard"


class TestQualityPreset:
    """Tests for 'quality' preset has tighter tolerances and multi-start."""

    def test_quality_preset_has_tighter_tolerances_and_multistart(self):
        """Test 'quality' preset has tighter tolerances and multi-start enabled."""
        config = WorkflowConfig.from_preset("quality")

        # Quality preset should have:
        # - Tighter tolerances (1e-10)
        # - Multi-start enabled
        # - More n_starts (20)
        assert config.gtol == 1e-10
        assert config.ftol == 1e-10
        assert config.xtol == 1e-10
        assert config.enable_multistart is True
        assert config.n_starts >= 10  # At least 10 starts
        assert config.preset == "quality"


class TestYAMLConfigLoading:
    """Tests for YAML config loading from ./nlsq.yaml."""

    def test_yaml_config_loading_returns_none_when_file_not_exists(self):
        """Test YAML config loading returns None when file doesn't exist."""
        # Use a non-existent path
        config = load_yaml_config("/nonexistent/path/nlsq.yaml")
        assert config is None

    def test_yaml_config_loading_from_file(self):
        """Test YAML config loading from ./nlsq.yaml."""
        # Check if pyyaml is available
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        # Create a temporary YAML config file
        yaml_content = """
default_workflow: quality
memory_limit_gb: 32.0

workflows:
  my_custom:
    tier: CHUNKED
    goal: QUALITY
    enable_multistart: true
    n_starts: 15
    gtol: 1.0e-9
    ftol: 1.0e-9
    xtol: 1.0e-9
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_yaml_config(temp_path)

            assert config is not None
            assert config.get("default_workflow") == "quality"
            assert config.get("memory_limit_gb") == 32.0
            assert "workflows" in config
            assert "my_custom" in config["workflows"]
        finally:
            os.unlink(temp_path)


class TestEnvironmentVariableOverrides:
    """Tests for environment variable overrides (NLSQ_WORKFLOW_GOAL, NLSQ_MEMORY_LIMIT_GB)."""

    def test_env_override_nlsq_workflow_goal(self):
        """Test NLSQ_WORKFLOW_GOAL environment variable override."""
        # Save original env
        original_goal = os.environ.get("NLSQ_WORKFLOW_GOAL")

        try:
            # Set environment variable
            os.environ["NLSQ_WORKFLOW_GOAL"] = "quality"

            overrides = get_env_overrides()

            assert "goal" in overrides
            assert overrides["goal"] == "QUALITY"
        finally:
            # Restore original env
            if original_goal is not None:
                os.environ["NLSQ_WORKFLOW_GOAL"] = original_goal
            else:
                os.environ.pop("NLSQ_WORKFLOW_GOAL", None)

    def test_env_override_nlsq_memory_limit_gb(self):
        """Test NLSQ_MEMORY_LIMIT_GB environment variable override."""
        # Save original env
        original_memory = os.environ.get("NLSQ_MEMORY_LIMIT_GB")

        try:
            # Set environment variable
            os.environ["NLSQ_MEMORY_LIMIT_GB"] = "16.0"

            overrides = get_env_overrides()

            assert "memory_limit_gb" in overrides
            assert overrides["memory_limit_gb"] == 16.0
        finally:
            # Restore original env
            if original_memory is not None:
                os.environ["NLSQ_MEMORY_LIMIT_GB"] = original_memory
            else:
                os.environ.pop("NLSQ_MEMORY_LIMIT_GB", None)

    def test_env_overrides_take_precedence_over_yaml(self):
        """Test environment variables override YAML settings."""
        # Check if pyyaml is available
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        # Save original env
        original_goal = os.environ.get("NLSQ_WORKFLOW_GOAL")

        # Create a temporary YAML config file with goal=fast
        yaml_content = """
goal: FAST
memory_limit_gb: 64.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Set environment variable to override YAML
            os.environ["NLSQ_WORKFLOW_GOAL"] = "quality"

            config = load_config_with_overrides(temp_path)

            # Environment variable should take precedence
            assert config.get("goal") == "QUALITY"
            # YAML value for memory should still be present
            assert config.get("memory_limit_gb") == 64.0
        finally:
            os.unlink(temp_path)
            if original_goal is not None:
                os.environ["NLSQ_WORKFLOW_GOAL"] = original_goal
            else:
                os.environ.pop("NLSQ_WORKFLOW_GOAL", None)


class TestCustomWorkflowDefinitionInYAML:
    """Tests for custom workflow definition in YAML."""

    def test_custom_workflow_definition_in_yaml(self):
        """Test custom workflow can be defined and loaded from YAML."""
        # Check if pyyaml is available
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        # Create a temporary YAML config file with custom workflow
        yaml_content = """
workflows:
  my_custom_workflow:
    tier: CHUNKED
    goal: QUALITY
    enable_multistart: true
    n_starts: 15
    gtol: 1.0e-9
    ftol: 1.0e-9
    xtol: 1.0e-9
    sampler: sobol
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = get_custom_workflow("my_custom_workflow", temp_path)

            assert config is not None
            assert config.tier == WorkflowTier.CHUNKED
            assert config.goal == OptimizationGoal.QUALITY
            assert config.enable_multistart is True
            assert config.n_starts == 15
            assert config.gtol == 1e-9
            assert config.sampler == "sobol"
            assert config.preset == "custom:my_custom_workflow"
        finally:
            os.unlink(temp_path)

    def test_custom_workflow_returns_none_when_not_found(self):
        """Test get_custom_workflow returns None when workflow not found."""
        # Check if pyyaml is available
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")

        # Create a temporary YAML config file without the requested workflow
        yaml_content = """
workflows:
  other_workflow:
    tier: STANDARD
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = get_custom_workflow("nonexistent_workflow", temp_path)
            assert config is None
        finally:
            os.unlink(temp_path)

    def test_validate_custom_workflow_valid(self):
        """Test validate_custom_workflow returns True for valid workflow."""
        workflow_def = {
            "tier": "CHUNKED",
            "goal": "QUALITY",
            "enable_multistart": True,
            "n_starts": 15,
        }
        is_valid, error = validate_custom_workflow(workflow_def)
        assert is_valid is True
        assert error is None

    def test_validate_custom_workflow_invalid(self):
        """Test validate_custom_workflow returns False for invalid workflow."""
        # Invalid tier
        workflow_def = {"tier": "INVALID_TIER"}
        is_valid, error = validate_custom_workflow(workflow_def)
        assert is_valid is False
        assert error is not None


class TestWorkflowConfigFromPreset:
    """Tests for WorkflowConfig.from_preset() classmethod."""

    def test_from_preset_returns_valid_config(self):
        """Test from_preset returns valid WorkflowConfig for all presets."""
        for preset_name in WORKFLOW_PRESETS:
            config = WorkflowConfig.from_preset(preset_name)

            assert isinstance(config, WorkflowConfig)
            assert config.preset == preset_name

    def test_from_preset_raises_for_invalid_preset(self):
        """Test from_preset raises ValueError for invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            WorkflowConfig.from_preset("nonexistent_preset")

    def test_from_preset_case_insensitive(self):
        """Test from_preset is case insensitive."""
        config1 = WorkflowConfig.from_preset("STANDARD")
        config2 = WorkflowConfig.from_preset("standard")
        config3 = WorkflowConfig.from_preset("Standard")

        assert config1.preset == config2.preset == config3.preset == "standard"

    def test_from_preset_fast(self):
        """Test 'fast' preset has looser tolerances."""
        config = WorkflowConfig.from_preset("fast")

        assert config.gtol == 1e-6
        assert config.enable_multistart is False

    def test_from_preset_large_robust(self):
        """Test 'large_robust' preset has chunked tier and multi-start."""
        config = WorkflowConfig.from_preset("large_robust")

        assert config.tier == WorkflowTier.CHUNKED
        assert config.enable_multistart is True

    def test_from_preset_streaming(self):
        """Test 'streaming' preset has streaming tier."""
        config = WorkflowConfig.from_preset("streaming")

        assert config.tier == WorkflowTier.STREAMING

    def test_from_preset_hpc_distributed(self):
        """Test 'hpc_distributed' preset has checkpoints and streaming."""
        config = WorkflowConfig.from_preset("hpc_distributed")

        assert config.tier == WorkflowTier.STREAMING_CHECKPOINT
        assert config.enable_checkpoints is True
        assert config.enable_multistart is True


class TestWorkflowConfigWithOverrides:
    """Tests for WorkflowConfig.with_overrides() method."""

    def test_with_overrides_creates_new_config(self):
        """Test with_overrides returns new config instance."""
        original = WorkflowConfig.from_preset("standard")
        modified = original.with_overrides(n_starts=20)

        assert modified is not original
        assert modified.n_starts == 20
        assert original.n_starts != 20  # Original unchanged

    def test_with_overrides_clears_preset(self):
        """Test with_overrides clears preset name."""
        original = WorkflowConfig.from_preset("standard")
        modified = original.with_overrides(enable_multistart=True)

        assert original.preset == "standard"
        assert modified.preset is None


# =============================================================================
# Task Group 7: Checkpointing and Quality Goal Implementation Tests
# =============================================================================


class TestCheckpointDirectoryCreation:
    """Tests for automatic checkpoint directory creation with timestamp."""

    def test_create_checkpoint_directory_creates_timestamped_dir(self):
        """Test automatic checkpoint directory creation with timestamp format."""
        from nlsq.core.workflow import create_checkpoint_directory

        # Create checkpoint directory
        checkpoint_dir = create_checkpoint_directory()

        try:
            # Verify directory was created
            assert os.path.exists(checkpoint_dir)
            assert os.path.isdir(checkpoint_dir)

            # Verify path format: ./nlsq_checkpoints/YYYYMMDD_HHMMSS/
            assert "nlsq_checkpoints" in checkpoint_dir
            # The directory name should match YYYYMMDD_HHMMSS pattern
            dir_name = os.path.basename(checkpoint_dir)
            assert len(dir_name) == 15  # YYYYMMDD_HHMMSS
            assert dir_name[8] == "_"  # Separator between date and time
        finally:
            # Cleanup (guard against race conditions in parallel tests)
            try:
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                # Also clean up parent if empty
                parent = os.path.dirname(checkpoint_dir)
                if os.path.exists(parent) and not os.listdir(parent):
                    os.rmdir(parent)
            except (FileNotFoundError, OSError):
                pass  # Directory already removed by another test or process

    def test_checkpoint_directory_path_format(self):
        """Test checkpoint directory path format (./nlsq_checkpoints/YYYYMMDD_HHMMSS/)."""
        from nlsq.core.workflow import create_checkpoint_directory

        checkpoint_dir = create_checkpoint_directory()

        try:
            # Path should contain nlsq_checkpoints
            assert "nlsq_checkpoints" in checkpoint_dir

            # Extract timestamp portion
            dir_name = os.path.basename(checkpoint_dir)

            # Verify date portion (first 8 characters) are digits
            date_part = dir_name[:8]
            assert date_part.isdigit()

            # Verify time portion (last 6 characters) are digits
            time_part = dir_name[9:]
            assert time_part.isdigit()
        finally:
            # Cleanup (guard against race conditions in parallel tests)
            try:
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                parent = os.path.dirname(checkpoint_dir)
                if os.path.exists(parent) and not os.listdir(parent):
                    os.rmdir(parent)
            except (FileNotFoundError, OSError):
                pass  # Directory already removed by another test or process


class TestQualityGoalMultiStart:
    """Tests for quality goal enabling multi-start."""

    def test_quality_goal_enables_multistart_via_workflow_selector(self):
        """Test quality goal enables multi-start via WorkflowSelector."""
        from nlsq.global_optimization import GlobalOptimizationConfig

        # Quality goal should enable multi-start
        selector = WorkflowSelector(memory_limit_gb=200.0)  # High memory
        config = selector.select(
            n_points=5_000,
            n_params=5,
            goal=OptimizationGoal.QUALITY,
        )

        # Should return GlobalOptimizationConfig with multi-start
        assert isinstance(config, GlobalOptimizationConfig)
        assert config.n_starts > 0

    def test_quality_goal_configures_n_starts_based_on_dataset_size(self):
        """Test quality goal configures n_starts based on dataset size."""
        from nlsq.core.workflow import get_quality_n_starts

        # Smaller datasets can afford more starts
        small_n_starts = get_quality_n_starts(n_points=5_000)
        medium_n_starts = get_quality_n_starts(n_points=500_000)
        large_n_starts = get_quality_n_starts(n_points=5_000_000)

        # n_starts should generally decrease or stay same with dataset size
        assert small_n_starts >= large_n_starts
        assert all(n >= 5 for n in [small_n_starts, medium_n_starts, large_n_starts])


class TestValidationPasses:
    """Tests for validation passes with perturbed parameters."""

    def test_quality_goal_runs_validation_passes(self):
        """Test quality goal runs validation passes (perturbed parameters)."""
        from nlsq.core.workflow import generate_perturbed_parameters

        # Initial parameters
        p0 = [1.0, 2.0, 3.0]

        # Generate perturbed parameters (3-5 perturbations)
        perturbed = generate_perturbed_parameters(p0, n_perturbations=5)

        # Should generate 5 perturbed parameter sets
        assert len(perturbed) == 5

        # Each perturbation should be different from p0
        for p in perturbed:
            assert len(p) == len(p0)
            # At least one parameter should be different
            assert any(p[i] != p0[i] for i in range(len(p0)))

    def test_validation_pass_compares_results(self):
        """Test validation pass compares results and warns on divergence."""
        from nlsq.core.workflow import compare_validation_results

        # Converging results (similar parameters)
        results_converging = [
            {"popt": [1.0, 2.0, 3.0], "cost": 0.01},
            {"popt": [1.01, 2.01, 3.01], "cost": 0.012},
            {"popt": [0.99, 1.99, 2.99], "cost": 0.011},
        ]

        # Diverging results (>10% difference)
        results_diverging = [
            {"popt": [1.0, 2.0, 3.0], "cost": 0.01},
            {"popt": [1.5, 2.5, 3.5], "cost": 0.015},  # >10% different
            {"popt": [0.5, 1.5, 2.5], "cost": 0.02},  # >10% different
        ]

        # Converging results should not warn
        is_converged, warning_msg = compare_validation_results(results_converging)
        assert is_converged is True
        assert warning_msg is None

        # Diverging results should warn
        is_converged, warning_msg = compare_validation_results(results_diverging)
        assert is_converged is False
        assert warning_msg is not None
        assert "diverge" in warning_msg.lower() or "differ" in warning_msg.lower()

    def test_validation_warns_on_divergence(self):
        """Test validation pass warns when solutions diverge significantly (>10%)."""
        from nlsq.core.workflow import compare_validation_results

        # Results that diverge by more than 10%
        results = [
            {"popt": [1.0, 2.0], "cost": 0.01},
            {"popt": [1.2, 2.2], "cost": 0.015},  # 20% different on param 0
        ]

        is_converged, warning_msg = compare_validation_results(results)

        # Should detect divergence
        assert is_converged is False
        assert warning_msg is not None


class TestPrecisionAutoIntegration:
    """Tests for precision='auto' integration with quality goal."""

    def test_precision_auto_integration_for_quality(self):
        """Test precision='auto' integration for float32->float64 upgrade."""
        from nlsq.core.workflow import get_quality_precision_config

        # Quality goal should recommend auto precision
        precision_config = get_quality_precision_config()

        assert precision_config["precision"] == "auto"
        # Auto precision leverages float32->float64 upgrade when needed
        assert "description" in precision_config
        assert (
            "float64" in precision_config["description"].lower()
            or "auto" in precision_config["description"].lower()
        )
