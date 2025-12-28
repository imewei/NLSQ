"""Tests for L-BFGS configuration parameters in HybridStreamingConfig.

This module tests the L-BFGS warmup configuration additions including
parameter validation, preset updates, and backward compatibility.

Task 1.1: Write 4-6 focused tests for L-BFGS configuration.
"""

import pytest

from nlsq.streaming.hybrid_config import HybridStreamingConfig


class TestLbfgsHistorySizeValidation:
    """Test lbfgs_history_size parameter validation."""

    def test_lbfgs_history_size_default(self):
        """Test that lbfgs_history_size defaults to 10 (standard from SciPy/PyTorch)."""
        config = HybridStreamingConfig()
        assert config.lbfgs_history_size == 10

    def test_lbfgs_history_size_positive_integer(self):
        """Test that lbfgs_history_size accepts positive integers."""
        config = HybridStreamingConfig(lbfgs_history_size=20)
        assert config.lbfgs_history_size == 20

    def test_lbfgs_history_size_zero_rejected(self):
        """Test that zero lbfgs_history_size is rejected."""
        with pytest.raises(ValueError, match="lbfgs_history_size"):
            HybridStreamingConfig(lbfgs_history_size=0)

    def test_lbfgs_history_size_negative_rejected(self):
        """Test that negative lbfgs_history_size is rejected."""
        with pytest.raises(ValueError, match="lbfgs_history_size"):
            HybridStreamingConfig(lbfgs_history_size=-5)

    def test_lbfgs_history_size_reasonable_range(self):
        """Test that lbfgs_history_size within reasonable range is accepted."""
        # Small value (minimum reasonable)
        config_small = HybridStreamingConfig(lbfgs_history_size=3)
        assert config_small.lbfgs_history_size == 3

        # Large but reasonable value
        config_large = HybridStreamingConfig(lbfgs_history_size=50)
        assert config_large.lbfgs_history_size == 50


class TestLbfgsInitialStepSizeValidation:
    """Test lbfgs_initial_step_size parameter validation."""

    def test_lbfgs_initial_step_size_default(self):
        """Test that lbfgs_initial_step_size defaults to 0.1 (cold start scaffolding)."""
        config = HybridStreamingConfig()
        assert config.lbfgs_initial_step_size == 0.1

    def test_lbfgs_initial_step_size_positive_float(self):
        """Test that lbfgs_initial_step_size accepts positive floats."""
        config = HybridStreamingConfig(lbfgs_initial_step_size=0.5)
        assert config.lbfgs_initial_step_size == 0.5

    def test_lbfgs_initial_step_size_zero_rejected(self):
        """Test that zero lbfgs_initial_step_size is rejected."""
        with pytest.raises(ValueError, match="lbfgs_initial_step_size"):
            HybridStreamingConfig(lbfgs_initial_step_size=0.0)

    def test_lbfgs_initial_step_size_negative_rejected(self):
        """Test that negative lbfgs_initial_step_size is rejected."""
        with pytest.raises(ValueError, match="lbfgs_initial_step_size"):
            HybridStreamingConfig(lbfgs_initial_step_size=-0.1)

    def test_lbfgs_initial_step_size_large_value(self):
        """Test that large lbfgs_initial_step_size values are accepted."""
        config = HybridStreamingConfig(lbfgs_initial_step_size=1.0)
        assert config.lbfgs_initial_step_size == 1.0


class TestLbfgsLineSearchValidation:
    """Test lbfgs_line_search parameter validation."""

    def test_lbfgs_line_search_default(self):
        """Test that lbfgs_line_search defaults to 'wolfe'."""
        config = HybridStreamingConfig()
        assert config.lbfgs_line_search == "wolfe"

    def test_lbfgs_line_search_valid_values(self):
        """Test that valid lbfgs_line_search values are accepted."""
        for value in ("wolfe", "strong_wolfe", "backtracking"):
            config = HybridStreamingConfig(lbfgs_line_search=value)
            assert config.lbfgs_line_search == value

    def test_lbfgs_line_search_invalid_rejected(self):
        """Test that invalid lbfgs_line_search is rejected."""
        with pytest.raises(ValueError, match="lbfgs_line_search"):
            HybridStreamingConfig(lbfgs_line_search="invalid")


class TestPresetLbfgsUpdates:
    """Test preset updates return expected L-BFGS values."""

    def test_aggressive_preset_warmup_iterations(self):
        """Test aggressive preset has reduced warmup iterations for L-BFGS."""
        config = HybridStreamingConfig.aggressive()
        # L-BFGS converges 5-10x faster, so warmup iterations should be ~50-100
        assert config.warmup_iterations == 50
        assert config.max_warmup_iterations == 100

    def test_conservative_preset_warmup_iterations(self):
        """Test conservative preset has reduced warmup iterations for L-BFGS."""
        config = HybridStreamingConfig.conservative()
        # Conservative still uses less warmup than default with L-BFGS
        assert config.warmup_iterations == 30
        assert config.max_warmup_iterations == 80

    def test_memory_optimized_preset_warmup_iterations(self):
        """Test memory_optimized preset has reduced warmup iterations for L-BFGS."""
        config = HybridStreamingConfig.memory_optimized()
        assert config.warmup_iterations == 40
        assert config.max_warmup_iterations == 100

    def test_scientific_default_preset_warmup_iterations(self):
        """Test scientific_default preset has updated warmup iterations for L-BFGS."""
        config = HybridStreamingConfig.scientific_default()
        # Scientific default should have moderate warmup iterations
        assert config.warmup_iterations <= 50
        assert config.max_warmup_iterations <= 150

    def test_defense_strict_preset_warmup_iterations(self):
        """Test defense_strict preset has reduced warmup iterations for L-BFGS."""
        config = HybridStreamingConfig.defense_strict()
        assert config.warmup_iterations == 25
        assert config.max_warmup_iterations == 60

    def test_defense_relaxed_preset_warmup_iterations(self):
        """Test defense_relaxed preset has reduced warmup iterations for L-BFGS."""
        config = HybridStreamingConfig.defense_relaxed()
        assert config.warmup_iterations == 50
        assert config.max_warmup_iterations == 120


class TestLbfgsLayer2StepSizes:
    """Test Layer 2 threshold configurations for L-BFGS initial step size."""

    def test_lbfgs_exploration_step_size_default(self):
        """Test lbfgs_exploration_step_size defaults to 0.1 to prevent overshoot."""
        config = HybridStreamingConfig()
        assert config.lbfgs_exploration_step_size == 0.1

    def test_lbfgs_refinement_step_size_default(self):
        """Test lbfgs_refinement_step_size defaults to 1.0 for near-Newton speed."""
        config = HybridStreamingConfig()
        assert config.lbfgs_refinement_step_size == 1.0

    def test_lbfgs_step_sizes_positive(self):
        """Test that lbfgs step size parameters must be positive."""
        with pytest.raises(ValueError, match="lbfgs_exploration_step_size"):
            HybridStreamingConfig(lbfgs_exploration_step_size=0.0)

        with pytest.raises(ValueError, match="lbfgs_refinement_step_size"):
            HybridStreamingConfig(lbfgs_refinement_step_size=-0.5)


class TestBackwardCompatibility:
    """Test backward compatibility - existing config still works."""

    def test_existing_config_parameters_unchanged(self):
        """Test that all existing config parameters still work."""
        config = HybridStreamingConfig(
            # Phase 0: Parameter normalization
            normalize=True,
            normalization_strategy="auto",
            # Phase 1: Adam warmup (existing)
            warmup_iterations=200,
            max_warmup_iterations=500,
            warmup_learning_rate=0.001,
            loss_plateau_threshold=1e-4,
            gradient_norm_threshold=1e-3,
            # 4-Layer Defense Strategy
            enable_warm_start_detection=True,
            warm_start_threshold=0.01,
            enable_adaptive_warmup_lr=True,
            warmup_lr_refinement=1e-6,
            warmup_lr_careful=1e-5,
            enable_cost_guard=True,
            cost_increase_tolerance=0.05,
            enable_step_clipping=True,
            max_warmup_step_size=0.1,
            # Phase 2: Gauss-Newton
            gauss_newton_max_iterations=100,
            gauss_newton_tol=1e-8,
            trust_region_initial=1.0,
            regularization_factor=1e-10,
            # Streaming configuration
            chunk_size=10000,
            loop_strategy="auto",
            # Other settings
            precision="float64",
            enable_multi_device=False,
        )

        # All existing parameters should be accessible
        assert config.normalize is True
        assert config.warmup_iterations == 200
        assert config.enable_warm_start_detection is True
        assert config.gauss_newton_max_iterations == 100
        assert config.chunk_size == 10000

    def test_existing_presets_return_valid_config(self):
        """Test that all existing preset methods still return valid configs."""
        # All presets should still work without errors
        presets = [
            HybridStreamingConfig.aggressive(),
            HybridStreamingConfig.conservative(),
            HybridStreamingConfig.memory_optimized(),
            HybridStreamingConfig.scientific_default(),
            HybridStreamingConfig.defense_strict(),
            HybridStreamingConfig.defense_relaxed(),
            HybridStreamingConfig.defense_disabled(),
            HybridStreamingConfig.with_multistart(n_starts=5),
        ]

        for preset in presets:
            assert isinstance(preset, HybridStreamingConfig)
            # New L-BFGS parameters should be present
            assert hasattr(preset, "lbfgs_history_size")
            assert hasattr(preset, "lbfgs_initial_step_size")
            assert hasattr(preset, "lbfgs_line_search")

    def test_default_config_has_new_lbfgs_params(self):
        """Test that default config includes new L-BFGS parameters with defaults."""
        config = HybridStreamingConfig()

        # New L-BFGS parameters should exist with correct defaults
        assert config.lbfgs_history_size == 10
        assert config.lbfgs_initial_step_size == 0.1
        assert config.lbfgs_line_search == "wolfe"
        assert config.lbfgs_exploration_step_size == 0.1
        assert config.lbfgs_refinement_step_size == 1.0
