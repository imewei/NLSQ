"""Tests for HybridStreamingConfig dataclass.

This module tests configuration validation and preset profiles for the
adaptive hybrid streaming optimizer.
"""

import pytest

from nlsq.streaming.hybrid_config import HybridStreamingConfig


class TestHybridStreamingConfigDefaults:
    """Test default values initialization."""

    def test_default_initialization(self):
        """Test that config initializes with sensible defaults."""
        config = HybridStreamingConfig()

        # Parameter normalization defaults
        assert config.normalize is True
        assert config.normalization_strategy == "auto"

        # Phase 1: L-BFGS warmup defaults
        assert config.warmup_iterations == 200
        assert config.max_warmup_iterations == 500
        assert config.warmup_learning_rate == 0.001
        assert config.loss_plateau_threshold == 1e-4
        assert config.gradient_norm_threshold == 1e-3
        assert config.active_switching_criteria == ["plateau", "gradient", "max_iter"]

        # L-BFGS configuration defaults
        assert config.lbfgs_history_size == 10
        assert config.lbfgs_initial_step_size == 0.1
        assert config.lbfgs_line_search == "wolfe"

        # Phase 2: Gauss-Newton defaults
        assert config.gauss_newton_max_iterations == 100
        assert config.gauss_newton_tol == 1e-8
        assert config.trust_region_initial == 1.0
        assert config.regularization_factor == 1e-10

        # Streaming defaults
        assert config.chunk_size == 10000

        # Fault tolerance defaults
        assert config.enable_checkpoints is True
        assert config.checkpoint_frequency == 100
        assert config.validate_numerics is True

        # Precision defaults
        assert config.precision == "auto"

        # Multi-device defaults
        assert config.enable_multi_device is False

        # Callback defaults
        assert config.callback_frequency == 10


class TestHybridStreamingConfigPresets:
    """Test preset profile class methods."""

    def test_aggressive_preset(self):
        """Test aggressive profile: faster convergence with L-BFGS, looser tolerances.

        Note: With L-BFGS, fewer warmup iterations are needed compared to Adam
        (5-10x faster convergence), so aggressive now means reduced warmup_iterations.
        """
        config = HybridStreamingConfig.aggressive()

        # L-BFGS achieves faster convergence with fewer iterations
        # 50 iterations with L-BFGS vs 300 iterations with Adam
        assert config.warmup_iterations == 50
        assert config.max_warmup_iterations == 100

        # Higher learning rate for faster progress
        assert config.warmup_learning_rate > 0.001

        # Looser tolerances for faster switching
        assert config.loss_plateau_threshold > 1e-4
        assert config.gradient_norm_threshold > 1e-3
        assert config.gauss_newton_tol > 1e-8

    def test_conservative_preset(self):
        """Test conservative profile: slower but robust, tighter tolerances."""
        config = HybridStreamingConfig.conservative()

        # Less warmup, rely more on Gauss-Newton (L-BFGS enables fewer iterations)
        assert config.warmup_iterations <= 200

        # Lower learning rate for stability
        assert config.warmup_learning_rate <= 0.001

        # Tighter tolerances for higher quality
        assert config.loss_plateau_threshold < 1e-4
        assert config.gradient_norm_threshold < 1e-3
        assert config.gauss_newton_tol < 1e-8

        # More Gauss-Newton iterations
        assert config.gauss_newton_max_iterations > 100

    def test_memory_optimized_preset(self):
        """Test memory_optimized profile: smaller chunks, efficient settings."""
        config = HybridStreamingConfig.memory_optimized()

        # Smaller chunks to reduce memory footprint
        assert config.chunk_size < 10000

        # Conservative warmup to reduce memory allocation
        assert config.warmup_iterations <= 200

        # Enable checkpoints for recovery (memory can be tight)
        assert config.enable_checkpoints is True


class TestHybridStreamingConfigValidation:
    """Test validation in __post_init__ (invalid values raise errors)."""

    def test_invalid_normalization_strategy(self):
        """Test that invalid normalization strategy raises error."""
        with pytest.raises(ValueError, match="normalization_strategy"):
            HybridStreamingConfig(normalization_strategy="invalid")

    def test_invalid_precision(self):
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError, match="precision"):
            HybridStreamingConfig(precision="float16")

    def test_invalid_loop_strategy(self):
        """Test that invalid loop_strategy raises error."""
        with pytest.raises(ValueError, match="loop_strategy"):
            HybridStreamingConfig(loop_strategy="invalid")

    def test_valid_loop_strategies(self):
        """Test that valid loop_strategy values are accepted."""
        for strategy in ("auto", "scan", "loop"):
            config = HybridStreamingConfig(loop_strategy=strategy)
            assert config.loop_strategy == strategy

    def test_warmup_iterations_constraint(self):
        """Test that warmup_iterations must be <= max_warmup_iterations."""
        with pytest.raises(ValueError, match="warmup_iterations"):
            HybridStreamingConfig(warmup_iterations=600, max_warmup_iterations=500)

    def test_negative_values_rejected(self):
        """Test that negative values for positive parameters are rejected."""
        with pytest.raises(ValueError):
            HybridStreamingConfig(warmup_iterations=-10)

        with pytest.raises(ValueError):
            HybridStreamingConfig(chunk_size=-100)

        with pytest.raises(ValueError):
            HybridStreamingConfig(gauss_newton_max_iterations=-5)


class TestHybridStreamingConfigParameterRanges:
    """Test parameter range assertions."""

    def test_learning_rate_positive(self):
        """Test that learning rate must be positive."""
        with pytest.raises(ValueError, match="warmup_learning_rate"):
            HybridStreamingConfig(warmup_learning_rate=0.0)

        with pytest.raises(ValueError, match="warmup_learning_rate"):
            HybridStreamingConfig(warmup_learning_rate=-0.001)

    def test_thresholds_positive(self):
        """Test that thresholds must be positive."""
        with pytest.raises(ValueError):
            HybridStreamingConfig(loss_plateau_threshold=0.0)

        with pytest.raises(ValueError):
            HybridStreamingConfig(gradient_norm_threshold=-1e-3)

        with pytest.raises(ValueError):
            HybridStreamingConfig(gauss_newton_tol=0.0)

    def test_trust_region_positive(self):
        """Test that trust region initial value must be positive."""
        with pytest.raises(ValueError, match="trust_region_initial"):
            HybridStreamingConfig(trust_region_initial=0.0)

        with pytest.raises(ValueError, match="trust_region_initial"):
            HybridStreamingConfig(trust_region_initial=-1.0)

    def test_valid_custom_config(self):
        """Test that valid custom configurations work."""
        config = HybridStreamingConfig(
            normalize=False,
            normalization_strategy="none",
            warmup_iterations=100,
            max_warmup_iterations=300,
            warmup_learning_rate=0.01,
            chunk_size=5000,
            precision="float64",
            enable_multi_device=True,
        )

        assert config.normalize is False
        assert config.normalization_strategy == "none"
        assert config.warmup_iterations == 100
        assert config.max_warmup_iterations == 300
        assert config.warmup_learning_rate == 0.01
        assert config.chunk_size == 5000
        assert config.precision == "float64"
        assert config.enable_multi_device is True
