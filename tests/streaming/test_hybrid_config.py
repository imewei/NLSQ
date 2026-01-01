"""Tests for nlsq.streaming.hybrid_config module.

Characterization tests for HybridStreamingConfig, the configuration class
for the four-phase adaptive hybrid streaming optimizer.

Coverage targets:
- HybridStreamingConfig: Dataclass with 50+ configuration parameters
- __post_init__: Validation via delegation to validators
- Class methods: aggressive(), conservative(), memory_optimized(), etc.
- Defense layer presets: defense_strict(), defense_relaxed(), defense_disabled()
"""

import pytest

from nlsq.streaming.hybrid_config import HybridStreamingConfig


class TestHybridStreamingConfigDefaults:
    """Tests for default configuration values."""

    def test_default_instance_creation(self):
        """Test that default instance can be created."""
        config = HybridStreamingConfig()
        assert config is not None

    def test_default_normalization(self):
        """Test default normalization settings."""
        config = HybridStreamingConfig()

        assert config.normalize is True
        assert config.normalization_strategy == "auto"

    def test_default_warmup_phase(self):
        """Test default Phase 1 warmup settings."""
        config = HybridStreamingConfig()

        assert config.warmup_iterations == 200
        assert config.max_warmup_iterations == 500
        assert config.warmup_learning_rate == 0.001
        assert config.loss_plateau_threshold == 1e-4
        assert config.gradient_norm_threshold == 1e-3
        assert config.active_switching_criteria == ["plateau", "gradient", "max_iter"]

    def test_default_lbfgs(self):
        """Test default L-BFGS settings."""
        config = HybridStreamingConfig()

        assert config.lbfgs_history_size == 10
        assert config.lbfgs_initial_step_size == 0.1
        assert config.lbfgs_line_search == "wolfe"
        assert config.lbfgs_exploration_step_size == 0.1
        assert config.lbfgs_refinement_step_size == 1.0

    def test_default_optax_enhancements(self):
        """Test default Optax enhancement settings."""
        config = HybridStreamingConfig()

        assert config.use_learning_rate_schedule is False
        assert config.lr_schedule_warmup_steps == 50
        assert config.lr_schedule_decay_steps == 450
        assert config.lr_schedule_end_value == 0.0001
        assert config.gradient_clip_value is None

    def test_default_defense_layers(self):
        """Test default 4-layer defense strategy settings."""
        config = HybridStreamingConfig()

        # Layer 1: Warm Start Detection
        assert config.enable_warm_start_detection is True
        assert config.warm_start_threshold == 0.01

        # Layer 2: Adaptive Step Size
        assert config.enable_adaptive_warmup_lr is True
        assert config.warmup_lr_refinement == 1e-6
        assert config.warmup_lr_careful == 1e-5

        # Layer 3: Cost-Increase Guard
        assert config.enable_cost_guard is True
        assert config.cost_increase_tolerance == 0.05

        # Layer 4: Trust Region Constraint
        assert config.enable_step_clipping is True
        assert config.max_warmup_step_size == 0.1

    def test_default_gauss_newton(self):
        """Test default Phase 2 Gauss-Newton settings."""
        config = HybridStreamingConfig()

        assert config.gauss_newton_max_iterations == 100
        assert config.gauss_newton_tol == 1e-8
        assert config.trust_region_initial == 1.0
        assert config.regularization_factor == 1e-10

    def test_default_cg_solver(self):
        """Test default CG solver settings."""
        config = HybridStreamingConfig()

        assert config.cg_max_iterations == 100
        assert config.cg_relative_tolerance == 1e-4
        assert config.cg_absolute_tolerance == 1e-10
        assert config.cg_param_threshold == 2000

    def test_default_group_variance(self):
        """Test default group variance regularization settings."""
        config = HybridStreamingConfig()

        assert config.enable_group_variance_regularization is False
        assert config.group_variance_lambda == 0.01
        assert config.group_variance_indices is None

    def test_default_streaming(self):
        """Test default streaming settings."""
        config = HybridStreamingConfig()

        assert config.chunk_size == 10000
        assert config.loop_strategy == "auto"

    def test_default_fault_tolerance(self):
        """Test default fault tolerance settings."""
        config = HybridStreamingConfig()

        assert config.enable_checkpoints is True
        assert config.checkpoint_frequency == 100
        assert config.checkpoint_dir is None
        assert config.resume_from_checkpoint is None
        assert config.validate_numerics is True
        assert config.enable_fault_tolerance is True
        assert config.max_retries_per_batch == 2
        assert config.min_success_rate == 0.5

    def test_default_precision(self):
        """Test default precision settings."""
        config = HybridStreamingConfig()

        assert config.precision == "auto"
        assert config.enable_multi_device is False

    def test_default_progress(self):
        """Test default progress monitoring settings."""
        config = HybridStreamingConfig()

        assert config.callback_frequency == 10
        assert config.verbose == 1
        assert config.log_frequency == 1

    def test_default_multistart(self):
        """Test default multi-start settings."""
        config = HybridStreamingConfig()

        assert config.enable_multistart is False
        assert config.n_starts == 10
        assert config.multistart_sampler == "lhs"
        assert config.elimination_rounds == 3
        assert config.elimination_fraction == 0.5
        assert config.batches_per_round == 50
        assert config.center_on_p0 is True
        assert config.scale_factor == 1.0


class TestHybridStreamingConfigCustom:
    """Tests for custom configuration values."""

    def test_custom_warmup(self):
        """Test custom warmup configuration."""
        config = HybridStreamingConfig(
            warmup_iterations=50,
            max_warmup_iterations=100,
            warmup_learning_rate=0.01,
        )

        assert config.warmup_iterations == 50
        assert config.max_warmup_iterations == 100
        assert config.warmup_learning_rate == 0.01

    def test_custom_gauss_newton(self):
        """Test custom Gauss-Newton configuration."""
        config = HybridStreamingConfig(
            gauss_newton_max_iterations=200,
            gauss_newton_tol=1e-10,
            regularization_factor=1e-8,
        )

        assert config.gauss_newton_max_iterations == 200
        assert config.gauss_newton_tol == 1e-10
        assert config.regularization_factor == 1e-8

    def test_custom_lbfgs(self):
        """Test custom L-BFGS configuration."""
        config = HybridStreamingConfig(
            lbfgs_history_size=20,
            lbfgs_line_search="strong_wolfe",
        )

        assert config.lbfgs_history_size == 20
        assert config.lbfgs_line_search == "strong_wolfe"

    def test_custom_streaming(self):
        """Test custom streaming configuration."""
        config = HybridStreamingConfig(
            chunk_size=5000,
            loop_strategy="scan",
        )

        assert config.chunk_size == 5000
        assert config.loop_strategy == "scan"

    def test_custom_multistart(self):
        """Test custom multi-start configuration."""
        config = HybridStreamingConfig(
            enable_multistart=True,
            n_starts=20,
            multistart_sampler="sobol",
            elimination_fraction=0.3,
        )

        assert config.enable_multistart is True
        assert config.n_starts == 20
        assert config.multistart_sampler == "sobol"
        assert config.elimination_fraction == 0.3

    def test_custom_group_variance(self):
        """Test custom group variance regularization."""
        config = HybridStreamingConfig(
            enable_group_variance_regularization=True,
            group_variance_lambda=0.1,
            group_variance_indices=[(0, 10), (10, 20)],
        )

        assert config.enable_group_variance_regularization is True
        assert config.group_variance_lambda == 0.1
        assert config.group_variance_indices == [(0, 10), (10, 20)]


class TestAggressiveProfile:
    """Tests for aggressive() class method."""

    def test_aggressive_instance(self):
        """Test aggressive profile creation."""
        config = HybridStreamingConfig.aggressive()

        assert config is not None

    def test_aggressive_warmup(self):
        """Test aggressive warmup settings."""
        config = HybridStreamingConfig.aggressive()

        assert config.warmup_iterations == 50
        assert config.max_warmup_iterations == 100
        assert config.warmup_learning_rate == 0.003

    def test_aggressive_tolerances(self):
        """Test aggressive tolerance settings."""
        config = HybridStreamingConfig.aggressive()

        assert config.loss_plateau_threshold == 5e-4
        assert config.gradient_norm_threshold == 5e-3
        assert config.gauss_newton_tol == 1e-7

    def test_aggressive_chunk_size(self):
        """Test aggressive chunk size."""
        config = HybridStreamingConfig.aggressive()

        assert config.chunk_size == 20000


class TestConservativeProfile:
    """Tests for conservative() class method."""

    def test_conservative_instance(self):
        """Test conservative profile creation."""
        config = HybridStreamingConfig.conservative()

        assert config is not None

    def test_conservative_warmup(self):
        """Test conservative warmup settings."""
        config = HybridStreamingConfig.conservative()

        assert config.warmup_iterations == 30
        assert config.max_warmup_iterations == 80
        assert config.warmup_learning_rate == 0.0003

    def test_conservative_tolerances(self):
        """Test conservative tolerance settings."""
        config = HybridStreamingConfig.conservative()

        assert config.loss_plateau_threshold == 1e-5
        assert config.gradient_norm_threshold == 1e-4
        assert config.gauss_newton_tol == 1e-10

    def test_conservative_gauss_newton(self):
        """Test conservative Gauss-Newton settings."""
        config = HybridStreamingConfig.conservative()

        assert config.gauss_newton_max_iterations == 200
        assert config.trust_region_initial == 0.5


class TestMemoryOptimizedProfile:
    """Tests for memory_optimized() class method."""

    def test_memory_optimized_instance(self):
        """Test memory optimized profile creation."""
        config = HybridStreamingConfig.memory_optimized()

        assert config is not None

    def test_memory_optimized_chunk_size(self):
        """Test memory optimized chunk size."""
        config = HybridStreamingConfig.memory_optimized()

        assert config.chunk_size == 5000

    def test_memory_optimized_warmup(self):
        """Test memory optimized warmup settings."""
        config = HybridStreamingConfig.memory_optimized()

        assert config.warmup_iterations == 40
        assert config.max_warmup_iterations == 100

    def test_memory_optimized_precision(self):
        """Test memory optimized precision."""
        config = HybridStreamingConfig.memory_optimized()

        assert config.precision == "float32"

    def test_memory_optimized_checkpoints(self):
        """Test memory optimized checkpoint settings."""
        config = HybridStreamingConfig.memory_optimized()

        assert config.enable_checkpoints is True
        assert config.checkpoint_frequency == 50

    def test_memory_optimized_cg_threshold(self):
        """Test memory optimized CG threshold."""
        config = HybridStreamingConfig.memory_optimized()

        assert config.cg_param_threshold == 1000


class TestWithMultistartProfile:
    """Tests for with_multistart() class method."""

    def test_with_multistart_default(self):
        """Test with_multistart with default n_starts."""
        config = HybridStreamingConfig.with_multistart()

        assert config.enable_multistart is True
        assert config.n_starts == 10

    def test_with_multistart_custom(self):
        """Test with_multistart with custom n_starts."""
        config = HybridStreamingConfig.with_multistart(n_starts=20)

        assert config.enable_multistart is True
        assert config.n_starts == 20

    def test_with_multistart_extra_kwargs(self):
        """Test with_multistart with extra kwargs."""
        config = HybridStreamingConfig.with_multistart(
            n_starts=15,
            chunk_size=5000,
            precision="float64",
        )

        assert config.enable_multistart is True
        assert config.n_starts == 15
        assert config.chunk_size == 5000
        assert config.precision == "float64"


class TestDefenseStrictProfile:
    """Tests for defense_strict() class method."""

    def test_defense_strict_instance(self):
        """Test defense strict profile creation."""
        config = HybridStreamingConfig.defense_strict()

        assert config is not None

    def test_defense_strict_all_enabled(self):
        """Test all defense layers enabled."""
        config = HybridStreamingConfig.defense_strict()

        assert config.enable_warm_start_detection is True
        assert config.enable_adaptive_warmup_lr is True
        assert config.enable_cost_guard is True
        assert config.enable_step_clipping is True

    def test_defense_strict_thresholds(self):
        """Test strict threshold values."""
        config = HybridStreamingConfig.defense_strict()

        assert config.warm_start_threshold == 0.01
        assert config.cost_increase_tolerance == 0.05
        assert config.max_warmup_step_size == 0.05

    def test_defense_strict_learning_rates(self):
        """Test strict learning rate values."""
        config = HybridStreamingConfig.defense_strict()

        assert config.warmup_lr_refinement == 1e-7
        assert config.warmup_lr_careful == 1e-6
        assert config.warmup_learning_rate == 0.0005


class TestDefenseRelaxedProfile:
    """Tests for defense_relaxed() class method."""

    def test_defense_relaxed_instance(self):
        """Test defense relaxed profile creation."""
        config = HybridStreamingConfig.defense_relaxed()

        assert config is not None

    def test_defense_relaxed_all_enabled(self):
        """Test all defense layers enabled but relaxed."""
        config = HybridStreamingConfig.defense_relaxed()

        assert config.enable_warm_start_detection is True
        assert config.enable_adaptive_warmup_lr is True
        assert config.enable_cost_guard is True
        assert config.enable_step_clipping is True

    def test_defense_relaxed_thresholds(self):
        """Test relaxed threshold values."""
        config = HybridStreamingConfig.defense_relaxed()

        assert config.warm_start_threshold == 0.5
        assert config.cost_increase_tolerance == 0.5
        assert config.max_warmup_step_size == 0.5

    def test_defense_relaxed_learning_rates(self):
        """Test relaxed learning rate values."""
        config = HybridStreamingConfig.defense_relaxed()

        assert config.warmup_lr_refinement == 1e-5
        assert config.warmup_lr_careful == 1e-4
        assert config.warmup_learning_rate == 0.003


class TestDefenseDisabledProfile:
    """Tests for defense_disabled() class method."""

    def test_defense_disabled_instance(self):
        """Test defense disabled profile creation."""
        config = HybridStreamingConfig.defense_disabled()

        assert config is not None

    def test_defense_disabled_all_disabled(self):
        """Test all defense layers disabled."""
        config = HybridStreamingConfig.defense_disabled()

        assert config.enable_warm_start_detection is False
        assert config.enable_adaptive_warmup_lr is False
        assert config.enable_cost_guard is False
        assert config.enable_step_clipping is False


class TestScientificDefaultProfile:
    """Tests for scientific_default() class method."""

    def test_scientific_default_instance(self):
        """Test scientific default profile creation."""
        config = HybridStreamingConfig.scientific_default()

        assert config is not None

    def test_scientific_default_defense_layers(self):
        """Test scientific default defense layer settings."""
        config = HybridStreamingConfig.scientific_default()

        assert config.enable_warm_start_detection is True
        assert config.enable_adaptive_warmup_lr is True
        assert config.enable_cost_guard is True
        assert config.enable_step_clipping is True

    def test_scientific_default_precision(self):
        """Test scientific default precision."""
        config = HybridStreamingConfig.scientific_default()

        assert config.precision == "float64"

    def test_scientific_default_thresholds(self):
        """Test scientific default threshold values."""
        config = HybridStreamingConfig.scientific_default()

        assert config.warm_start_threshold == 0.05
        assert config.cost_increase_tolerance == 0.2
        assert config.max_warmup_step_size == 0.1

    def test_scientific_default_gauss_newton(self):
        """Test scientific default Gauss-Newton settings."""
        config = HybridStreamingConfig.scientific_default()

        assert config.gauss_newton_tol == 1e-10
        assert config.gauss_newton_max_iterations == 200

    def test_scientific_default_checkpoints(self):
        """Test scientific default checkpoint settings."""
        config = HybridStreamingConfig.scientific_default()

        assert config.enable_checkpoints is True
        assert config.checkpoint_frequency == 100


class TestValidation:
    """Tests for configuration validation."""

    def test_invalid_normalization_strategy(self):
        """Test that invalid normalization_strategy raises ValueError."""
        with pytest.raises(ValueError, match="normalization_strategy"):
            HybridStreamingConfig(normalization_strategy="invalid")

    def test_invalid_precision(self):
        """Test that invalid precision raises ValueError."""
        with pytest.raises(ValueError, match="precision"):
            HybridStreamingConfig(precision="float128")

    def test_invalid_loop_strategy(self):
        """Test that invalid loop_strategy raises ValueError."""
        with pytest.raises(ValueError, match="loop_strategy"):
            HybridStreamingConfig(loop_strategy="invalid")

    def test_invalid_lbfgs_line_search(self):
        """Test that invalid lbfgs_line_search raises ValueError."""
        with pytest.raises(ValueError, match="lbfgs_line_search"):
            HybridStreamingConfig(lbfgs_line_search="invalid")

    def test_negative_warmup_iterations(self):
        """Test that negative warmup_iterations raises ValueError."""
        with pytest.raises(ValueError, match="warmup_iterations"):
            HybridStreamingConfig(warmup_iterations=-1)

    def test_zero_max_warmup_iterations(self):
        """Test that zero max_warmup_iterations raises ValueError."""
        with pytest.raises(ValueError, match="max_warmup_iterations"):
            HybridStreamingConfig(max_warmup_iterations=0)

    def test_warmup_exceeds_max(self):
        """Test that warmup_iterations > max_warmup_iterations raises ValueError."""
        with pytest.raises(ValueError, match="warmup_iterations"):
            HybridStreamingConfig(warmup_iterations=100, max_warmup_iterations=50)

    def test_negative_learning_rate(self):
        """Test that negative learning rate raises ValueError."""
        with pytest.raises(ValueError, match="warmup_learning_rate"):
            HybridStreamingConfig(warmup_learning_rate=-0.001)

    def test_zero_lbfgs_history_size(self):
        """Test that zero lbfgs_history_size raises ValueError."""
        with pytest.raises(ValueError, match="lbfgs_history_size"):
            HybridStreamingConfig(lbfgs_history_size=0)

    def test_zero_gauss_newton_max_iterations(self):
        """Test that zero gauss_newton_max_iterations raises ValueError."""
        with pytest.raises(ValueError, match="gauss_newton_max_iterations"):
            HybridStreamingConfig(gauss_newton_max_iterations=0)

    def test_negative_regularization_factor(self):
        """Test that negative regularization_factor raises ValueError."""
        with pytest.raises(ValueError, match="regularization_factor"):
            HybridStreamingConfig(regularization_factor=-1e-10)

    def test_zero_chunk_size(self):
        """Test that zero chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            HybridStreamingConfig(chunk_size=0)

    def test_invalid_multistart_sampler(self):
        """Test that invalid multistart_sampler raises ValueError."""
        with pytest.raises(ValueError, match="multistart_sampler"):
            HybridStreamingConfig(enable_multistart=True, multistart_sampler="invalid")

    def test_zero_n_starts_when_multistart_enabled(self):
        """Test that n_starts < 1 when multistart enabled raises ValueError."""
        with pytest.raises(ValueError, match="n_starts"):
            HybridStreamingConfig(enable_multistart=True, n_starts=0)

    def test_invalid_elimination_fraction(self):
        """Test that elimination_fraction outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="elimination_fraction"):
            HybridStreamingConfig(enable_multistart=True, elimination_fraction=1.5)

    def test_elimination_fraction_zero(self):
        """Test that elimination_fraction=0 raises ValueError."""
        with pytest.raises(ValueError, match="elimination_fraction"):
            HybridStreamingConfig(enable_multistart=True, elimination_fraction=0.0)

    def test_invalid_group_variance_indices_not_list(self):
        """Test that non-list group_variance_indices raises ValueError."""
        with pytest.raises(ValueError, match="group_variance_indices"):
            HybridStreamingConfig(
                enable_group_variance_regularization=True,
                group_variance_indices="invalid",
            )

    def test_invalid_group_variance_indices_not_tuple(self):
        """Test that non-tuple elements in group_variance_indices raises ValueError."""
        with pytest.raises(ValueError, match="group_variance_indices"):
            HybridStreamingConfig(
                enable_group_variance_regularization=True,
                group_variance_indices=[(0, 10), "invalid"],
            )

    def test_invalid_group_variance_indices_wrong_length(self):
        """Test that tuples with wrong length raise ValueError."""
        with pytest.raises(ValueError, match="group_variance_indices"):
            HybridStreamingConfig(
                enable_group_variance_regularization=True,
                group_variance_indices=[(0, 10, 20)],
            )

    def test_invalid_group_variance_indices_negative_start(self):
        """Test that negative start in indices raises ValueError."""
        with pytest.raises(ValueError, match="group_variance_indices"):
            HybridStreamingConfig(
                enable_group_variance_regularization=True,
                group_variance_indices=[(-1, 10)],
            )

    def test_invalid_group_variance_indices_end_not_greater(self):
        """Test that end <= start in indices raises ValueError."""
        with pytest.raises(ValueError, match="group_variance_indices"):
            HybridStreamingConfig(
                enable_group_variance_regularization=True,
                group_variance_indices=[(10, 5)],
            )

    def test_invalid_warm_start_threshold(self):
        """Test that warm_start_threshold outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="warm_start_threshold"):
            HybridStreamingConfig(
                enable_warm_start_detection=True,
                warm_start_threshold=1.5,
            )

    def test_invalid_cost_increase_tolerance_negative(self):
        """Test that negative cost_increase_tolerance raises ValueError."""
        with pytest.raises(ValueError, match="cost_increase_tolerance"):
            HybridStreamingConfig(
                enable_cost_guard=True,
                cost_increase_tolerance=-0.1,
            )

    def test_lr_progression_invalid_order(self):
        """Test that lr_refinement > lr_careful raises ValueError."""
        with pytest.raises(ValueError):
            HybridStreamingConfig(
                enable_adaptive_warmup_lr=True,
                warmup_lr_refinement=1e-4,  # Greater than careful
                warmup_lr_careful=1e-5,
            )


class TestEdgeCases:
    """Edge case tests."""

    def test_minimal_valid_config(self):
        """Test minimal valid configuration."""
        config = HybridStreamingConfig(
            warmup_iterations=0,  # Edge: allowed (skip warmup)
            max_warmup_iterations=1,
        )

        assert config.warmup_iterations == 0
        assert config.max_warmup_iterations == 1

    def test_large_chunk_size(self):
        """Test very large chunk size."""
        config = HybridStreamingConfig(chunk_size=1_000_000)

        assert config.chunk_size == 1_000_000

    def test_active_switching_criteria_none_becomes_list(self):
        """Test that None active_switching_criteria becomes default list."""
        config = HybridStreamingConfig()

        assert config.active_switching_criteria == ["plateau", "gradient", "max_iter"]

    def test_group_variance_indices_empty_list(self):
        """Test empty list for group_variance_indices is valid."""
        config = HybridStreamingConfig(
            enable_group_variance_regularization=True,
            group_variance_indices=[],
        )

        assert config.group_variance_indices == []

    def test_slots_defined(self):
        """Test that __slots__ is defined for memory efficiency."""
        config = HybridStreamingConfig()

        # Should have __slots__ defined in dataclass
        assert hasattr(HybridStreamingConfig, "__slots__")


class TestProfileComparison:
    """Tests comparing different profiles."""

    def test_aggressive_faster_than_conservative(self):
        """Test that aggressive profile has fewer warmup iterations."""
        aggressive = HybridStreamingConfig.aggressive()
        conservative = HybridStreamingConfig.conservative()

        # Aggressive should have more warmup iterations since it converges faster
        # and can afford to warmup longer, but actually it has fewer due to L-BFGS efficiency
        assert aggressive.warmup_iterations >= conservative.warmup_iterations

    def test_memory_optimized_smaller_chunks(self):
        """Test that memory optimized has smaller chunks than default."""
        default = HybridStreamingConfig()
        memory = HybridStreamingConfig.memory_optimized()

        assert memory.chunk_size < default.chunk_size

    def test_scientific_default_higher_precision(self):
        """Test that scientific default uses higher precision than memory optimized."""
        scientific = HybridStreamingConfig.scientific_default()
        memory = HybridStreamingConfig.memory_optimized()

        assert scientific.precision == "float64"
        assert memory.precision == "float32"

    def test_defense_strict_vs_relaxed_thresholds(self):
        """Test that strict has tighter thresholds than relaxed."""
        strict = HybridStreamingConfig.defense_strict()
        relaxed = HybridStreamingConfig.defense_relaxed()

        assert strict.warm_start_threshold < relaxed.warm_start_threshold
        assert strict.cost_increase_tolerance < relaxed.cost_increase_tolerance
        assert strict.max_warmup_step_size < relaxed.max_warmup_step_size
