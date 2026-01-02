"""Tests for nlsq.streaming.validators module.

Characterization tests for configuration validation functions used by
HybridStreamingConfig and other streaming components.

Coverage targets:
- ConfigValidationError: Custom exception class
- Primitive validators: validate_positive, validate_non_negative, validate_range, etc.
- Enum validators: validate_normalization_strategy, validate_precision, etc.
- Composite validators: validate_warmup_config, validate_lbfgs_config, etc.
"""

import pytest

from nlsq.streaming.validators import (
    ConfigValidationError,
    validate_cg_config,
    validate_defense_layer_config,
    validate_enum_value,
    validate_gauss_newton_config,
    validate_gradient_clip,
    validate_group_variance_config,
    validate_lbfgs_config,
    validate_lbfgs_line_search,
    validate_less_than_or_equal,
    validate_loop_strategy,
    validate_lr_schedule_config,
    validate_multistart_config,
    validate_multistart_sampler,
    validate_non_negative,
    validate_normalization_strategy,
    validate_positive,
    validate_precision,
    validate_range,
    validate_streaming_config,
    validate_warmup_config,
)


class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_is_value_error_subclass(self):
        """Test that ConfigValidationError is a ValueError subclass."""
        assert issubclass(ConfigValidationError, ValueError)

    def test_can_be_raised(self):
        """Test that ConfigValidationError can be raised and caught."""
        with pytest.raises(ConfigValidationError):
            raise ConfigValidationError("test message")

    def test_message_preserved(self):
        """Test that error message is preserved."""
        with pytest.raises(ConfigValidationError, match="custom error message"):
            raise ConfigValidationError("custom error message")


class TestValidateEnumValue:
    """Tests for validate_enum_value function."""

    def test_valid_value(self):
        """Test with valid enum value."""
        # Should not raise
        validate_enum_value("a", ("a", "b", "c"), "test_param")

    def test_invalid_value(self):
        """Test with invalid enum value."""
        with pytest.raises(ConfigValidationError, match="test_param"):
            validate_enum_value("d", ("a", "b", "c"), "test_param")

    def test_error_includes_valid_options(self):
        """Test that error message includes valid options."""
        with pytest.raises(ConfigValidationError, match=r"a.*b.*c"):
            validate_enum_value("invalid", ("a", "b", "c"), "param")

    def test_case_sensitive(self):
        """Test that validation is case-sensitive."""
        with pytest.raises(ConfigValidationError):
            validate_enum_value("A", ("a", "b", "c"), "param")


class TestValidatePositive:
    """Tests for validate_positive function."""

    def test_positive_value(self):
        """Test with positive value."""
        validate_positive(1, "param")
        validate_positive(0.001, "param")
        validate_positive(1e-10, "param")

    def test_zero_value(self):
        """Test with zero value."""
        with pytest.raises(ConfigValidationError, match="param"):
            validate_positive(0, "param")

    def test_negative_value(self):
        """Test with negative value."""
        with pytest.raises(ConfigValidationError, match="param"):
            validate_positive(-1, "param")

    def test_error_message_includes_param_name(self):
        """Test that error message includes parameter name."""
        with pytest.raises(ConfigValidationError, match="my_parameter"):
            validate_positive(0, "my_parameter")


class TestValidateNonNegative:
    """Tests for validate_non_negative function."""

    def test_positive_value(self):
        """Test with positive value."""
        validate_non_negative(1, "param")

    def test_zero_value(self):
        """Test with zero value."""
        validate_non_negative(0, "param")

    def test_negative_value(self):
        """Test with negative value."""
        with pytest.raises(ConfigValidationError, match="param"):
            validate_non_negative(-1, "param")

    def test_small_positive_value(self):
        """Test with very small positive value."""
        validate_non_negative(1e-100, "param")


class TestValidateRange:
    """Tests for validate_range function."""

    def test_value_in_range_inclusive(self):
        """Test value within inclusive range."""
        validate_range(0.5, 0, 1, "param")
        validate_range(0, 0, 1, "param")  # At lower bound
        validate_range(1, 0, 1, "param")  # At upper bound

    def test_value_below_range(self):
        """Test value below range."""
        with pytest.raises(ConfigValidationError, match="param"):
            validate_range(-0.1, 0, 1, "param")

    def test_value_above_range(self):
        """Test value above range."""
        with pytest.raises(ConfigValidationError, match="param"):
            validate_range(1.1, 0, 1, "param")

    def test_exclusive_min(self):
        """Test exclusive minimum."""
        with pytest.raises(ConfigValidationError):
            validate_range(0, 0, 1, "param", inclusive_min=False)

        # Just above should work
        validate_range(0.001, 0, 1, "param", inclusive_min=False)

    def test_exclusive_max(self):
        """Test exclusive maximum."""
        with pytest.raises(ConfigValidationError):
            validate_range(1, 0, 1, "param", inclusive_max=False)

        # Just below should work
        validate_range(0.999, 0, 1, "param", inclusive_max=False)

    def test_exclusive_both(self):
        """Test exclusive on both ends."""
        with pytest.raises(ConfigValidationError):
            validate_range(0, 0, 1, "param", inclusive_min=False, inclusive_max=False)

        with pytest.raises(ConfigValidationError):
            validate_range(1, 0, 1, "param", inclusive_min=False, inclusive_max=False)

        validate_range(0.5, 0, 1, "param", inclusive_min=False, inclusive_max=False)


class TestValidateLessThanOrEqual:
    """Tests for validate_less_than_or_equal function."""

    def test_less_than(self):
        """Test value1 < value2."""
        validate_less_than_or_equal(1, 2, "a", "b")

    def test_equal(self):
        """Test value1 == value2."""
        validate_less_than_or_equal(2, 2, "a", "b")

    def test_greater_than(self):
        """Test value1 > value2 raises error."""
        with pytest.raises(ConfigValidationError, match=r"a.*b"):
            validate_less_than_or_equal(3, 2, "a", "b")

    def test_error_includes_values(self):
        """Test error message includes actual values."""
        with pytest.raises(ConfigValidationError, match=r"10.*5"):
            validate_less_than_or_equal(10, 5, "high", "low")


class TestValidateNormalizationStrategy:
    """Tests for validate_normalization_strategy function."""

    def test_valid_strategies(self):
        """Test all valid normalization strategies."""
        for strategy in ("auto", "bounds", "p0", "none"):
            validate_normalization_strategy(strategy)

    def test_invalid_strategy(self):
        """Test invalid strategy."""
        with pytest.raises(ConfigValidationError, match="normalization_strategy"):
            validate_normalization_strategy("invalid")


class TestValidatePrecision:
    """Tests for validate_precision function."""

    def test_valid_precisions(self):
        """Test all valid precision values."""
        for precision in ("float32", "float64", "auto"):
            validate_precision(precision)

    def test_invalid_precision(self):
        """Test invalid precision."""
        with pytest.raises(ConfigValidationError, match="precision"):
            validate_precision("float16")

        with pytest.raises(ConfigValidationError, match="precision"):
            validate_precision("double")


class TestValidateLoopStrategy:
    """Tests for validate_loop_strategy function."""

    def test_valid_strategies(self):
        """Test all valid loop strategies."""
        for strategy in ("auto", "scan", "loop"):
            validate_loop_strategy(strategy)

    def test_invalid_strategy(self):
        """Test invalid strategy."""
        with pytest.raises(ConfigValidationError, match="loop_strategy"):
            validate_loop_strategy("parallel")


class TestValidateLbfgsLineSearch:
    """Tests for validate_lbfgs_line_search function."""

    def test_valid_line_searches(self):
        """Test all valid line search methods."""
        for method in ("wolfe", "strong_wolfe", "backtracking"):
            validate_lbfgs_line_search(method)

    def test_invalid_line_search(self):
        """Test invalid line search."""
        with pytest.raises(ConfigValidationError, match="lbfgs_line_search"):
            validate_lbfgs_line_search("golden_section")


class TestValidateMultistartSampler:
    """Tests for validate_multistart_sampler function."""

    def test_valid_samplers(self):
        """Test all valid sampler methods."""
        for sampler in ("lhs", "sobol", "halton"):
            validate_multistart_sampler(sampler)

    def test_invalid_sampler(self):
        """Test invalid sampler."""
        with pytest.raises(ConfigValidationError, match="multistart_sampler"):
            validate_multistart_sampler("random")


class TestValidateWarmupConfig:
    """Tests for validate_warmup_config function."""

    def test_valid_config(self):
        """Test with valid warmup config."""
        validate_warmup_config(
            warmup_iterations=200,
            max_warmup_iterations=500,
            warmup_learning_rate=0.001,
            loss_plateau_threshold=1e-4,
            gradient_norm_threshold=1e-3,
        )

    def test_zero_warmup_iterations(self):
        """Test zero warmup_iterations (allowed)."""
        validate_warmup_config(
            warmup_iterations=0,  # Allowed
            max_warmup_iterations=500,
            warmup_learning_rate=0.001,
            loss_plateau_threshold=1e-4,
            gradient_norm_threshold=1e-3,
        )

    def test_negative_warmup_iterations(self):
        """Test negative warmup_iterations."""
        with pytest.raises(ConfigValidationError, match="warmup_iterations"):
            validate_warmup_config(
                warmup_iterations=-1,
                max_warmup_iterations=500,
                warmup_learning_rate=0.001,
                loss_plateau_threshold=1e-4,
                gradient_norm_threshold=1e-3,
            )

    def test_zero_max_warmup_iterations(self):
        """Test zero max_warmup_iterations."""
        with pytest.raises(ConfigValidationError, match="max_warmup_iterations"):
            validate_warmup_config(
                warmup_iterations=0,
                max_warmup_iterations=0,
                warmup_learning_rate=0.001,
                loss_plateau_threshold=1e-4,
                gradient_norm_threshold=1e-3,
            )

    def test_warmup_exceeds_max(self):
        """Test warmup_iterations > max_warmup_iterations."""
        with pytest.raises(ConfigValidationError, match="warmup_iterations"):
            validate_warmup_config(
                warmup_iterations=600,
                max_warmup_iterations=500,
                warmup_learning_rate=0.001,
                loss_plateau_threshold=1e-4,
                gradient_norm_threshold=1e-3,
            )

    def test_zero_learning_rate(self):
        """Test zero learning rate."""
        with pytest.raises(ConfigValidationError, match="warmup_learning_rate"):
            validate_warmup_config(
                warmup_iterations=200,
                max_warmup_iterations=500,
                warmup_learning_rate=0,
                loss_plateau_threshold=1e-4,
                gradient_norm_threshold=1e-3,
            )


class TestValidateLbfgsConfig:
    """Tests for validate_lbfgs_config function."""

    def test_valid_config(self):
        """Test with valid L-BFGS config."""
        validate_lbfgs_config(
            history_size=10,
            initial_step_size=0.1,
            line_search="wolfe",
            exploration_step_size=0.1,
            refinement_step_size=1.0,
        )

    def test_zero_history_size(self):
        """Test zero history size."""
        with pytest.raises(ConfigValidationError, match="lbfgs_history_size"):
            validate_lbfgs_config(
                history_size=0,
                initial_step_size=0.1,
                line_search="wolfe",
                exploration_step_size=0.1,
                refinement_step_size=1.0,
            )

    def test_invalid_line_search(self):
        """Test invalid line search."""
        with pytest.raises(ConfigValidationError, match="lbfgs_line_search"):
            validate_lbfgs_config(
                history_size=10,
                initial_step_size=0.1,
                line_search="invalid",
                exploration_step_size=0.1,
                refinement_step_size=1.0,
            )

    def test_zero_step_size(self):
        """Test zero step sizes."""
        with pytest.raises(ConfigValidationError, match="lbfgs_initial_step_size"):
            validate_lbfgs_config(
                history_size=10,
                initial_step_size=0,
                line_search="wolfe",
                exploration_step_size=0.1,
                refinement_step_size=1.0,
            )


class TestValidateGaussNewtonConfig:
    """Tests for validate_gauss_newton_config function."""

    def test_valid_config(self):
        """Test with valid Gauss-Newton config."""
        validate_gauss_newton_config(
            max_iterations=100,
            tol=1e-8,
            trust_region_initial=1.0,
            regularization_factor=1e-10,
        )

    def test_zero_max_iterations(self):
        """Test zero max iterations."""
        with pytest.raises(ConfigValidationError, match="gauss_newton_max_iterations"):
            validate_gauss_newton_config(
                max_iterations=0,
                tol=1e-8,
                trust_region_initial=1.0,
                regularization_factor=1e-10,
            )

    def test_zero_tol(self):
        """Test zero tolerance."""
        with pytest.raises(ConfigValidationError, match="gauss_newton_tol"):
            validate_gauss_newton_config(
                max_iterations=100,
                tol=0,
                trust_region_initial=1.0,
                regularization_factor=1e-10,
            )

    def test_negative_regularization(self):
        """Test negative regularization factor."""
        with pytest.raises(ConfigValidationError, match="regularization_factor"):
            validate_gauss_newton_config(
                max_iterations=100,
                tol=1e-8,
                trust_region_initial=1.0,
                regularization_factor=-1e-10,
            )

    def test_zero_regularization_allowed(self):
        """Test zero regularization is allowed."""
        validate_gauss_newton_config(
            max_iterations=100,
            tol=1e-8,
            trust_region_initial=1.0,
            regularization_factor=0,  # Should be allowed
        )


class TestValidateCgConfig:
    """Tests for validate_cg_config function."""

    def test_valid_config(self):
        """Test with valid CG config."""
        validate_cg_config(
            max_iterations=100,
            relative_tolerance=1e-4,
            absolute_tolerance=1e-10,
            param_threshold=2000,
        )

    def test_zero_max_iterations(self):
        """Test zero max iterations."""
        with pytest.raises(ConfigValidationError, match="cg_max_iterations"):
            validate_cg_config(
                max_iterations=0,
                relative_tolerance=1e-4,
                absolute_tolerance=1e-10,
                param_threshold=2000,
            )

    def test_zero_relative_tolerance(self):
        """Test zero relative tolerance."""
        with pytest.raises(ConfigValidationError, match="cg_relative_tolerance"):
            validate_cg_config(
                max_iterations=100,
                relative_tolerance=0,
                absolute_tolerance=1e-10,
                param_threshold=2000,
            )

    def test_zero_param_threshold(self):
        """Test zero param threshold."""
        with pytest.raises(ConfigValidationError, match="cg_param_threshold"):
            validate_cg_config(
                max_iterations=100,
                relative_tolerance=1e-4,
                absolute_tolerance=1e-10,
                param_threshold=0,
            )


class TestValidateGroupVarianceConfig:
    """Tests for validate_group_variance_config function."""

    def test_disabled(self):
        """Test when disabled (no validation)."""
        validate_group_variance_config(
            enabled=False,
            lambda_val=-1,  # Invalid but ignored when disabled
            indices="invalid",  # Invalid but ignored
        )

    def test_valid_config(self):
        """Test valid enabled config."""
        validate_group_variance_config(
            enabled=True,
            lambda_val=0.01,
            indices=[(0, 10), (10, 20)],
        )

    def test_enabled_none_indices(self):
        """Test enabled with None indices (valid)."""
        validate_group_variance_config(
            enabled=True,
            lambda_val=0.01,
            indices=None,
        )

    def test_zero_lambda(self):
        """Test zero lambda when enabled."""
        with pytest.raises(ConfigValidationError, match="group_variance_lambda"):
            validate_group_variance_config(
                enabled=True,
                lambda_val=0,
                indices=[(0, 10)],
            )

    def test_not_list(self):
        """Test indices not a list."""
        with pytest.raises(ConfigValidationError, match="group_variance_indices"):
            validate_group_variance_config(
                enabled=True,
                lambda_val=0.01,
                indices="invalid",
            )

    def test_not_tuple(self):
        """Test element not a tuple."""
        with pytest.raises(ConfigValidationError, match="group_variance_indices"):
            validate_group_variance_config(
                enabled=True,
                lambda_val=0.01,
                indices=["invalid"],
            )

    def test_wrong_length(self):
        """Test tuple with wrong length."""
        with pytest.raises(ConfigValidationError, match="group_variance_indices"):
            validate_group_variance_config(
                enabled=True,
                lambda_val=0.01,
                indices=[(0, 10, 20)],
            )

    def test_non_integer_values(self):
        """Test non-integer values in tuple."""
        with pytest.raises(ConfigValidationError, match="group_variance_indices"):
            validate_group_variance_config(
                enabled=True,
                lambda_val=0.01,
                indices=[(0.5, 10)],
            )

    def test_negative_start(self):
        """Test negative start index."""
        with pytest.raises(ConfigValidationError, match="group_variance_indices"):
            validate_group_variance_config(
                enabled=True,
                lambda_val=0.01,
                indices=[(-1, 10)],
            )

    def test_end_not_greater_than_start(self):
        """Test end <= start."""
        with pytest.raises(ConfigValidationError, match="group_variance_indices"):
            validate_group_variance_config(
                enabled=True,
                lambda_val=0.01,
                indices=[(10, 5)],
            )

    def test_end_equals_start(self):
        """Test end == start."""
        with pytest.raises(ConfigValidationError, match="group_variance_indices"):
            validate_group_variance_config(
                enabled=True,
                lambda_val=0.01,
                indices=[(5, 5)],
            )


class TestValidateStreamingConfig:
    """Tests for validate_streaming_config function."""

    def test_valid_config(self):
        """Test valid streaming config."""
        validate_streaming_config(
            chunk_size=10000,
            checkpoint_frequency=100,
            callback_frequency=10,
        )

    def test_zero_chunk_size(self):
        """Test zero chunk size."""
        with pytest.raises(ConfigValidationError, match="chunk_size"):
            validate_streaming_config(
                chunk_size=0,
                checkpoint_frequency=100,
                callback_frequency=10,
            )

    def test_zero_checkpoint_frequency(self):
        """Test zero checkpoint frequency."""
        with pytest.raises(ConfigValidationError, match="checkpoint_frequency"):
            validate_streaming_config(
                chunk_size=10000,
                checkpoint_frequency=0,
                callback_frequency=10,
            )

    def test_zero_callback_frequency(self):
        """Test zero callback frequency."""
        with pytest.raises(ConfigValidationError, match="callback_frequency"):
            validate_streaming_config(
                chunk_size=10000,
                checkpoint_frequency=100,
                callback_frequency=0,
            )


class TestValidateLrScheduleConfig:
    """Tests for validate_lr_schedule_config function."""

    def test_disabled(self):
        """Test when disabled (no validation)."""
        validate_lr_schedule_config(
            enabled=False,
            warmup_steps=-1,  # Invalid but ignored
            decay_steps=0,  # Invalid but ignored
            end_value=-1,  # Invalid but ignored
        )

    def test_valid_config(self):
        """Test valid enabled config."""
        validate_lr_schedule_config(
            enabled=True,
            warmup_steps=50,
            decay_steps=450,
            end_value=0.0001,
        )

    def test_zero_warmup_steps_allowed(self):
        """Test zero warmup steps is allowed."""
        validate_lr_schedule_config(
            enabled=True,
            warmup_steps=0,
            decay_steps=450,
            end_value=0.0001,
        )

    def test_negative_warmup_steps(self):
        """Test negative warmup steps."""
        with pytest.raises(ConfigValidationError, match="lr_schedule_warmup_steps"):
            validate_lr_schedule_config(
                enabled=True,
                warmup_steps=-1,
                decay_steps=450,
                end_value=0.0001,
            )

    def test_zero_decay_steps(self):
        """Test zero decay steps."""
        with pytest.raises(ConfigValidationError, match="lr_schedule_decay_steps"):
            validate_lr_schedule_config(
                enabled=True,
                warmup_steps=50,
                decay_steps=0,
                end_value=0.0001,
            )

    def test_zero_end_value(self):
        """Test zero end value."""
        with pytest.raises(ConfigValidationError, match="lr_schedule_end_value"):
            validate_lr_schedule_config(
                enabled=True,
                warmup_steps=50,
                decay_steps=450,
                end_value=0,
            )


class TestValidateGradientClip:
    """Tests for validate_gradient_clip function."""

    def test_none_value(self):
        """Test None value (no clipping)."""
        validate_gradient_clip(None)

    def test_valid_positive(self):
        """Test valid positive clip value."""
        validate_gradient_clip(1.0)
        validate_gradient_clip(0.1)

    def test_zero_value(self):
        """Test zero clip value."""
        with pytest.raises(ConfigValidationError, match="gradient_clip_value"):
            validate_gradient_clip(0)

    def test_negative_value(self):
        """Test negative clip value."""
        with pytest.raises(ConfigValidationError, match="gradient_clip_value"):
            validate_gradient_clip(-1.0)


class TestValidateDefenseLayerConfig:
    """Tests for validate_defense_layer_config function."""

    def test_all_disabled(self):
        """Test with all layers disabled."""
        validate_defense_layer_config(
            enable_warm_start=False,
            warm_start_threshold=2.0,  # Invalid but ignored
            enable_adaptive_lr=False,
            lr_refinement=-1,  # Invalid but ignored
            lr_careful=-1,  # Invalid but ignored
            lr_default=-1,  # Invalid but ignored
            enable_cost_guard=False,
            cost_tolerance=-1,  # Invalid but ignored
            enable_step_clipping=False,
            max_step_size=-1,  # Invalid but ignored
        )

    def test_valid_config(self):
        """Test valid complete config."""
        validate_defense_layer_config(
            enable_warm_start=True,
            warm_start_threshold=0.01,
            enable_adaptive_lr=True,
            lr_refinement=1e-6,
            lr_careful=1e-5,
            lr_default=0.001,
            enable_cost_guard=True,
            cost_tolerance=0.05,
            enable_step_clipping=True,
            max_step_size=0.1,
        )

    def test_warm_start_threshold_out_of_range(self):
        """Test warm start threshold outside (0, 1)."""
        with pytest.raises(ConfigValidationError, match="warm_start_threshold"):
            validate_defense_layer_config(
                enable_warm_start=True,
                warm_start_threshold=1.5,
                enable_adaptive_lr=False,
                lr_refinement=1e-6,
                lr_careful=1e-5,
                lr_default=0.001,
                enable_cost_guard=False,
                cost_tolerance=0.05,
                enable_step_clipping=False,
                max_step_size=0.1,
            )

    def test_lr_progression_invalid(self):
        """Test invalid learning rate progression."""
        # refinement > careful should fail
        with pytest.raises(ConfigValidationError):
            validate_defense_layer_config(
                enable_warm_start=False,
                warm_start_threshold=0.01,
                enable_adaptive_lr=True,
                lr_refinement=1e-4,  # > lr_careful
                lr_careful=1e-5,
                lr_default=0.001,
                enable_cost_guard=False,
                cost_tolerance=0.05,
                enable_step_clipping=False,
                max_step_size=0.1,
            )

    def test_cost_tolerance_negative(self):
        """Test negative cost tolerance."""
        with pytest.raises(ConfigValidationError, match="cost_increase_tolerance"):
            validate_defense_layer_config(
                enable_warm_start=False,
                warm_start_threshold=0.01,
                enable_adaptive_lr=False,
                lr_refinement=1e-6,
                lr_careful=1e-5,
                lr_default=0.001,
                enable_cost_guard=True,
                cost_tolerance=-0.1,
                enable_step_clipping=False,
                max_step_size=0.1,
            )

    def test_max_step_size_zero(self):
        """Test zero max step size."""
        with pytest.raises(ConfigValidationError, match="max_warmup_step_size"):
            validate_defense_layer_config(
                enable_warm_start=False,
                warm_start_threshold=0.01,
                enable_adaptive_lr=False,
                lr_refinement=1e-6,
                lr_careful=1e-5,
                lr_default=0.001,
                enable_cost_guard=False,
                cost_tolerance=0.05,
                enable_step_clipping=True,
                max_step_size=0,
            )


class TestValidateMultistartConfig:
    """Tests for validate_multistart_config function."""

    def test_disabled(self):
        """Test when disabled (no validation)."""
        validate_multistart_config(
            enabled=False,
            n_starts=0,  # Invalid but ignored
            sampler="invalid",  # Invalid but ignored
            elimination_fraction=2.0,  # Invalid but ignored
            elimination_rounds=-1,  # Invalid but ignored
            batches_per_round=0,  # Invalid but ignored
            scale_factor=-1,  # Invalid but ignored
        )

    def test_valid_config(self):
        """Test valid enabled config."""
        validate_multistart_config(
            enabled=True,
            n_starts=10,
            sampler="lhs",
            elimination_fraction=0.5,
            elimination_rounds=3,
            batches_per_round=50,
            scale_factor=1.0,
        )

    def test_zero_n_starts(self):
        """Test zero n_starts."""
        with pytest.raises(ConfigValidationError, match="n_starts"):
            validate_multistart_config(
                enabled=True,
                n_starts=0,
                sampler="lhs",
                elimination_fraction=0.5,
                elimination_rounds=3,
                batches_per_round=50,
                scale_factor=1.0,
            )

    def test_invalid_sampler(self):
        """Test invalid sampler."""
        with pytest.raises(ConfigValidationError, match="multistart_sampler"):
            validate_multistart_config(
                enabled=True,
                n_starts=10,
                sampler="invalid",
                elimination_fraction=0.5,
                elimination_rounds=3,
                batches_per_round=50,
                scale_factor=1.0,
            )

    def test_elimination_fraction_out_of_range(self):
        """Test elimination fraction outside (0, 1)."""
        with pytest.raises(ConfigValidationError, match="elimination_fraction"):
            validate_multistart_config(
                enabled=True,
                n_starts=10,
                sampler="lhs",
                elimination_fraction=1.5,
                elimination_rounds=3,
                batches_per_round=50,
                scale_factor=1.0,
            )

    def test_elimination_fraction_zero(self):
        """Test elimination fraction = 0."""
        with pytest.raises(ConfigValidationError, match="elimination_fraction"):
            validate_multistart_config(
                enabled=True,
                n_starts=10,
                sampler="lhs",
                elimination_fraction=0.0,
                elimination_rounds=3,
                batches_per_round=50,
                scale_factor=1.0,
            )

    def test_elimination_fraction_one(self):
        """Test elimination fraction = 1."""
        with pytest.raises(ConfigValidationError, match="elimination_fraction"):
            validate_multistart_config(
                enabled=True,
                n_starts=10,
                sampler="lhs",
                elimination_fraction=1.0,
                elimination_rounds=3,
                batches_per_round=50,
                scale_factor=1.0,
            )

    def test_negative_elimination_rounds(self):
        """Test negative elimination rounds."""
        with pytest.raises(ConfigValidationError, match="elimination_rounds"):
            validate_multistart_config(
                enabled=True,
                n_starts=10,
                sampler="lhs",
                elimination_fraction=0.5,
                elimination_rounds=-1,
                batches_per_round=50,
                scale_factor=1.0,
            )

    def test_zero_elimination_rounds_allowed(self):
        """Test zero elimination rounds is allowed."""
        validate_multistart_config(
            enabled=True,
            n_starts=10,
            sampler="lhs",
            elimination_fraction=0.5,
            elimination_rounds=0,  # Allowed
            batches_per_round=50,
            scale_factor=1.0,
        )

    def test_zero_batches_per_round(self):
        """Test zero batches per round."""
        with pytest.raises(ConfigValidationError, match="batches_per_round"):
            validate_multistart_config(
                enabled=True,
                n_starts=10,
                sampler="lhs",
                elimination_fraction=0.5,
                elimination_rounds=3,
                batches_per_round=0,
                scale_factor=1.0,
            )

    def test_zero_scale_factor(self):
        """Test zero scale factor."""
        with pytest.raises(ConfigValidationError, match="scale_factor"):
            validate_multistart_config(
                enabled=True,
                n_starts=10,
                sampler="lhs",
                elimination_fraction=0.5,
                elimination_rounds=3,
                batches_per_round=50,
                scale_factor=0,
            )
