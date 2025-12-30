"""Validators for streaming configuration parameters.

This module provides validation functions for HybridStreamingConfig parameters,
extracted from the monolithic __post_init__ method to reduce complexity and
improve testability.
"""

from typing import Any


class ConfigValidationError(ValueError):
    """Exception raised when configuration validation fails."""

    pass


def validate_enum_value(
    value: Any,
    valid_options: tuple[str, ...],
    param_name: str,
) -> None:
    """Validate that value is one of the valid options.

    Parameters
    ----------
    value : Any
        The value to validate.
    valid_options : tuple[str, ...]
        Tuple of valid option strings.
    param_name : str
        Name of the parameter for error messages.

    Raises
    ------
    ConfigValidationError
        If value is not in valid_options.
    """
    if value not in valid_options:
        raise ConfigValidationError(
            f"{param_name} must be one of: {valid_options}, got: {value}"
        )


def validate_positive(value: float | int, param_name: str) -> None:
    """Validate that value is strictly positive.

    Parameters
    ----------
    value : float or int
        The value to validate.
    param_name : str
        Name of the parameter for error messages.

    Raises
    ------
    ConfigValidationError
        If value is not positive.
    """
    if value <= 0:
        raise ConfigValidationError(f"{param_name} must be positive")


def validate_non_negative(value: float | int, param_name: str) -> None:
    """Validate that value is non-negative.

    Parameters
    ----------
    value : float or int
        The value to validate.
    param_name : str
        Name of the parameter for error messages.

    Raises
    ------
    ConfigValidationError
        If value is negative.
    """
    if value < 0:
        raise ConfigValidationError(f"{param_name} must be non-negative")


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    param_name: str,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> None:
    """Validate that value is within a range.

    Parameters
    ----------
    value : float
        The value to validate.
    min_val : float
        Minimum value.
    max_val : float
        Maximum value.
    param_name : str
        Name of the parameter for error messages.
    inclusive_min : bool
        Whether min is inclusive.
    inclusive_max : bool
        Whether max is inclusive.

    Raises
    ------
    ConfigValidationError
        If value is out of range.
    """
    min_ok = value >= min_val if inclusive_min else value > min_val
    max_ok = value <= max_val if inclusive_max else value < max_val

    if not (min_ok and max_ok):
        min_bracket = "[" if inclusive_min else "("
        max_bracket = "]" if inclusive_max else ")"
        raise ConfigValidationError(
            f"{param_name} must be in {min_bracket}{min_val}, {max_val}{max_bracket}, "
            f"got {value}"
        )


def validate_less_than_or_equal(
    value1: float | int,
    value2: float | int,
    name1: str,
    name2: str,
) -> None:
    """Validate that value1 <= value2.

    Parameters
    ----------
    value1 : float or int
        First value.
    value2 : float or int
        Second value.
    name1 : str
        Name of first parameter.
    name2 : str
        Name of second parameter.

    Raises
    ------
    ConfigValidationError
        If value1 > value2.
    """
    if value1 > value2:
        raise ConfigValidationError(f"{name1} ({value1}) must be <= {name2} ({value2})")


def validate_normalization_strategy(strategy: str) -> None:
    """Validate normalization strategy parameter."""
    validate_enum_value(
        strategy,
        ("auto", "bounds", "p0", "none"),
        "normalization_strategy",
    )


def validate_precision(precision: str) -> None:
    """Validate precision parameter."""
    validate_enum_value(precision, ("float32", "float64", "auto"), "precision")


def validate_loop_strategy(strategy: str) -> None:
    """Validate loop strategy parameter."""
    validate_enum_value(strategy, ("auto", "scan", "loop"), "loop_strategy")


def validate_lbfgs_line_search(line_search: str) -> None:
    """Validate L-BFGS line search method."""
    validate_enum_value(
        line_search,
        ("wolfe", "strong_wolfe", "backtracking"),
        "lbfgs_line_search",
    )


def validate_multistart_sampler(sampler: str) -> None:
    """Validate multi-start sampler method."""
    validate_enum_value(sampler, ("lhs", "sobol", "halton"), "multistart_sampler")


def validate_warmup_config(
    warmup_iterations: int,
    max_warmup_iterations: int,
    warmup_learning_rate: float,
    loss_plateau_threshold: float,
    gradient_norm_threshold: float,
) -> None:
    """Validate Phase 1 warmup configuration parameters.

    Parameters
    ----------
    warmup_iterations : int
        Number of warmup iterations.
    max_warmup_iterations : int
        Maximum warmup iterations.
    warmup_learning_rate : float
        Learning rate for warmup.
    loss_plateau_threshold : float
        Loss plateau detection threshold.
    gradient_norm_threshold : float
        Gradient norm threshold.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid.
    """
    validate_non_negative(warmup_iterations, "warmup_iterations")
    validate_positive(max_warmup_iterations, "max_warmup_iterations")
    validate_positive(warmup_learning_rate, "warmup_learning_rate")
    validate_positive(loss_plateau_threshold, "loss_plateau_threshold")
    validate_positive(gradient_norm_threshold, "gradient_norm_threshold")
    validate_less_than_or_equal(
        warmup_iterations,
        max_warmup_iterations,
        "warmup_iterations",
        "max_warmup_iterations",
    )


def validate_lbfgs_config(
    history_size: int,
    initial_step_size: float,
    line_search: str,
    exploration_step_size: float,
    refinement_step_size: float,
) -> None:
    """Validate L-BFGS configuration parameters.

    Parameters
    ----------
    history_size : int
        L-BFGS history size.
    initial_step_size : float
        Initial step size.
    line_search : str
        Line search method.
    exploration_step_size : float
        Step size for exploration mode.
    refinement_step_size : float
        Step size for refinement mode.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid.
    """
    validate_positive(history_size, "lbfgs_history_size")
    validate_positive(initial_step_size, "lbfgs_initial_step_size")
    validate_lbfgs_line_search(line_search)
    validate_positive(exploration_step_size, "lbfgs_exploration_step_size")
    validate_positive(refinement_step_size, "lbfgs_refinement_step_size")


def validate_gauss_newton_config(
    max_iterations: int,
    tol: float,
    trust_region_initial: float,
    regularization_factor: float,
) -> None:
    """Validate Phase 2 Gauss-Newton configuration parameters.

    Parameters
    ----------
    max_iterations : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    trust_region_initial : float
        Initial trust region radius.
    regularization_factor : float
        Regularization factor.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid.
    """
    validate_positive(max_iterations, "gauss_newton_max_iterations")
    validate_positive(tol, "gauss_newton_tol")
    validate_positive(trust_region_initial, "trust_region_initial")
    validate_non_negative(regularization_factor, "regularization_factor")


def validate_cg_config(
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    param_threshold: int,
) -> None:
    """Validate Conjugate Gradient solver configuration.

    Parameters
    ----------
    max_iterations : int
        Maximum CG iterations.
    relative_tolerance : float
        Relative tolerance.
    absolute_tolerance : float
        Absolute tolerance.
    param_threshold : int
        Parameter count threshold.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid.
    """
    validate_positive(max_iterations, "cg_max_iterations")
    validate_positive(relative_tolerance, "cg_relative_tolerance")
    validate_positive(absolute_tolerance, "cg_absolute_tolerance")
    validate_positive(param_threshold, "cg_param_threshold")


def validate_group_variance_config(
    enabled: bool,
    lambda_val: float,
    indices: list[tuple[int, int]] | None,
) -> None:
    """Validate group variance regularization configuration.

    Parameters
    ----------
    enabled : bool
        Whether regularization is enabled.
    lambda_val : float
        Regularization strength.
    indices : list of tuple or None
        Parameter group indices.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid when enabled.
    """
    if not enabled:
        return

    validate_positive(lambda_val, "group_variance_lambda")

    if indices is None:
        return

    if not isinstance(indices, list):
        raise ConfigValidationError(
            "group_variance_indices must be a list of (start, end) tuples"
        )

    for idx, item in enumerate(indices):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ConfigValidationError(
                f"group_variance_indices[{idx}] must be a (start, end) tuple"
            )
        start, end = item
        if not isinstance(start, int) or not isinstance(end, int):
            raise ConfigValidationError(
                f"group_variance_indices[{idx}] start/end must be integers"
            )
        if start < 0:
            raise ConfigValidationError(
                f"group_variance_indices[{idx}] start must be non-negative"
            )
        if end <= start:
            raise ConfigValidationError(
                f"group_variance_indices[{idx}] end ({end}) must be > start ({start})"
            )


def validate_streaming_config(
    chunk_size: int,
    checkpoint_frequency: int,
    callback_frequency: int,
) -> None:
    """Validate streaming configuration parameters.

    Parameters
    ----------
    chunk_size : int
        Data chunk size.
    checkpoint_frequency : int
        Checkpoint save frequency.
    callback_frequency : int
        Callback frequency.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid.
    """
    validate_positive(chunk_size, "chunk_size")
    validate_positive(checkpoint_frequency, "checkpoint_frequency")
    validate_positive(callback_frequency, "callback_frequency")


def validate_lr_schedule_config(
    enabled: bool,
    warmup_steps: int,
    decay_steps: int,
    end_value: float,
) -> None:
    """Validate learning rate schedule configuration.

    Parameters
    ----------
    enabled : bool
        Whether schedule is enabled.
    warmup_steps : int
        Warmup steps.
    decay_steps : int
        Decay steps.
    end_value : float
        End learning rate value.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid when enabled.
    """
    if not enabled:
        return

    validate_non_negative(warmup_steps, "lr_schedule_warmup_steps")
    validate_positive(decay_steps, "lr_schedule_decay_steps")
    validate_positive(end_value, "lr_schedule_end_value")


def validate_gradient_clip(clip_value: float | None) -> None:
    """Validate gradient clipping value.

    Parameters
    ----------
    clip_value : float or None
        Gradient clip value.

    Raises
    ------
    ConfigValidationError
        If clip_value is not None and not positive.
    """
    if clip_value is not None:
        validate_positive(clip_value, "gradient_clip_value")


def validate_defense_layer_config(
    enable_warm_start: bool,
    warm_start_threshold: float,
    enable_adaptive_lr: bool,
    lr_refinement: float,
    lr_careful: float,
    lr_default: float,
    enable_cost_guard: bool,
    cost_tolerance: float,
    enable_step_clipping: bool,
    max_step_size: float,
) -> None:
    """Validate 4-layer defense strategy configuration.

    Parameters
    ----------
    enable_warm_start : bool
        Enable Layer 1.
    warm_start_threshold : float
        Warm start threshold.
    enable_adaptive_lr : bool
        Enable Layer 2.
    lr_refinement : float
        Refinement learning rate.
    lr_careful : float
        Careful learning rate.
    lr_default : float
        Default learning rate.
    enable_cost_guard : bool
        Enable Layer 3.
    cost_tolerance : float
        Cost increase tolerance.
    enable_step_clipping : bool
        Enable Layer 4.
    max_step_size : float
        Maximum step size.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid.
    """
    # Layer 1: Warm start detection
    if enable_warm_start:
        validate_range(
            warm_start_threshold,
            0,
            1,
            "warm_start_threshold",
            inclusive_min=False,
            inclusive_max=False,
        )

    # Layer 2: Adaptive learning rate
    if enable_adaptive_lr:
        validate_positive(lr_refinement, "warmup_lr_refinement")
        validate_positive(lr_careful, "warmup_lr_careful")
        validate_less_than_or_equal(
            lr_refinement, lr_careful, "warmup_lr_refinement", "warmup_lr_careful"
        )
        validate_less_than_or_equal(
            lr_careful, lr_default, "warmup_lr_careful", "warmup_learning_rate"
        )

    # Layer 3: Cost-increase guard
    if enable_cost_guard:
        validate_range(
            cost_tolerance,
            0,
            1,
            "cost_increase_tolerance",
            inclusive_min=True,
            inclusive_max=True,
        )

    # Layer 4: Step clipping
    if enable_step_clipping:
        validate_positive(max_step_size, "max_warmup_step_size")


def validate_multistart_config(
    enabled: bool,
    n_starts: int,
    sampler: str,
    elimination_fraction: float,
    elimination_rounds: int,
    batches_per_round: int,
    scale_factor: float,
) -> None:
    """Validate multi-start optimization configuration.

    Parameters
    ----------
    enabled : bool
        Whether multi-start is enabled.
    n_starts : int
        Number of starting points.
    sampler : str
        Sampling method.
    elimination_fraction : float
        Fraction to eliminate per round.
    elimination_rounds : int
        Number of rounds.
    batches_per_round : int
        Batches per round.
    scale_factor : float
        Scale factor for sampling.

    Raises
    ------
    ConfigValidationError
        If any parameter is invalid when enabled.
    """
    if not enabled:
        return

    if n_starts < 1:
        raise ConfigValidationError("n_starts must be >= 1 when enable_multistart=True")

    validate_multistart_sampler(sampler)
    validate_range(
        elimination_fraction,
        0,
        1,
        "elimination_fraction",
        inclusive_min=False,
        inclusive_max=False,
    )
    validate_non_negative(elimination_rounds, "elimination_rounds")
    validate_positive(batches_per_round, "batches_per_round")
    validate_positive(scale_factor, "scale_factor")
