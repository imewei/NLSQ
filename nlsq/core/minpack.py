"""MINPACK-style algorithms for nonlinear least squares optimization.

This module provides JAX implementations of classic MINPACK algorithms including
Levenberg-Marquardt and Trust Region Reflective methods with GPU/TPU acceleration.
These algorithms are the foundation of NLSQ's curve fitting capabilities.

The MINPACK algorithms combine robust optimization strategies with efficient
linear algebra operations, all JIT-compiled for high performance on modern hardware.

Key Components:
    - fit: Unified entry point with workflow-based automatic strategy selection
    - curve_fit: Main high-level interface (SciPy-compatible)
    - CurveFit: Class-based interface with state management
    - Levenberg-Marquardt: Classic damped least squares algorithm
    - Trust Region: Advanced bounded optimization

Algorithms:
    - 'lm': Levenberg-Marquardt (unbounded problems)
    - 'trf': Trust Region Reflective (bounded problems, recommended)
    - 'dogbox': Dogleg with rectangular trust regions

Example:
    >>> from nlsq import CurveFit
    >>> import jax.numpy as jnp
    >>>
    >>> def exponential(x, a, b): return a * jnp.exp(-b * x)
    >>>
    >>> # Class-based interface for reusing compilations
    >>> fitter = CurveFit()
    >>> popt1, pcov1 = fitter.curve_fit(exponential, x1, y1, p0=[2.0, 0.5])
    >>> popt2, pcov2 = fitter.curve_fit(exponential, x2, y2, p0=[2.5, 0.6])

See Also:
    nlsq.curve_fit : Function-based interface
    nlsq.fit : Unified entry point with automatic workflow selection
    nlsq.least_squares : Lower-level optimization interface
    nlsq.trf : Trust Region Reflective implementation
"""

# mypy: ignore-errors

import time
import warnings
from collections.abc import Callable
from inspect import signature
from typing import Any, Literal, Union

import numpy as np

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import cholesky as jax_cholesky
from jax.scipy.linalg import svd as jax_svd

from nlsq.caching.memory_manager import get_memory_manager
from nlsq.caching.unified_cache import UnifiedCache, get_global_cache
from nlsq.common_scipy import EPS
from nlsq.core._optimize import OptimizeWarning
from nlsq.core.least_squares import LeastSquares, prepare_bounds
from nlsq.precision.algorithm_selector import auto_select_algorithm
from nlsq.precision.parameter_estimation import estimate_initial_parameters
from nlsq.result import CurveFitResult
from nlsq.stability.guard import NumericalStabilityGuard
from nlsq.stability.recovery import OptimizationRecovery
from nlsq.types import ArrayLike, ModelFunction
from nlsq.utils.diagnostics import OptimizationDiagnostics
from nlsq.utils.error_messages import OptimizationError
from nlsq.utils.logging import get_logger
from nlsq.utils.validators import InputValidator

__all__ = ["WORKFLOW_PRESETS", "CurveFit", "curve_fit", "fit"]


# Predefined workflow presets for fit() function
# Following GlobalOptimizationConfig.PRESETS pattern
WORKFLOW_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {
        "description": "Standard curve_fit() with default tolerances",
        "tier": "STANDARD",
        "enable_multistart": False,
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
    },
    "quality": {
        "description": "Highest precision with multi-start and tighter tolerances",
        "tier": "STANDARD",
        "enable_multistart": True,
        "n_starts": 20,
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
    },
    "fast": {
        "description": "Speed-optimized with looser tolerances",
        "tier": "STANDARD",
        "enable_multistart": False,
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
    },
    "large_robust": {
        "description": "Chunked processing with multi-start for large datasets",
        "tier": "CHUNKED",
        "enable_multistart": True,
        "n_starts": 10,
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
    },
    "streaming": {
        "description": "AdaptiveHybridStreamingOptimizer for huge datasets",
        "tier": "STREAMING",
        "enable_multistart": False,
        "gtol": 1e-7,
        "ftol": 1e-7,
        "xtol": 1e-7,
    },
    "hpc_distributed": {
        "description": "Multi-GPU/node configuration for HPC clusters",
        "tier": "STREAMING_CHECKPOINT",
        "enable_multistart": True,
        "n_starts": 10,
        "enable_checkpoints": True,
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
    },
}


def fit(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    check_finite: bool = True,
    bounds: tuple = (-float("inf"), float("inf")),
    method: str | None = None,
    workflow: Union[
        str, "LDMemoryConfig", "HybridStreamingConfig", "GlobalOptimizationConfig"
    ] = "auto",
    goal: Union[str, "OptimizationGoal", None] = None,
    **kwargs: Any,
) -> CurveFitResult:
    """Unified curve fitting entry point with automatic workflow selection.

    This function provides a simplified API for curve fitting that automatically
    selects the optimal strategy based on dataset size, available memory, and
    user-specified goals. It coexists with `curve_fit()` and `curve_fit_large()`
    for users who prefer explicit control.

    Parameters
    ----------
    f : callable
        Model function f(x, \\*params) -> y. Must use jax.numpy operations.
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    p0 : array_like, optional
        Initial parameter guess.
    sigma : array_like, optional
        Uncertainties in ydata for weighted fitting.
    absolute_sigma : bool, optional
        Whether sigma represents absolute uncertainties.
    check_finite : bool, optional
        Check for finite input values.
    bounds : tuple, optional
        Parameter bounds as (lower, upper).
    method : str, optional
        Optimization algorithm ('trf', 'lm', or None for auto).
    workflow : str or config object, optional
        Workflow selection:

        - 'auto': Automatically select based on dataset size and memory (default)
        - 'standard': Basic curve_fit() with defaults
        - 'quality': Tightest tolerances + multi-start
        - 'fast': Loose tolerances for speed
        - 'large_robust': Chunked + multi-start for large datasets
        - 'streaming': AdaptiveHybridStreamingOptimizer for huge datasets
        - 'hpc_distributed': Multi-GPU/node configuration
        - Custom config object: LDMemoryConfig, HybridStreamingConfig, or GlobalOptimizationConfig
    goal : str or OptimizationGoal, optional
        Optimization goal that modifies tolerances:

        - 'fast': Prioritize speed, use looser tolerances
        - 'robust' or 'global': Standard tolerances, enable multi-start
        - 'memory_efficient': Minimize memory usage
        - 'quality': Highest precision, tighter tolerances, multi-start

    **kwargs
        Additional optimization parameters (ftol, xtol, gtol, max_nfev, loss).
        These are passed through to the underlying curve_fit() function.

    Returns
    -------
    result : CurveFitResult
        Optimization result. Contains popt, pcov, and additional diagnostics.
        Supports tuple unpacking: popt, pcov = fit(...)

    Raises
    ------
    ValueError
        If workflow name is invalid or goal string is not recognized.

    Examples
    --------
    Basic usage with automatic workflow selection:

    >>> import jax.numpy as jnp
    >>> def model(x, a, b): return a * jnp.exp(-b * x)
    >>> popt, pcov = fit(model, xdata, ydata, p0=[1, 2])

    Using a named workflow:

    >>> result = fit(model, xdata, ydata, p0=[1, 2], workflow='quality',
    ...              bounds=([0, 0], [10, 10]))

    Using a goal to adjust tolerances:

    >>> result = fit(model, xdata, ydata, p0=[1, 2], goal='quality')

    Using a custom config object:

    >>> from nlsq.streaming.large_dataset import LDMemoryConfig
    >>> config = LDMemoryConfig(memory_limit_gb=8.0)
    >>> result = fit(model, xdata, ydata, p0=[1, 2], workflow=config)

    See Also
    --------
    curve_fit : Lower-level API with full control
    curve_fit_large : Specialized API for large datasets
    auto_select_workflow : Automatic workflow selection function
    WORKFLOW_PRESETS : Available workflow presets
    """
    # Import workflow module components
    from nlsq.core.workflow import (
        OptimizationGoal,
        auto_select_workflow,
    )

    # Convert data to arrays for size calculations
    xdata_arr = np.asarray(xdata)
    ydata_arr = np.asarray(ydata)
    n_points = len(ydata_arr)

    # Handle empty data
    if n_points == 0:
        raise ValueError("`ydata` must not be empty!")

    # Determine number of parameters
    if p0 is not None:
        p0_arr = np.atleast_1d(p0)
        n_params = len(p0_arr)
    else:
        # Infer from function signature
        sig = signature(f)
        args = sig.parameters
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n_params = len(args) - 1

    # Convert goal string to OptimizationGoal enum if needed
    goal_enum: OptimizationGoal | None = None
    if goal is not None:
        if isinstance(goal, str):
            goal_lower = goal.lower()
            goal_map = {
                "fast": OptimizationGoal.FAST,
                "robust": OptimizationGoal.ROBUST,
                "global": OptimizationGoal.GLOBAL,
                "memory_efficient": OptimizationGoal.MEMORY_EFFICIENT,
                "quality": OptimizationGoal.QUALITY,
            }
            if goal_lower not in goal_map:
                raise ValueError(
                    f"Unknown goal '{goal}'. Must be one of: {list(goal_map.keys())}"
                )
            goal_enum = goal_map[goal_lower]
        elif isinstance(goal, OptimizationGoal):
            goal_enum = goal
        else:
            raise ValueError(
                f"goal must be a string or OptimizationGoal enum, got {type(goal)}"
            )

    # Process workflow parameter
    # Import config classes for isinstance checks
    from nlsq.global_optimization import GlobalOptimizationConfig
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.large_dataset import LDMemoryConfig

    if isinstance(
        workflow, (LDMemoryConfig, HybridStreamingConfig, GlobalOptimizationConfig)
    ):
        # Custom config object path - route directly to appropriate backend
        return _fit_with_config(
            f=f,
            xdata=xdata_arr,
            ydata=ydata_arr,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            method=method,
            config=workflow,
            goal=goal_enum,
            **kwargs,
        )

    if isinstance(workflow, str):
        workflow_lower = workflow.lower()

        if workflow_lower == "auto":
            # Auto-select workflow based on dataset characteristics
            config = auto_select_workflow(
                n_points=n_points,
                n_params=n_params,
                goal=goal_enum,
            )
            return _fit_with_config(
                f=f,
                xdata=xdata_arr,
                ydata=ydata_arr,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                method=method,
                config=config,
                goal=goal_enum,
                **kwargs,
            )

        elif workflow_lower in WORKFLOW_PRESETS:
            # Named workflow path - apply preset configuration
            preset = WORKFLOW_PRESETS[workflow_lower]
            return _fit_with_preset(
                f=f,
                xdata=xdata_arr,
                ydata=ydata_arr,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                method=method,
                preset=preset,
                goal=goal_enum,
                n_points=n_points,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown workflow '{workflow}'. Must be 'auto', one of "
                f"{list(WORKFLOW_PRESETS.keys())}, or a config object."
            )

    raise ValueError(
        f"workflow must be a string or config object, got {type(workflow)}"
    )


def _fit_with_config(
    f: ModelFunction,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: ArrayLike | None,
    sigma: ArrayLike | None,
    absolute_sigma: bool,
    check_finite: bool,
    bounds: tuple,
    method: str | None,
    config: Any,
    goal: Any,
    **kwargs: Any,
) -> CurveFitResult:
    """Route fit to appropriate backend based on config type."""
    from nlsq.core.workflow import calculate_adaptive_tolerances
    from nlsq.global_optimization import GlobalOptimizationConfig
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig

    n_points = len(ydata)

    # Apply adaptive tolerances if goal is specified and tolerances not in kwargs
    if goal is not None:
        n_params = len(np.atleast_1d(p0)) if p0 is not None else 1
        adaptive_tols = calculate_adaptive_tolerances(n_points, goal)
        for tol_key in ["gtol", "ftol", "xtol"]:
            if tol_key not in kwargs:
                kwargs[tol_key] = adaptive_tols[tol_key]

    if isinstance(config, GlobalOptimizationConfig):
        # Multi-start optimization path
        from nlsq.global_optimization import MultiStartOrchestrator

        orchestrator = MultiStartOrchestrator(config=config)
        result = orchestrator.fit(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=np.asarray(p0) if p0 is not None else None,
            bounds=bounds,
            sigma=np.asarray(sigma) if sigma is not None else None,
            absolute_sigma=absolute_sigma,
            method=method,
            **kwargs,
        )
        return result

    elif isinstance(config, HybridStreamingConfig):
        # Streaming optimization path
        from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer

        # Prepare p0
        if p0 is None:
            sig = signature(f)
            args = sig.parameters
            if len(args) < 2:
                raise ValueError("Unable to determine number of fit parameters.")
            n_params = len(args) - 1
            p0 = np.ones(n_params)
        p0_arr = np.atleast_1d(p0)

        # Prepare bounds
        lb, ub = prepare_bounds(bounds, len(p0_arr))
        bounds_tuple = (
            (lb, ub)
            if not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))
            else None
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config=config)
        result_dict = optimizer.fit(
            data_source=(xdata, ydata),
            func=f,
            p0=p0_arr,
            bounds=bounds_tuple,
            sigma=np.asarray(sigma) if sigma is not None else None,
            absolute_sigma=absolute_sigma,
            callback=kwargs.get("callback"),
            verbose=kwargs.get("verbose", 1),
        )

        # Convert to CurveFitResult
        result = CurveFitResult(result_dict)
        result["model"] = f
        result["xdata"] = xdata
        result["ydata"] = ydata
        result["pcov"] = result_dict.get(
            "pcov", np.full((len(p0_arr), len(p0_arr)), np.inf)
        )
        return result

    elif isinstance(config, LDMemoryConfig):
        # Chunked processing path - use standard curve_fit for small datasets
        # or LargeDatasetFitter for large ones
        if n_points < 1_000_000:
            # Small dataset, use standard curve_fit
            return curve_fit(
                f=f,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                method=method,
                **kwargs,
            )
        else:
            # Large dataset, use LargeDatasetFitter
            fitter = LargeDatasetFitter(
                memory_limit_gb=config.memory_limit_gb,
                config=config,
            )
            result = fitter.fit(
                f,
                xdata,
                ydata,
                p0=np.asarray(p0) if p0 is not None else None,
                bounds=bounds,
                method=method if method else "trf",
                **kwargs,
            )

            # Ensure we have a CurveFitResult
            if not isinstance(result, CurveFitResult):
                # Convert from dict or OptimizeResult
                result = CurveFitResult(result)
                result["model"] = f
                result["xdata"] = xdata
                result["ydata"] = ydata

            return result

    else:
        # Unknown config type - fall back to curve_fit
        return curve_fit(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            method=method,
            **kwargs,
        )


def _fit_with_preset(
    f: ModelFunction,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: ArrayLike | None,
    sigma: ArrayLike | None,
    absolute_sigma: bool,
    check_finite: bool,
    bounds: tuple,
    method: str | None,
    preset: dict[str, Any],
    goal: Any,
    n_points: int,
    **kwargs: Any,
) -> CurveFitResult:
    """Apply a named workflow preset and route to appropriate backend."""
    from nlsq.core.workflow import WorkflowTier, calculate_adaptive_tolerances

    tier_str = preset.get("tier", "STANDARD")
    tier = WorkflowTier[tier_str]

    # Apply preset tolerances (unless overridden in kwargs or by goal)
    if goal is not None:
        # Goal overrides preset tolerances
        adaptive_tols = calculate_adaptive_tolerances(n_points, goal)
        for tol_key in ["gtol", "ftol", "xtol"]:
            if tol_key not in kwargs:
                kwargs[tol_key] = adaptive_tols[tol_key]
    else:
        # Use preset tolerances
        for tol_key in ["gtol", "ftol", "xtol"]:
            if tol_key not in kwargs and tol_key in preset:
                kwargs[tol_key] = preset[tol_key]

    # Check if multi-start is enabled
    enable_multistart = preset.get("enable_multistart", False)
    n_starts = preset.get("n_starts", 10)

    if tier == WorkflowTier.STANDARD:
        # Standard curve_fit path
        if enable_multistart:
            return curve_fit(
                f=f,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                method=method,
                multistart=True,
                n_starts=n_starts,
                **kwargs,
            )
        else:
            return curve_fit(
                f=f,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                method=method,
                **kwargs,
            )

    elif tier == WorkflowTier.CHUNKED:
        # Large dataset with chunking
        from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig

        config = LDMemoryConfig(
            memory_limit_gb=8.0,  # Default, can be overridden
            min_chunk_size=1000,
            max_chunk_size=min(1_000_000, max(10_000, n_points // 10)),
        )

        if n_points < 1_000_000:
            # Small enough for standard curve_fit
            return curve_fit(
                f=f,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                method=method,
                multistart=enable_multistart,
                n_starts=n_starts if enable_multistart else 0,
                **kwargs,
            )
        else:
            fitter = LargeDatasetFitter(
                memory_limit_gb=config.memory_limit_gb,
                config=config,
            )
            result = fitter.fit(
                f,
                xdata,
                ydata,
                p0=np.asarray(p0) if p0 is not None else None,
                bounds=bounds,
                method=method if method else "trf",
                multistart=enable_multistart,
                n_starts=n_starts if enable_multistart else 0,
                **kwargs,
            )

            if not isinstance(result, CurveFitResult):
                result = CurveFitResult(result)
                result["model"] = f
                result["xdata"] = xdata
                result["ydata"] = ydata

            return result

    elif tier in (WorkflowTier.STREAMING, WorkflowTier.STREAMING_CHECKPOINT):
        # Streaming optimization path
        from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        # Prepare p0
        if p0 is None:
            sig = signature(f)
            args = sig.parameters
            if len(args) < 2:
                raise ValueError("Unable to determine number of fit parameters.")
            n_params = len(args) - 1
            p0 = np.ones(n_params)
        p0_arr = np.atleast_1d(p0)

        # Prepare bounds
        lb, ub = prepare_bounds(bounds, len(p0_arr))
        bounds_tuple = (
            (lb, ub)
            if not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))
            else None
        )

        enable_checkpoints = preset.get(
            "enable_checkpoints", tier == WorkflowTier.STREAMING_CHECKPOINT
        )

        config = HybridStreamingConfig(
            normalize=True,
            warmup_iterations=200,
            gauss_newton_tol=kwargs.get("gtol", preset.get("gtol", 1e-8)),
            enable_checkpoints=enable_checkpoints,
            enable_multistart=enable_multistart,
            n_starts=n_starts if enable_multistart else 0,
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config=config)
        result_dict = optimizer.fit(
            data_source=(xdata, ydata),
            func=f,
            p0=p0_arr,
            bounds=bounds_tuple,
            sigma=np.asarray(sigma) if sigma is not None else None,
            absolute_sigma=absolute_sigma,
            callback=kwargs.get("callback"),
            verbose=kwargs.get("verbose", 1),
        )

        result = CurveFitResult(result_dict)
        result["model"] = f
        result["xdata"] = xdata
        result["ydata"] = ydata
        result["pcov"] = result_dict.get(
            "pcov", np.full((len(p0_arr), len(p0_arr)), np.inf)
        )
        return result

    else:
        # Fallback to standard curve_fit
        return curve_fit(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            method=method,
            **kwargs,
        )


def curve_fit(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    *args: Any,
    auto_bounds: bool = False,
    bounds_safety_factor: float = 10.0,
    stability: Literal["auto", "check", False] = False,
    rescale_data: bool = True,
    max_jacobian_elements_for_svd: int = 10_000_000,
    fallback: bool = False,
    max_fallback_attempts: int = 10,
    fallback_verbose: bool = False,
    # Multi-start optimization parameters (Task Group 5)
    multistart: bool = False,
    n_starts: int = 10,
    global_search: bool = False,
    sampler: Literal["lhs", "sobol", "halton"] = "lhs",
    center_on_p0: bool = True,
    scale_factor: float = 1.0,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray] | CurveFitResult:
    """
    Use nonlinear least squares to fit a function to data with GPU/TPU acceleration.

    This is the main user-facing function that provides a drop-in replacement for
    `scipy.optimize.curve_fit` with GPU/TPU acceleration via JAX. The function
    automatically handles JAX JIT compilation, double precision configuration,
    and optimization algorithm selection.

    Parameters
    ----------
    f : callable
        The model function f(x, \\*popt) -> y. Must be JAX-compatible, meaning it should
        use `jax.numpy` instead of `numpy` for mathematical operations to enable
        GPU acceleration and automatic differentiation.
    xdata : array_like
        The independent variable where the data is measured.
    ydata : array_like
        The dependent data, nominally ``f(xdata, *popt)``.
    auto_bounds : bool, optional
        Enable automatic parameter bounds inference from data characteristics.
        When True, reasonable bounds are inferred based on:

        - Data ranges (x and y)
        - Initial parameter guess (p0)
        - Parameter positivity constraints
        - Safety factors to avoid over-constraining

        The inferred bounds are merged with any user-provided bounds via the
        ``bounds`` parameter. User bounds take precedence where specified.
        Default: False.
    bounds_safety_factor : float, optional
        Safety multiplier for automatic bounds (larger = more conservative).
        Only used when auto_bounds=True. Default: 10.0.
    stability : {'auto', 'check', False}, optional
        Control numerical stability checks and automatic fixes:

        - 'auto': Check for stability issues and automatically apply fixes
          (optionally rescale data, normalize parameters, handle NaN/Inf)
        - 'check': Check for stability issues and warn, but don't apply fixes
        - False: Skip stability checks entirely (default)

        When 'auto', detected issues are fixed before optimization:

        - Ill-conditioned data (condition number > 1e10) is rescaled to [0, 1]
          (only if rescale_data=True)
        - Large data ranges (> 1e4) are normalized (only if rescale_data=True)
        - NaN/Inf values are replaced with mean
        - Parameter scale mismatches (ratio > 1e6) are normalized

        Default: False.
    rescale_data : bool, optional
        When stability='auto', controls whether data is automatically rescaled
        to [0, 1] for ill-conditioned or large-range data. Set to False for
        physics applications where data must maintain physical units (e.g.,
        time delays in seconds, scattering vectors in nm^-1). NaN/Inf handling
        and parameter normalization are still applied when stability='auto'.
        Default: True.
    max_jacobian_elements_for_svd : int, optional
        Maximum number of elements in the Jacobian matrix (m x n) for which
        SVD will be computed during stability checks. For larger Jacobians,
        SVD is skipped to avoid O(min(m,n)^2 x max(m,n)) computation overhead.
        NaN/Inf checking is still performed for large Jacobians.

        Examples of element counts:
        - 1M data points x 7 params = 7M elements
        - 100K data points x 100 params = 10M elements

        Set to a larger value if you need condition number checks for large
        problems, or a smaller value to skip SVD more aggressively.
        Default: 10,000,000 (10M elements).
    fallback : bool, optional
        Enable automatic fallback strategies for difficult optimization problems.
        When True, the optimizer will automatically try alternative approaches if
        the initial optimization fails, including:

        - Alternative optimization methods
        - Perturbed initial guesses
        - Relaxed tolerances
        - Inferred parameter bounds
        - Robust loss functions
        - Problem rescaling

        Default: False. Enabling this improves success rate on difficult problems
        but adds overhead when optimizations fail.
    max_fallback_attempts : int, optional
        Maximum number of fallback attempts to try before giving up.
        Only used when fallback=True. Default: 10.
    fallback_verbose : bool, optional
        Print detailed information about fallback attempts.
        Only used when fallback=True. Default: False.
    multistart : bool, optional
        Enable multi-start optimization for global search. When True, generates
        multiple starting points using Latin Hypercube Sampling (or other samplers)
        and evaluates each, selecting the best result. This helps find global
        optima in problems with multiple local minima. Default: False.
    n_starts : int, optional
        Number of starting points for multi-start optimization. Only used when
        multistart=True or global_search=True. Default: 10.
    global_search : bool, optional
        Shorthand for enabling multi-start with n_starts=20. Equivalent to
        multistart=True, n_starts=20. Useful for thorough global search.
        Default: False.
    sampler : {'lhs', 'sobol', 'halton'}, optional
        Sampling strategy for generating starting points in multi-start:

        - 'lhs': Latin Hypercube Sampling (stratified random, default)
        - 'sobol': Sobol quasi-random sequence (deterministic, low-discrepancy)
        - 'halton': Halton quasi-random sequence (deterministic, prime bases)

        Only used when multistart=True or global_search=True. Default: 'lhs'.
    center_on_p0 : bool, optional
        When True, center multi-start samples around the initial guess p0 rather
        than uniformly across the full parameter bounds. This provides more
        focused exploration around a data-informed starting region.
        Only used when multistart=True or global_search=True. Default: True.
    scale_factor : float, optional
        Scale factor for the exploration region when center_on_p0=True.
        Multiplier for the exploration range around p0. Smaller values (0.5)
        mean tighter exploration, larger values (2.0) mean wider exploration.
        Only used when multistart=True or global_search=True. Default: 1.0.
    *args, **kwargs
        Additional arguments passed to CurveFit.curve_fit method.

    Returns
    -------
    popt : ndarray
        Optimal values for the parameters.
    pcov : ndarray
        The estimated covariance of popt.

    When fallback=True, the returned object also contains:

    - fallback_strategy_used : str or None
        Name of the fallback strategy that succeeded, or None if original succeeded
    - fallback_attempts : int
        Number of optimization attempts before success

    When multistart=True or global_search=True, the returned object contains:

    - multistart_diagnostics : dict
        Dictionary with multi-start exploration details including n_starts,
        best_loss, all_losses, exploration time, etc.

    Notes
    -----
    This function creates a CurveFit instance internally and calls its curve_fit method.
    For multiple fits with the same function signature, consider creating a CurveFit
    instance directly to benefit from JAX compilation caching.

    When fallback=True, the optimizer tries increasingly aggressive recovery strategies
    if the initial optimization fails. This is particularly useful for:

    - Poor initial parameter guesses
    - Ill-conditioned problems
    - Problems with outliers
    - Numerically challenging models

    When multistart=True or global_search=True, the optimizer explores multiple
    starting points to find the global optimum. This is particularly useful for:

    - Problems with multiple local minima
    - Complex multi-modal objective functions
    - Cases where the initial guess may not be close to the global optimum

    See Also
    --------
    CurveFit.curve_fit : The underlying method with full parameter documentation
    fit : Unified entry point with automatic workflow selection
    curve_fit_large : For datasets with millions of points requiring special handling
    FallbackOrchestrator : Direct access to fallback system for custom configurations
    MultiStartOrchestrator : Direct access to multi-start system

    Examples
    --------
    Basic usage without fallback:

    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>>
    >>> def exponential(x, a, b):
    ...     return a * jnp.exp(-b * x)
    >>>
    >>> x = np.linspace(0, 4, 50)
    >>> y = 2.5 * np.exp(-1.3 * x) + 0.1 * np.random.normal(size=len(x))
    >>> popt, _pcov = curve_fit(exponential, x, y, p0=[2, 1])

    Using multi-start for global search:

    >>> # Enable multi-start with default n_starts=10
    >>> result = curve_fit(exponential, x, y, p0=[2, 1],
    ...                   bounds=([0, 0], [10, 5]), multistart=True)
    >>>
    >>> # Use global_search shorthand for thorough exploration (n_starts=20)
    >>> result = curve_fit(exponential, x, y, p0=[2, 1],
    ...                   bounds=([0, 0], [10, 5]), global_search=True)
    >>>
    >>> # Customize multi-start with different sampler
    >>> result = curve_fit(exponential, x, y, p0=[2, 1],
    ...                   bounds=([0, 0], [10, 5]),
    ...                   multistart=True, n_starts=15, sampler='sobol')

    Using fallback for difficult problems:

    >>> # Very poor initial guess - may fail without fallback
    >>> result = curve_fit(exponential, x, y, p0=[100, 50], fallback=True)
    >>>
    >>> # Check which strategy was used
    >>> if result.fallback_strategy_used:
    ...     print(f"Recovered using: {result.fallback_strategy_used}")

    Combined multi-start + fallback for maximum robustness:

    >>> result = curve_fit(
    ...     exponential, x, y, p0=[2, 1],
    ...     bounds=([0, 0], [10, 5]),
    ...     global_search=True,
    ...     fallback=True
    ... )
    """
    # Handle global_search shorthand
    if global_search:
        multistart = True
        n_starts = 20

    # Handle automatic bounds inference
    if auto_bounds:
        from nlsq.precision.bound_inference import infer_bounds, merge_bounds

        # Extract p0 from args or kwargs
        p0 = None
        if args and len(args) >= 1:
            p0 = args[0]
        elif "p0" in kwargs:
            p0 = kwargs["p0"]

        if p0 is not None:
            # Infer bounds from data
            inferred_bounds = infer_bounds(
                xdata, ydata, p0, safety_factor=bounds_safety_factor
            )

            # Get user-provided bounds if any
            user_bounds = kwargs.get("bounds", (-np.inf, np.inf))

            # Merge inferred with user bounds (user takes precedence)
            merged_bounds = merge_bounds(inferred_bounds, user_bounds)

            # Update kwargs with merged bounds
            kwargs["bounds"] = merged_bounds

    # Handle numerical stability checks and fixes
    if stability:
        from nlsq.stability.guard import apply_automatic_fixes, check_problem_stability

        logger = get_logger("minpack")

        # Extract p0 from args or kwargs
        p0 = None
        if args and len(args) >= 1:
            p0 = args[0]
        elif "p0" in kwargs:
            p0 = kwargs["p0"]

        # Check stability
        stability_report = check_problem_stability(xdata, ydata, p0, f)

        # Handle based on stability mode
        if stability == "check":
            # Just check and warn
            if stability_report["severity"] == "critical":
                logger.warning(
                    f"Critical stability issues detected ({len(stability_report['issues'])} issues):"
                )
                for issue_type, message, severity in stability_report["issues"]:
                    logger.warning(f"  [{severity.upper()}] {message}")
                if stability_report["recommendations"]:
                    logger.info("Recommendations:")
                    for rec in stability_report["recommendations"]:
                        logger.info(f"  - {rec}")
            elif stability_report["severity"] == "warning":
                logger.warning(
                    f"Stability warnings detected ({len(stability_report['issues'])} issues)"
                )
                for issue_type, message, severity in stability_report["issues"]:
                    logger.warning(f"  [{severity.upper()}] {message}")

        elif stability == "auto":
            # Apply automatic fixes if issues detected
            if stability_report["severity"] in ["warning", "critical"]:
                logger.info(
                    f"Applying automatic fixes for {len(stability_report['issues'])} stability issues..."
                )
                if not rescale_data:
                    logger.info(
                        "  (rescale_data=False: data rescaling disabled for physics applications)"
                    )

                xdata_fixed, ydata_fixed, p0_fixed, fix_info = apply_automatic_fixes(
                    xdata,
                    ydata,
                    p0,
                    stability_report=stability_report,
                    rescale_data=rescale_data,
                )

                # Update data and parameters
                xdata = xdata_fixed
                ydata = ydata_fixed

                if p0_fixed is not None:
                    # Update p0 in kwargs (move from args if needed)
                    kwargs["p0"] = p0_fixed
                    # If p0 was in args, we need to remove it from args
                    if args and len(args) >= 1:
                        args = args[1:]

                # Log applied fixes
                for fix in fix_info["applied_fixes"]:
                    logger.info(f"  - {fix}")

    # Handle multi-start optimization
    if multistart and n_starts > 0:
        from nlsq.global_optimization import (
            GlobalOptimizationConfig,
            MultiStartOrchestrator,
        )

        # Extract p0 from args or kwargs
        p0 = None
        if args and len(args) >= 1:
            p0 = args[0]
        elif "p0" in kwargs:
            p0 = kwargs["p0"]

        # Extract bounds from kwargs
        bounds = kwargs.get("bounds", (-np.inf, np.inf))

        # Create multi-start config
        multistart_config = GlobalOptimizationConfig(
            n_starts=n_starts,
            sampler=sampler,
            center_on_p0=center_on_p0,
            scale_factor=scale_factor,
        )

        # Create orchestrator
        orchestrator = MultiStartOrchestrator(config=multistart_config)

        # Run multi-start optimization
        result = orchestrator.fit(
            f=f,
            xdata=np.asarray(xdata),
            ydata=np.asarray(ydata),
            p0=np.asarray(p0) if p0 is not None else None,
            bounds=bounds,
            **{k: v for k, v in kwargs.items() if k not in ["bounds", "p0"]},
        )

        return result

    # Use fallback orchestrator if requested
    if fallback:
        from nlsq.stability.fallback import FallbackOrchestrator

        orchestrator = FallbackOrchestrator(
            max_attempts=max_fallback_attempts, verbose=fallback_verbose
        )

        # Build kwargs for fallback
        fallback_kwargs = kwargs.copy()
        if args:
            # Handle positional arguments (typically p0)
            if len(args) >= 1:
                fallback_kwargs.setdefault("p0", args[0])
            if len(args) >= 2:
                fallback_kwargs.setdefault("sigma", args[1])
            if len(args) >= 3:
                fallback_kwargs.setdefault("absolute_sigma", args[2])
            # Remaining args would be unusual, pass through kwargs

        result = orchestrator.fit_with_fallback(f, xdata, ydata, **fallback_kwargs)

        # Add empty multi-start diagnostics for consistency
        if (
            not hasattr(result, "multistart_diagnostics")
            and "multistart_diagnostics" not in result
        ):
            result["multistart_diagnostics"] = {
                "n_starts_configured": 0,
                "bypassed": True,
            }

        return result

    # Standard path without fallback or multi-start
    # Extract CurveFit constructor parameters from kwargs
    flength = kwargs.pop("flength", None)
    use_dynamic_sizing = kwargs.pop("use_dynamic_sizing", False)
    cache_config = kwargs.pop("cache_config", None)

    # Create CurveFit instance with appropriate parameters
    jcf = CurveFit(
        flength=flength,
        use_dynamic_sizing=use_dynamic_sizing,
        cache_config=cache_config,
        max_jacobian_elements_for_svd=max_jacobian_elements_for_svd,
    )
    result = jcf.curve_fit(f, xdata, ydata, *args, **kwargs)

    # Add empty multi-start diagnostics for consistency
    result["multistart_diagnostics"] = {
        "n_starts_configured": n_starts if multistart else 0,
        "bypassed": True,  # Bypassed because n_starts was 0 or multistart was False
    }

    # Return enhanced result object that supports both tuple unpacking
    # (popt, pcov = curve_fit(...)) and direct use (result = curve_fit(...))
    return result


def _initialize_feasible(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Initialize feasible parameters for optimization.

    This function initializes feasible parameters for optimization based on the
    lower and upper bounds of the variables. If both bounds are finite, the
    feasible parameters are set to the midpoint between the bounds. If only the
    lower bound is finite, the feasible parameters are set to the lower bound
    plus 1. If only the upper bound is finite, the feasible parameters are set
    to the upper bound minus 1. If neither bound is finite, the feasible
    parameters are set to 1.

    Parameters
    ----------
    lb : np.ndarray
        The lower bounds of the variables.
    ub : np.ndarray
        The upper bounds of the variables.

    Returns
    -------
    np.ndarray
        The initialized feasible parameters.
    """

    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])

    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1

    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1

    return p0


class CurveFit:
    """Main class for nonlinear least squares curve fitting with JAX acceleration.

    This class provides the core curve fitting functionality with JAX JIT compilation,
    automatic differentiation for Jacobian computation, and multiple optimization
    algorithms. It handles data preprocessing, optimization algorithm selection,
    and covariance matrix computation.

    The class maintains compiled versions of fitting functions to avoid recompilation
    overhead when fitting multiple datasets with the same function signature.

    Attributes
    ----------
    flength : float or None
        Fixed data length for input padding to avoid JAX retracing.
    use_dynamic_sizing : bool
        Whether to use dynamic sizing instead of fixed padding.
    logger : Logger
        Internal logger for debugging and performance monitoring.

    Methods
    -------
    curve_fit : Main fitting method
    create_sigma_transform_funcs : Internal method for sigma transformation setup
    """

    def __init__(
        self,
        flength: float | None = None,
        use_dynamic_sizing: bool = False,
        enable_stability: bool = False,
        enable_recovery: bool = False,
        enable_overflow_check: bool = False,
        cache_config: dict[str, Any] | None = None,
        max_jacobian_elements_for_svd: int = 10_000_000,
    ) -> None:
        """Initialize CurveFit instance.

        Parameters
        ----------
        flength : float, optional
            Fixed data length for JAX compilation. Input data is padded to this length
            to avoid recompilation when fitting datasets of different sizes. If None,
            no padding is applied and each dataset size triggers recompilation.
            Ignored when use_dynamic_sizing=True for large datasets.

        use_dynamic_sizing : bool, default False
            Enable dynamic sizing to reduce memory usage. When True, padding is only
            applied when data size is smaller than flength. For large datasets,
            uses actual size to prevent excessive memory allocation. Default False
            maintains backward compatibility with fixed padding behavior.

        enable_stability : bool, default False
            Enable numerical stability checks and fixes (validation, algorithm selection).
            Note: This does NOT include overflow checking which adds overhead.

        enable_recovery : bool, default False
            Enable automatic recovery from optimization failures.

        enable_overflow_check : bool, default False
            Enable overflow/underflow checking in function evaluations. This adds
            ~30% overhead so it's separate from other stability features.

        cache_config : dict, optional
            Configuration for the unified JIT compilation cache. Supported keys:

            - 'maxsize' : int, default=128
                Maximum number of compiled functions to cache
            - 'enable_stats' : bool, default=True
                Track cache statistics (hits, misses, compile_time_ms)
            - 'disk_cache_enabled' : bool, default=False
                Enable disk caching tier (Phase 2 feature)

            If None, uses global cache with default settings.

        max_jacobian_elements_for_svd : int, default 10_000_000
            Maximum Jacobian size (m x n elements) for SVD computation during
            stability checks. SVD is skipped for larger Jacobians to avoid
            O(min(m,n)^2 x max(m,n)) overhead. Only applies when enable_stability=True.

        Notes
        -----
        Fixed length compilation trades memory usage for compilation speed:
        - flength=None: Minimal memory, recompiles for each dataset size
        - flength=large_value: Higher memory, avoids recompilation
        - use_dynamic_sizing=True: Balanced approach for mixed dataset sizes
        """
        self.flength = flength
        self.use_dynamic_sizing = use_dynamic_sizing
        self.logger = get_logger("curve_fit")
        self.create_sigma_transform_funcs()
        self.create_covariance_svd()
        self.ls = LeastSquares()

        # Initialize unified cache
        if cache_config is not None:
            self.cache = UnifiedCache(**cache_config)
        else:
            self.cache = get_global_cache()

        # Initialize stability and recovery systems
        self.enable_stability = enable_stability
        self.enable_recovery = enable_recovery
        self.enable_overflow_check = enable_overflow_check
        self.max_jacobian_elements_for_svd = max_jacobian_elements_for_svd

        if enable_stability:
            self.stability_guard = NumericalStabilityGuard(
                max_jacobian_elements_for_svd=max_jacobian_elements_for_svd
            )
            # Use fast validation mode by default for performance
            self.validator = InputValidator(fast_mode=True)
            self.memory_manager = get_memory_manager()

        if enable_recovery:
            self.recovery = OptimizationRecovery()
            self.diagnostics = OptimizationDiagnostics()

    def update_flength(self, flength: float) -> None:
        """Set the fixed input data length.

        Parameters
        ----------
        flength : float
            The fixed input data length.
        """
        self.flength = flength

    def create_sigma_transform_funcs(self) -> None:
        """Create JIT-compiled sigma transform functions.

        This function creates two JIT-compiled functions: `sigma_transform1d` and
        `sigma_transform2d`, which are used to compute the sigma transform for 1D
        and 2D data, respectively. The functions are stored as attributes of the
        object on which the method is called.
        """

        @jit
        def sigma_transform1d(
            sigma: jnp.ndarray, data_mask: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute the sigma transform for 1D data.

            Parameters
            ----------
            sigma : jnp.ndarray
                The standard deviation of the data.
            data_mask : jnp.ndarray
                A binary mask indicating which data points to use in the fit.

            Returns
            -------
            jnp.ndarray
                The sigma transform for the data.
            """
            transform = 1.0 / sigma
            return transform

        @jit
        def sigma_transform2d(
            sigma: jnp.ndarray, data_mask: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute the sigma transform for 2D data.

            Parameters
            ----------
            sigma : jnp.ndarray
                The standard deviation of the data.
            data_mask : jnp.ndarray
                A binary mask indicating which data points to use in the fit.

            Returns
            -------
            jnp.ndarray
                The sigma transform for the data.
            """
            sigma = jnp.asarray(sigma)
            transform = jax_cholesky(sigma, lower=True)
            return transform

        self.sigma_transform1d = sigma_transform1d
        self.sigma_transform2d = sigma_transform2d
        """For fixed input arrays we need to pad the actual data to match the
        fixed input array size"""

    def create_covariance_svd(self) -> None:
        """Create JIT-compiled SVD function for covariance computation."""

        @jit
        def covariance_svd(jac):
            _, s, VT = jax_svd(jac, full_matrices=False)
            return s, VT

        self.covariance_svd = covariance_svd

    def _select_tr_solver(
        self, solver: str, m: int, n: int, batch_size: int | None = None
    ) -> str | None:
        """Select appropriate trust region solver based on solver type and problem size.

        Parameters
        ----------
        solver : str
            Requested solver type
        m : int
            Number of data points
        n : int
            Number of parameters
        batch_size : int, optional
            Batch size for minibatch processing

        Returns
        -------
        str or None
            Trust region solver to use, or None to use default
        """
        if solver == "auto":
            # Auto-select based on problem size
            if m * n < 10000:  # Small problems
                return "exact"  # Use SVD-based exact solver
            else:  # Large problems
                return "lsmr"  # Use iterative LSMR solver
        elif solver == "svd":
            return "exact"  # SVD-based exact solver
        elif solver == "cg":
            return "lsmr"  # LSMR is the closest to CG in current implementation
        elif solver == "lsqr":
            return "lsmr"  # Direct mapping
        elif solver == "minibatch":
            # For minibatch, we'll use lsmr but need to handle batching separately
            # This is a placeholder - full minibatch implementation would require
            # more substantial changes to the optimization loop
            self.logger.warning(
                "Minibatch solver not fully implemented yet. Using LSMR solver.",
                requested_batch_size=batch_size,
            )
            return "lsmr"
        else:
            return None  # Use default

    def pad_fit_data(
        self, xdata: np.ndarray, ydata: np.ndarray, xdims: int, len_diff: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad fit data to match the fixed input data length.

        This function pads the input data arrays with small values to match the
        fixed input data length to avoid JAX retracing the JITted functions.
        The padding is added along the second dimension of the `xdata` array
        if it's multidimensional data otherwise along the first dimension. The
        small values are chosen to be `EPS`, a global constant defined as a
        very small positive value which avoids numerical issues.

        Parameters
        ----------
        xdata : np.ndarray
            The independent variables of the data.
        ydata : np.ndarray
            The dependent variables of the data.
        xdims : int
            The number of dimensions in the `xdata` array.
        len_diff : int
            The difference in length between the data arrays and the fixed input data length.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The padded `xdata` and `ydata` arrays.
        """

        if xdims > 1:
            xpad = EPS * np.ones([xdims, len_diff])
            xdata = np.concatenate([xdata, xpad], axis=1)
        else:
            xpad = EPS * np.ones([len_diff])
            xdata = np.concatenate([xdata, xpad])
        ypad = EPS * np.ones([len_diff])
        ydata = np.concatenate([ydata, ypad])
        return xdata, ydata

    def _setup_sigma_transform(
        self,
        sigma: np.ndarray | None,
        ydata: np.ndarray,
        data_mask: np.ndarray,
        len_diff: int,
        m: int,
    ):
        """Setup sigma transformation for weighted least squares.

        Parameters
        ----------
        sigma : np.ndarray | None
            Uncertainty in ydata (1-D errors or 2-D covariance matrix)
        ydata : np.ndarray
            Dependent data array
        data_mask : np.ndarray
            Boolean mask for valid data points
        len_diff : int
            Difference in length for padding
        m : int
            Original number of data points

        Returns
        -------
        transform : callable | None
            Transformation function for sigma

        Raises
        ------
        ValueError
            If sigma has incorrect shape or is not positive definite
        """
        if sigma is None:
            return None

        if not isinstance(sigma, np.ndarray):
            raise ValueError("Sigma must be numpy array.")

        ysize = ydata.size - len_diff

        # if 1-D, sigma are errors, define transform = 1/sigma
        if sigma.shape == (ysize,):
            if len_diff > 0:
                sigma = np.concatenate([sigma, np.ones([len_diff])])
            return self.sigma_transform1d(sigma, data_mask)

        # if 2-D, sigma is the covariance matrix,
        # define transform = L such that L L^T = C
        elif sigma.shape == (ysize, ysize):
            try:
                if len_diff >= 0:
                    sigma_padded = np.identity(m + len_diff)
                    sigma_padded[:m, :m] = sigma
                    sigma = sigma_padded
                # scipy.linalg.cholesky requires lower=True to return L L^T = A
                return self.sigma_transform2d(sigma, data_mask)
            except (np.linalg.LinAlgError, ValueError) as e:
                # Check eigenvalues to provide more informative error
                try:
                    eigenvalues = np.linalg.eigvalsh(sigma[:ysize, :ysize])
                    min_eig = np.min(eigenvalues)
                    if min_eig <= 0:
                        raise ValueError(
                            f"Covariance matrix `sigma` is not positive definite. "
                            f"Minimum eigenvalue: {min_eig:.6e}. "
                            "All eigenvalues must be positive."
                        ) from e
                except Exception as eigenvalue_error:
                    # If eigenvalue check fails, provide generic error (log for debugging)
                    self.logger.debug(
                        f"Eigenvalue check failed (non-critical): {eigenvalue_error}"
                    )
                raise ValueError(
                    "Failed to compute Cholesky decomposition of `sigma`. "
                    "The covariance matrix must be symmetric and positive definite."
                ) from e
        else:
            raise ValueError("`sigma` has incorrect shape.")

    def _compute_covariance(
        self,
        res,
        ysize: int,
        p0: np.ndarray,
        absolute_sigma: bool,
    ) -> tuple[np.ndarray, bool]:
        """Compute covariance matrix from optimization result.

        Parameters
        ----------
        res : OptimizeResult
            Result from least_squares optimization
        ysize : int
            Number of data points
        p0 : np.ndarray
            Initial parameter guess
        absolute_sigma : bool
            Whether sigma is absolute or relative

        Returns
        -------
        pcov : np.ndarray
            Covariance matrix
        warn_cov : bool
            Whether to warn about covariance estimation
        """
        cost = 2 * res.cost  # res.cost is half sum of squares!

        # Do Moore-Penrose inverse discarding zero singular values
        outputs = self.covariance_svd(res.jac)
        # Convert JAX arrays to NumPy more efficiently using np.asarray
        s, VT = (np.asarray(output) for output in outputs)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[: s.size]
        pcov = np.dot(VT.T / s**2, VT)

        warn_cov = False
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((len(res.x), len(res.x)), dtype=float)
            pcov.fill(np.inf)
            warn_cov = True
        elif not absolute_sigma:
            if ysize > p0.size:
                s_sq = cost / (ysize - p0.size)
                pcov = pcov * s_sq
            else:
                pcov.fill(np.inf)
                warn_cov = True

        if warn_cov:
            self.logger.warning(
                "Covariance could not be estimated",
                reason="insufficient_data" if ysize <= p0.size else "singular_jacobian",
            )
            warnings.warn(
                "Covariance of the parameters could not be estimated",
                stacklevel=2,
                category=OptimizeWarning,
            )

        return pcov, warn_cov

    def _determine_parameter_count(
        self,
        f: Callable,
        p0: np.ndarray | None | str,
        xdata: np.ndarray | None = None,
        ydata: np.ndarray | None = None,
    ) -> tuple[int, np.ndarray | None]:
        """Determine number of fit parameters from p0 or function signature.

        Parameters
        ----------
        f : Callable
            The fit function
        p0 : np.ndarray | None | 'auto'
            Initial parameter guess. If 'auto', will estimate from data
            if xdata and ydata are provided. If None, uses default behavior
            (determined by bounds in _prepare_bounds_and_initial_guess).
        xdata : np.ndarray, optional
            Independent variable data (for auto p0 estimation)
        ydata : np.ndarray, optional
            Dependent variable data (for auto p0 estimation)

        Returns
        -------
        n : int
            Number of parameters
        p0 : np.ndarray | None
            Validated p0 array (or None if auto-estimation not requested)
        """
        # If p0 is explicitly provided (not None or 'auto'), use it
        if p0 is not None and not (isinstance(p0, str) and p0 == "auto"):
            p0 = np.atleast_1d(p0)
            n = p0.size
            return n, p0

        # Only auto-estimate if p0='auto' is explicitly requested
        # (not when p0=None, to preserve backward compatibility)
        if (
            isinstance(p0, str)
            and p0 == "auto"
            and xdata is not None
            and ydata is not None
        ):
            try:
                p0_estimated = estimate_initial_parameters(f, xdata, ydata, p0)
                p0 = np.atleast_1d(p0_estimated)
                n = p0.size
                self.logger.debug(
                    "Auto-estimated initial parameters",
                    p0=p0.tolist(),
                    n_params=n,
                )
                return n, p0
            except Exception as e:
                # If auto-estimation fails, fall back to default behavior
                self.logger.warning(
                    "Auto p0 estimation failed, using defaults",
                    error=str(e),
                )

        # Fall back: determine n from function signature, p0 stays None
        # (will be initialized to defaults in _prepare_bounds_and_initial_guess)
        sig = signature(f)
        args = sig.parameters
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1

        return n, p0

    def _validate_solver_config(self, solver: str, batch_size: int | None) -> None:
        """Validate solver and batch_size configuration.

        Parameters
        ----------
        solver : str
            Solver type
        batch_size : int | None
            Batch size for minibatch solver

        Raises
        ------
        ValueError
            If solver or batch_size is invalid
        """
        valid_solvers = {"auto", "svd", "cg", "lsqr", "minibatch"}
        if solver not in valid_solvers:
            raise ValueError(
                f"Invalid solver '{solver}'. Must be one of {valid_solvers}."
            )

        if solver == "minibatch" and batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when using minibatch solver.")

    def _prepare_bounds_and_initial_guess(
        self, bounds: tuple, n: int, p0: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare bounds and initialize p0 if needed.

        Parameters
        ----------
        bounds : tuple
            Bounds tuple (lower, upper)
        n : int
            Number of parameters
        p0 : np.ndarray | None
            Initial parameter guess

        Returns
        -------
        lb : np.ndarray
            Lower bounds
        ub : np.ndarray
            Upper bounds
        p0 : np.ndarray
            Initial parameter guess (clipped to bounds if necessary)
        """
        lb, ub = prepare_bounds(bounds, n)
        if p0 is None:
            p0 = _initialize_feasible(lb, ub)
        else:
            # Clip auto-estimated p0 to bounds to ensure feasibility
            p0 = np.clip(p0, lb, ub)

        return lb, ub, p0

    def _select_optimization_method(
        self,
        method: str | None,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple,
        kwargs: dict,
    ) -> str:
        """Select optimization method, auto-selecting if needed.

        Parameters
        ----------
        method : str | None
            Optimization method (None for auto-selection)
        f : Callable
            Fit function
        xdata : np.ndarray
            X data
        ydata : np.ndarray
            Y data
        p0 : np.ndarray
            Initial parameter guess
        bounds : tuple
            Bounds tuple
        kwargs : dict
            Additional optimization parameters

        Returns
        -------
        method : str
            Selected optimization method
        """
        if method is None:
            if self.enable_stability:
                recommendations = auto_select_algorithm(f, xdata, ydata, p0, bounds)
                method = recommendations["algorithm"]
                self.logger.info(
                    "Auto-selected algorithm",
                    method=method,
                    loss=recommendations.get("loss", "linear"),
                )
                # Apply recommended parameters to kwargs
                for key in ["ftol", "xtol", "gtol", "max_nfev", "x_scale"]:
                    if key in recommendations and key not in kwargs:
                        kwargs[key] = recommendations[key]
            else:
                method = "trf"

        return method

    def _validate_and_sanitize_inputs(
        self,
        f: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and sanitize curve fit inputs if stability is enabled.

        Parameters
        ----------
        f : Callable
            Fit function
        xdata : np.ndarray
            X data
        ydata : np.ndarray
            Y data
        p0 : np.ndarray
            Initial parameter guess

        Returns
        -------
        xdata : np.ndarray
            Cleaned X data
        ydata : np.ndarray
            Cleaned Y data
        """
        if self.enable_stability:
            try:
                errors, warnings_list, xdata_clean, ydata_clean = (
                    self.validator.validate_curve_fit_inputs(f, xdata, ydata, p0)
                )

                if errors:
                    error_msg = f"Input validation failed: {'; '.join(errors)}"
                    self.logger.error("Input validation failed", error=error_msg)
                    raise ValueError(error_msg)

                for warning in warnings_list:
                    self.logger.warning("Input validation warning", warning=warning)

                xdata = xdata_clean
                ydata = ydata_clean

            except ValueError as e:
                if "too many values to unpack" not in str(e):
                    self.logger.error("Input validation failed", error=str(e))
                raise

        return xdata, ydata

    def _convert_and_validate_arrays(
        self,
        xdata: np.ndarray | tuple | list,
        ydata: np.ndarray,
        check_finite: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert inputs to arrays and validate finiteness.

        Parameters
        ----------
        xdata : array-like
            X data
        ydata : array-like
            Y data
        check_finite : bool
            Whether to check for finite values

        Returns
        -------
        xdata : np.ndarray
            X data as array
        ydata : np.ndarray
            Y data as array
        """
        # Convert ydata
        if check_finite:
            ydata = np.asarray_chkfinite(ydata, float)
        else:
            ydata = np.asarray(ydata, float)

        # Convert xdata
        if hasattr(xdata, "__array__") or isinstance(
            xdata, (list, tuple, np.ndarray, jnp.ndarray)
        ):
            if check_finite:
                xdata = np.asarray_chkfinite(xdata, float)
            else:
                xdata = np.asarray(xdata, float)
        else:
            raise ValueError("X needs arrays")

        if ydata.size == 0:
            raise ValueError("`ydata` must not be empty!")

        return xdata, ydata

    def _validate_data_lengths(
        self, xdata: np.ndarray, ydata: np.ndarray
    ) -> tuple[int, int]:
        """Validate that X and Y data lengths match.

        Parameters
        ----------
        xdata : np.ndarray
            X data
        ydata : np.ndarray
            Y data

        Returns
        -------
        m : int
            Data length
        xdims : int
            X data dimensionality

        Raises
        ------
        ValueError
            If X and Y lengths don't match
        """
        m = len(ydata)
        xdims = xdata.ndim
        xlen = len(xdata) if xdims == 1 else len(xdata[0])
        if xlen != m:
            raise ValueError("X and Y data lengths dont match")

        return m, xdims

    def _setup_data_mask_and_padding(
        self, data_mask: np.ndarray | None, m: int
    ) -> tuple[np.ndarray, bool, int, bool]:
        """Setup data mask and compute padding parameters.

        Parameters
        ----------
        data_mask : np.ndarray | None
            Optional data mask
        m : int
            Data length

        Returns
        -------
        data_mask : np.ndarray
            Data mask array
        none_mask : bool
            Whether data_mask was None on input
        len_diff : int
            Length difference for padding
        should_pad : bool
            Whether padding is needed
        """
        none_mask = data_mask is None
        should_pad = False
        len_diff = 0

        if self.flength is not None:
            len_diff = self.flength - m
            if self.use_dynamic_sizing:
                should_pad = len_diff > 0
            else:
                should_pad = len_diff >= 0

            if data_mask is not None:
                if len(data_mask) != m:
                    raise ValueError("Data mask doesn't match data lengths.")
            else:
                data_mask = np.ones(m, dtype=bool)
                if should_pad and len_diff > 0:
                    data_mask = np.concatenate(
                        [data_mask, np.zeros(len_diff, dtype=bool)]
                    )
        else:
            data_mask = np.ones(m, dtype=bool)

        return data_mask, none_mask, len_diff, should_pad

    def _apply_padding_if_needed(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xdims: int,
        m: int,
        len_diff: int,
        should_pad: bool,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Apply padding to data if needed.

        Parameters
        ----------
        xdata : np.ndarray
            X data
        ydata : np.ndarray
            Y data
        xdims : int
            X data dimensionality
        m : int
            Data length
        len_diff : int
            Length difference
        should_pad : bool
            Whether padding is needed

        Returns
        -------
        xdata : np.ndarray
            Possibly padded X data
        ydata : np.ndarray
            Possibly padded Y data
        len_diff : int
            Updated length difference
        """
        if self.flength is not None and should_pad:
            if len_diff > 0:
                xdata, ydata = self.pad_fit_data(xdata, ydata, xdims, len_diff)
            elif len_diff < 0 and not self.use_dynamic_sizing:
                self.logger.debug(
                    "Data size exceeds fixed length, JIT retracing may occur",
                    data_size=m,
                    flength=self.flength,
                )
        elif self.use_dynamic_sizing and self.flength is not None and len_diff < 0:
            len_diff = 0

        return xdata, ydata, len_diff

    def _prepare_curve_fit_inputs(
        self,
        f: Callable,
        xdata: np.ndarray | tuple[np.ndarray],
        ydata: np.ndarray,
        p0: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray],
        solver: str,
        batch_size: int | None,
        method: str | None,
        check_finite: bool,
        data_mask: np.ndarray | None,
        kwargs: dict,
    ) -> tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        np.ndarray,
        np.ndarray,
        int,
        int,
        bool,
        bool,
    ]:
        """Prepare and validate inputs for curve fitting.

        This method orchestrates the input preparation pipeline by calling
        focused helper methods for each validation step.

        Returns
        -------
        n : int
            Number of parameters
        p0 : np.ndarray
            Initial parameter guess
        xdata : np.ndarray
            Validated x data
        ydata : np.ndarray
            Validated y data
        data_mask : np.ndarray
            Data mask array
        method : str
            Optimization method
        lb : np.ndarray
            Lower bounds
        ub : np.ndarray
            Upper bounds
        m : int
            Data length
        len_diff : int
            Length difference for padding
        should_pad : bool
            Whether padding is needed
        none_mask : bool
            Whether data_mask was None on input
        """
        # Step 1: Determine parameter count and auto-estimate p0 if needed
        n, p0 = self._determine_parameter_count(f, p0, xdata, ydata)

        # Step 2: Validate solver configuration
        self._validate_solver_config(solver, batch_size)

        # Step 3: Log curve fit start
        # Check if bounds are provided (not infinite)
        has_bounds = False
        if isinstance(bounds, tuple) and len(bounds) == 2:
            lower_b, upper_b = bounds
            lower_arr = np.atleast_1d(lower_b)
            upper_arr = np.atleast_1d(upper_b)
            has_bounds = not (
                np.all(np.isneginf(lower_arr)) and np.all(np.isposinf(upper_arr))
            )

        self.logger.info(
            "Starting curve fit",
            n_params=n,
            n_data_points=len(ydata),
            method=method if method else "trf",
            solver=solver,
            batch_size=batch_size if solver == "minibatch" else None,
            has_bounds=has_bounds,
            dynamic_sizing=self.use_dynamic_sizing,
        )

        # Step 4: Prepare bounds and initial guess
        lb, ub, p0 = self._prepare_bounds_and_initial_guess(bounds, n, p0)

        # Step 5: Select optimization method
        method = self._select_optimization_method(
            method, f, xdata, ydata, p0, bounds, kwargs
        )

        # Step 6: Validate and sanitize inputs (if stability enabled)
        xdata, ydata = self._validate_and_sanitize_inputs(f, xdata, ydata, p0)

        # Step 7: Convert to arrays and validate finiteness
        xdata, ydata = self._convert_and_validate_arrays(xdata, ydata, check_finite)

        # Step 8: Validate data lengths
        m, xdims = self._validate_data_lengths(xdata, ydata)

        # Step 9: Setup data mask and padding parameters
        data_mask, none_mask, len_diff, should_pad = self._setup_data_mask_and_padding(
            data_mask, m
        )

        # Step 10: Apply padding if needed
        xdata, ydata, len_diff = self._apply_padding_if_needed(
            xdata, ydata, xdims, m, len_diff, should_pad
        )

        return (
            n,
            p0,
            xdata,
            ydata,
            data_mask,
            method,
            lb,
            ub,
            m,
            len_diff,
            should_pad,
            none_mask,
        )

    def _curve_fit_hybrid_streaming(
        self,
        f: Callable,
        xdata: np.ndarray | tuple[np.ndarray],
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        absolute_sigma: bool = False,
        check_finite: bool = True,
        bounds: tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
        callback: Callable | None = None,
        **kwargs,
    ) -> CurveFitResult:
        """Handle curve fitting with hybrid_streaming method.

        This method delegates to AdaptiveHybridStreamingOptimizer for
        large-scale optimization with parameter normalization, Adam warmup,
        and streaming Gauss-Newton.

        Parameters
        ----------
        f : Callable
            Model function f(x, *params) -> predictions
        xdata : array_like
            Independent variable data
        ydata : array_like
            Dependent variable data
        p0 : array_like, optional
            Initial parameter guess
        sigma : array_like, optional
            Uncertainties in ydata
        absolute_sigma : bool, default=False
            Whether sigma is absolute or relative
        check_finite : bool, default=True
            Check for finite input values
        bounds : tuple, optional
            Parameter bounds as (lower, upper)
        callback : callable, optional
            Progress callback function
        **kwargs
            Additional parameters (verbose, config overrides)

        Returns
        -------
        result : CurveFitResult
            Optimization result compatible with scipy.optimize.curve_fit
        """
        from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
        from nlsq.streaming.hybrid_config import HybridStreamingConfig

        # Convert inputs to arrays
        if check_finite:
            xdata = np.asarray_chkfinite(xdata, float)
            ydata = np.asarray_chkfinite(ydata, float)
        else:
            xdata = np.asarray(xdata, float)
            ydata = np.asarray(ydata, float)

        # Determine parameter count and prepare p0
        if p0 is None:
            n, p0 = self._determine_parameter_count(f, p0, xdata, ydata)
        else:
            p0 = np.atleast_1d(p0)
            n = p0.size

        # Prepare bounds
        lb, ub, p0 = self._prepare_bounds_and_initial_guess(bounds, n, p0)
        bounds_tuple = (
            (lb, ub)
            if not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))
            else None
        )

        # Extract verbosity from kwargs
        verbose = kwargs.pop("verbose", 1)

        # Create configuration (allow kwargs to override defaults)
        config_overrides = {}
        for key in list(kwargs.keys()):
            if hasattr(HybridStreamingConfig, key):
                config_overrides[key] = kwargs.pop(key)

        config = (
            HybridStreamingConfig(**config_overrides)
            if config_overrides
            else HybridStreamingConfig()
        )

        # Create optimizer
        optimizer = AdaptiveHybridStreamingOptimizer(config=config)

        # Run optimization
        result_dict = optimizer.fit(
            data_source=(xdata, ydata),
            func=f,
            p0=p0,
            bounds=bounds_tuple,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            callback=callback,
            verbose=verbose,
        )

        # Convert to CurveFitResult format
        # The result_dict from AdaptiveHybridStreamingOptimizer should have:
        # 'x', 'success', 'message', 'fun', 'pcov', 'perr', 'streaming_diagnostics'
        result = CurveFitResult(result_dict)
        result["model"] = f
        result["xdata"] = xdata
        result["ydata"] = ydata
        result["pcov"] = result_dict.get("pcov", np.full((n, n), np.inf))

        return result

    def _run_optimization(
        self,
        f: Callable,
        p0: np.ndarray,
        xdata: np.ndarray,
        ydata: np.ndarray,
        data_mask: np.ndarray,
        transform,
        bounds: tuple[np.ndarray, np.ndarray],
        method: str,
        solver: str,
        batch_size: int | None,
        jac: Callable | None,
        m: int,
        n: int,
        sigma: np.ndarray | None,
        timeit: bool,
        callback: Callable | None,
        kwargs: dict,
    ) -> tuple:
        """Setup and run the optimization.

        Returns
        -------
        res : OptimizeResult
            Optimization result
        jnp_xdata : jnp.ndarray
            JAX array of x data
        ctime : float
            Conversion time (if timeit=True)
        """
        # Validate kwargs
        if "args" in kwargs:
            raise ValueError("'args' is not a supported keyword argument.")

        if "max_nfev" not in kwargs:
            kwargs["max_nfev"] = kwargs.pop("maxfev", None)

        # Determine the appropriate solver and configure tr_solver
        tr_solver = self._select_tr_solver(solver, m, n, batch_size)
        if tr_solver is not None:
            kwargs["tr_solver"] = tr_solver

        # Handle minibatch processing if requested
        if solver == "minibatch":
            if batch_size is None:
                batch_size = min(1000, max(100, m // 10))
                self.logger.debug(f"Using default batch size: {batch_size}")

            self.logger.info(
                "Minibatch processing requested",
                batch_size=batch_size,
                n_batches=m // batch_size + (1 if m % batch_size > 0 else 0),
            )

        # Convert to JAX arrays (no .block_until_ready() - one-time setup)
        st = time.time()
        jnp_xdata = jnp.asarray(xdata)
        jnp_ydata = jnp.asarray(ydata)
        ctime = time.time() - st

        jnp_data_mask = jnp.array(data_mask, dtype=bool)

        # Check memory requirements if stability is enabled
        if self.enable_stability:
            memory_required = self.memory_manager.predict_memory_requirement(
                m, n, method
            )
            is_available, msg = self.memory_manager.check_memory_availability(
                memory_required
            )
            if not is_available:
                self.logger.warning("Memory constraint detected", message=msg)
                kwargs["tr_solver"] = "lsmr"

        # Start curve fit timer and call least squares
        with self.logger.timer("curve_fit"):
            self.logger.debug(
                "Calling least squares solver",
                has_sigma=sigma is not None,
                has_jacobian=jac is not None,
            )

            # Create wrapper for overflow checking if enabled
            if self.enable_overflow_check:
                original_f = f

                def stable_f(x, *params):
                    result = original_f(x, *params)
                    max_val = jnp.max(jnp.abs(result))
                    result = jnp.where(
                        max_val > 1e8,
                        jnp.clip(result, -1e10, 1e10),
                        result,
                    )
                    return result

                f_to_use = stable_f
            else:
                f_to_use = f

            try:
                res = self.ls.least_squares(
                    f_to_use,
                    p0,
                    jac=jac,
                    xdata=jnp_xdata,
                    ydata=jnp_ydata,
                    data_mask=jnp_data_mask,
                    transform=transform,
                    bounds=bounds,
                    method=method,
                    timeit=timeit,
                    callback=callback,
                    **kwargs,
                )
            except Exception as e:
                if self.enable_recovery:
                    self.logger.warning(
                        "Optimization failed, attempting recovery", error=str(e)
                    )
                    recovery_state = {
                        "params": p0,
                        "xdata": xdata,
                        "ydata": ydata,
                        "method": method if method is not None else "trf",
                        "bounds": bounds,
                    }

                    success, result = self.recovery.recover_from_failure(
                        "optimization_error",
                        recovery_state,
                        lambda **state: self.ls.least_squares(
                            f_to_use,
                            state["params"],
                            jac=jac,
                            xdata=jnp.asarray(state["xdata"]),
                            ydata=jnp.asarray(state["ydata"]),
                            data_mask=jnp_data_mask,
                            transform=transform,
                            bounds=state["bounds"],
                            method=state["method"],
                            timeit=timeit,
                            callback=callback,
                            **kwargs,
                        ),
                    )

                    if success:
                        res = result
                    else:
                        raise RuntimeError(
                            f"Optimization failed and recovery unsuccessful: {e}"
                        ) from e
                else:
                    raise

        if not res.success:
            self.logger.error(
                "Optimization failed", reason=res.message, status=res.status
            )
            # Extract tolerances for enhanced error message
            gtol = kwargs.get("gtol", 1e-8)
            ftol = kwargs.get("ftol", 1e-8)
            xtol = kwargs.get("xtol", 1e-8)
            max_nfev = kwargs.get("max_nfev")
            if max_nfev is None:
                max_nfev = len(p0) * 100  # Default estimate
            raise OptimizationError(res, gtol, ftol, xtol, max_nfev)

        return res, jnp_xdata, ctime

    def curve_fit(
        self,
        f: Callable,
        xdata: np.ndarray | tuple[np.ndarray],
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        absolute_sigma: bool = False,
        check_finite: bool = True,
        bounds: tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
        method: str | None = None,
        solver: str = "auto",
        batch_size: int | None = None,
        jac: Callable | None = None,
        data_mask: np.ndarray | None = None,
        timeit: bool = False,
        return_eval: bool = False,
        callback: Callable | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Use non-linear least squares to fit a function, f, to data.
        Assumes ``ydata = f(xdata, \\*params) + eps``.

        Parameters
        ----------
        method : str | None, optional
            Optimization algorithm. Options:
            - 'trf' (default): Trust Region Reflective
            - 'lm': Levenberg-Marquardt
            - 'hybrid_streaming': Adaptive Hybrid Streaming Optimizer (for large datasets)
            If None, auto-selects 'trf'.
        """
        # Check for hybrid_streaming method early and delegate
        if method == "hybrid_streaming":
            return self._curve_fit_hybrid_streaming(
                f=f,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                callback=callback,
                **kwargs,
            )

        """
        Use non-linear least squares to fit a function, f, to data.
        Assumes ``ydata = f(xdata, \\*params) + eps``.

        Parameters
        ----------
        f : callable
            The model function, f(x, ...). It must take the independent
            variable as the first argument and the parameters to fit as
            separate remaining arguments.
        xdata : array_like or object
            The independent variable where the data is measured.
            Should usually be an M-length sequence or an (k,M)-shaped array for
            functions with k predictors, but can actually be any object.
        ydata : array_like
            The dependent data, a length M array - nominally ``f(xdata, ...)``.
        p0 : array_like or 'auto' or None, optional
            Initial guess for the parameters (length N). If None or 'auto',
            initial parameters will be estimated automatically from the data
            characteristics. For best results with auto estimation, use
            well-scaled data or provide custom estimation via f.estimate_p0().
        sigma : None or M-length sequence or MxM array, optional
            Determines the uncertainty in `ydata`. If we define residuals as
            ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
            depends on its number of dimensions:
            - A 1-D `sigma` should contain values of standard deviations of
            errors in `ydata`. In this case, the optimized function is
            ``chisq = sum((r / sigma) ** 2)``.
            - A 2-D `sigma` should contain the covariance matrix of
            errors in `ydata`. In this case, the optimized function is
            ``chisq = r.T @ inv(sigma) @ r``.
            .. versionadded:: 0.19

            None (default) is equivalent of 1-D `sigma` filled with ones.
        absolute_sigma : bool, optional
            If True, `sigma` is used in an absolute sense and the estimated parameter
            covariance `pcov` reflects these absolute values.
            If False (default), only the relative magnitudes of the `sigma` values matter.
            The returned parameter covariance matrix `pcov` is based on scaling
            `sigma` by a constant factor. This constant is set by demanding that the
            reduced `chisq` for the optimal parameters `popt` when using the
            *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
            match the sample variance of the residuals after the fit. Default is False.
            Mathematically,
            ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
        check_finite : bool, optional
            If True, check that the input arrays do not contain nans of infs,
            and raise a ValueError if they do. Setting this parameter to
            False may silently produce nonsensical results if the input arrays
            do contain nans. Default is True.
        bounds : 2-tuple of array_like, optional
            Lower and upper bounds on parameters. Defaults to no bounds.
            Each element of the tuple must be either an array with the length equal
            to the number of parameters, or a scalar (in which case the bound is
            taken to be the same for all parameters). Use ``np.inf`` with an
            appropriate sign to disable bounds on all or some parameters.
            .. versionadded:: 0.17
        method : {'trf'}, optional
            Method to use for optimization. See `least_squares` for more details.
            Currently only 'trf' is implemented.
            .. versionadded:: 0.17
        solver : {'auto', 'svd', 'cg', 'lsqr', 'minibatch'}, optional
            Solver method for handling large datasets and different problem types:
            - 'auto' (default): Automatically selects the best solver based on problem size
            - 'svd': Uses SVD decomposition (good for small to medium datasets)
            - 'cg': Uses conjugate gradient method (memory efficient for large problems)
            - 'lsqr': Uses LSQR iterative solver (good for sparse problems)
            - 'minibatch': Processes data in batches (for very large datasets)
        batch_size : int, optional
            Batch size for minibatch solver. Only used when solver='minibatch'.
            If None and minibatch solver is selected, a reasonable default based
            on data size will be chosen.
        jac : callable, string or None, optional
            Function with signature ``jac(x, ...)`` which computes the Jacobian
            matrix of the model function with respect to parameters as a dense
            array_like structure. It will be scaled according to provided `sigma`.
            If None (default), the Jacobian will be determined using JAX's automatic
            differentiation (AD) capabilities. We recommend not using an analytical
            Jacobian, as it is usually faster to use AD.
        callback : callable or None, optional
            Callback function called after each optimization iteration with signature
            ``callback(iteration, cost, params, info)``. Useful for monitoring
            optimization progress, logging, or implementing custom stopping criteria.
            If None (default), no callback is invoked. See ``nlsq.callbacks`` module
            for built-in callbacks (ProgressBar, IterationLogger, EarlyStopping).
            .. versionadded:: 0.2.0
        kwargs
            Keyword arguments passed to `leastsq` for ``method='lm'`` or
            `least_squares` otherwise.

        Returns
        -------
        popt : array
            Optimal values for the parameters so that the sum of the squared
            residuals of ``f(xdata, *popt) - ydata`` is minimized.
        pcov : 2-D array
            The estimated covariance of popt. The diagonals provide the variance
            of the parameter estimate. To compute one standard deviation errors
            on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
            How the `sigma` parameter affects the estimated covariance
            depends on `absolute_sigma` argument, as described above.
            If the Jacobian matrix at the solution doesn't have a full rank, then
            'lm' method returns a matrix filled with ``np.inf``, on the other hand
            'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
            the covariance matrix.

        Raises
        ------
        ValueError
            if either `ydata` or `xdata` contain NaNs, or if incompatible options
            are used.
        RuntimeError
            if the least-squares minimization fails.
        OptimizeWarning
            if covariance of the parameters can not be estimated.
        See Also
        --------
        least_squares : Minimize the sum of squares of nonlinear functions.

        Notes
        -----
        Refer to the docstring of `least_squares` for more information.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import jax.numpy as jnp
        >>> from jaxfit import CurveFit
        >>> def func(x, a, b, c):
        ...     return a * jnp.exp(-b * x) + c
        Define the data to be fit with some noise:
        >>> xdata = np.linspace(0, 4, 50)
        >>> y = func(xdata, 2.5, 1.3, 0.5)
        >>> rng = np.random.default_rng()
        >>> y_noise = 0.2 * rng.normal(size=xdata.size)
        >>> ydata = y + y_noise
        >>> plt.plot(xdata, ydata, 'b-', label='data')
        Fit for the parameters a, b, c of the function `func`:
        >>> cf = CurveFit()
        >>> popt, _pcov = cf.curve_fit(func, xdata, ydata)
        >>> popt
        array([2.56274217, 1.37268521, 0.47427475])
        >>> plt.plot(xdata, func(xdata, *popt), 'r-',
        ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        Constrain the optimization to the region of ``0 <= a <= 3``,
        ``0 <= b <= 1`` and ``0 <= c <= 0.5``:
        >>> cf = CurveFit()
        >>> popt, _pcov = cf.curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
        >>> popt
        array([2.43736712, 1.        , 0.34463856])
        >>> plt.plot(xdata, func(xdata, *popt), 'g--',
        ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        >>> plt.xlabel('x')
        >>> plt.ylabel('y')
        >>> plt.legend()
        >>> plt.show()
        """

        # Prepare and validate all inputs
        (
            n,
            p0,
            xdata,
            ydata,
            data_mask,
            method,
            _lb,
            _ub,
            m,
            len_diff,
            _should_pad,
            none_mask,
        ) = self._prepare_curve_fit_inputs(
            f,
            xdata,
            ydata,
            p0,
            bounds,
            solver,
            batch_size,
            method,
            check_finite,
            data_mask,
            kwargs,
        )

        # Setup sigma transformation
        transform = self._setup_sigma_transform(sigma, ydata, data_mask, len_diff, m)

        # Run optimization
        res, jnp_xdata, ctime = self._run_optimization(
            f,
            p0,
            xdata,
            ydata,
            data_mask,
            transform,
            bounds,
            method,
            solver,
            batch_size,
            jac,
            m,
            n,
            sigma,
            timeit,
            callback,
            kwargs,
        )

        popt = res.x
        self.logger.debug(
            "Optimization succeeded",
            final_cost=res.cost,
            nfev=res.nfev,
            optimality=getattr(res, "optimality", None),
        )

        st = time.time()
        ysize = m
        cost = 2 * res.cost  # res.cost is half sum of squares!

        # Compute covariance matrix
        pcov, warn_cov = self._compute_covariance(res, ysize, p0, absolute_sigma)
        _pcov = pcov

        return_full = False

        # self.res = res
        post_time = time.time() - st

        # Log curve fit completion
        total_time = self.logger.timers.get("curve_fit", 0)
        self.logger.info(
            "Curve fit completed",
            total_time=total_time,
            final_cost=cost,
            covariance_warning=warn_cov,
        )

        if return_eval:
            feval = f(jnp_xdata, *popt)
            feval = np.array(feval)
            if none_mask:
                # data_mask = np.ndarray.astype(data_mask, bool)
                return popt, _pcov, feval[data_mask]
            else:
                return popt, _pcov, feval

        if return_full:
            raise RuntimeError("Return full only works for LM")
            # return popt, _pcov, infodict, errmsg, ier
        elif timeit:
            # lower GPU memory usage before returning raw res
            res.pop("jac", None)
            res.pop("fun", None)
            return popt, _pcov, res, post_time, ctime
        else:
            # Create enhanced result object that supports tuple unpacking
            # for backward compatibility: popt, pcov = curve_fit(...)
            # Keep 'fun' (residuals) and 'jac' for statistical computations
            result = CurveFitResult(res)
            result["model"] = f
            result["xdata"] = xdata
            result["ydata"] = ydata
            result["pcov"] = _pcov

            # Add cache statistics if available
            try:
                cache_stats = self.cache.get_stats()
                if cache_stats.get("enabled", True):
                    # Extract relevant cache metrics
                    result["cache_stats"] = {
                        "hit": cache_stats.get("hits", 0)
                        > cache_stats.get("misses", 0),
                        "compile_time_ms": cache_stats.get("compile_time_ms", 0.0),
                        "hit_rate": cache_stats.get("hit_rate", 0.0),
                        "total_hits": cache_stats.get("hits", 0),
                        "total_misses": cache_stats.get("misses", 0),
                        "total_compilations": cache_stats.get("compilations", 0),
                        "cache_size": cache_stats.get("cache_size", 0),
                    }
            except (AttributeError, KeyError) as e:
                # Cache not available or statistics disabled - log and continue
                self.logger.debug(f"Cache statistics not available: {e}")

            return result
