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
    - 'lm': accepted alias for unbounded problems; currently runs the same
      trust-region solver as 'trf' (no separate Levenberg-Marquardt/MINPACK path)
    - 'trf': Trust Region Reflective (bounded problems, recommended)
    - 'dogbox': not implemented (rejected as an unknown method)

Examples
--------
>>> from nlsq import CurveFit
>>> import jax.numpy as jnp
>>>
>>> def exponential(x, a, b): return a * jnp.exp(-b * x)
>>>
>>> # Class-based interface for reusing compilations
>>> fitter = CurveFit()
>>> popt1, pcov1 = fitter.curve_fit(exponential, x1, y1, p0=[2.0, 0.5])
>>> popt2, pcov2 = fitter.curve_fit(exponential, x2, y2, p0=[2.5, 0.6])

See Also
--------
nlsq.curve_fit : Function-based interface
nlsq.fit : Unified entry point with automatic workflow selection
nlsq.least_squares : Lower-level optimization interface
nlsq.trf : Trust Region Reflective implementation
"""

# mypy: disable-error-code="arg-type,return-value,assignment,attr-defined,index"
# Note: mypy errors are mostly arg-type/return-value mismatches where Optional values
# are passed to methods expecting non-Optional, return type mismatches (many methods
# have legacy tuple return types but actually return CurveFitResult), and dict indexing
# on result types that support both tuple unpacking and dict-like access.

from __future__ import annotations

import contextvars
import dataclasses
import logging
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from nlsq.core.workflow import OptimizationGoal
    from nlsq.global_optimization.cmaes_config import CMAESConfig
    from nlsq.global_optimization.config import GlobalOptimizationConfig
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.large_dataset import LDMemoryConfig

# Initialize JAX configuration through central config
from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax
import jax.numpy as jnp
from jax import jit
from jax.errors import TracerArrayConversionError
from jax.scipy.linalg import cholesky as jax_cholesky
from jax.scipy.linalg import svd as jax_svd

from nlsq.caching.memory_manager import get_memory_manager
from nlsq.caching.unified_cache import UnifiedCache, get_global_cache
from nlsq.constants import DEFAULT_FTOL, DEFAULT_XTOL
from nlsq.core import device_policy
from nlsq.core._option_checks import (
    _unwrap_scalar,
    check_not_nan,
    check_sigma_values,
    coerce_absolute_sigma,
    validate_f_scale,
    validate_max_nfev,
    validate_tr_solver,
)
from nlsq.core.covariance import _NLSQ_DIR, pcov_from_jacobian
from nlsq.core.least_squares import LeastSquares, prepare_bounds
from nlsq.core.problem import (
    PrepareContext,
    _initialize_feasible,
    _is_auto_p0,
    _n_params_from_signature,
    coerce_real_sigma,
    finite_rows,
    prepare_problem,
    reject_all_zero_weight,
    reject_complex_inputs,
    rows_used_by_fit,
    zero_weight_rows,
)
from nlsq.core.problem import pad_fit_data as _pad_fit_data

# Diagnostics types are needed at module level for function signatures
from nlsq.diagnostics.types import DiagnosticLevel, DiagnosticsConfig
from nlsq.result import CurveFitResult, OptimizeWarning
from nlsq.types import ArrayLike, ModelFunction
from nlsq.utils.error_messages import OptimizationError
from nlsq.utils.logging import get_logger
from nlsq.utils.validators import InputValidator

__all__ = ["CurveFit", "curve_fit", "fit"]

# enable_overflow_check clips |f| to this bound; values inside it are untouched.
_OVERFLOW_CLIP_BOUND = 1e10

# Logger for workflow selection messages
_logger = get_logger(__name__)

# =============================================================================
# NEW WORKFLOW SYSTEM (v0.6.3)
# =============================================================================
#
# Valid workflows: "auto", "auto_global", "hpc"
# - "auto": Memory-aware local optimization (default)
# - "auto_global": Memory-aware global optimization (requires bounds)
# - "hpc": auto_global + checkpointing for HPC environments
#
# The following presets were removed in v0.6.3 and will raise ValueError:

REMOVED_PRESETS: dict[str, str] = {
    "standard": (
        "Use workflow='auto' instead. "
        "The system automatically selects the optimal memory strategy."
    ),
    "fast": (
        "Use workflow='auto' with gtol=1e-6, ftol=1e-6, xtol=1e-6 instead:\n"
        "    fit(model, x, y, workflow='auto', gtol=1e-6, ftol=1e-6, xtol=1e-6)"
    ),
    "quality": (
        "Use workflow='auto_global' with bounds and n_starts=20 instead:\n"
        "    fit(model, x, y, workflow='auto_global', bounds=bounds, n_starts=20)"
    ),
    "large_robust": (
        "Use workflow='auto' instead. "
        "The system automatically detects large datasets and applies chunking."
    ),
    "streaming": (
        "Use workflow='auto' instead. "
        "The system automatically detects memory pressure and enables streaming."
    ),
    "hpc_distributed": (
        "Use workflow='hpc' with bounds instead:\n"
        "    fit(model, x, y, workflow='hpc', bounds=bounds, checkpoint_dir='/path/to/checkpoints')"
    ),
    "cmaes": (
        "Use workflow='auto_global' with bounds instead. "
        "CMA-ES is auto-selected when scale_ratio > 1000:\n"
        "    fit(model, x, y, workflow='auto_global', bounds=bounds)"
    ),
    "cmaes-global": (
        "Use workflow='auto_global' with extended CMA-ES config:\n"
        "    from nlsq.global_optimization import CMAESConfig\n"
        "    fit(model, x, y, workflow='auto_global', bounds=bounds, "
        "cmaes_config=CMAESConfig(n_generations=200))"
    ),
    "global_auto": (
        "Use workflow='auto_global' instead:\n"
        "    fit(model, x, y, workflow='auto_global', bounds=bounds)"
    ),
}

# Documentation URL for migration guide
_MIGRATION_DOCS_URL = "https://nlsq.readthedocs.io/en/latest/howto/migration.html"


def _raise_removed_preset_error(preset_name: str) -> None:
    """Raise ValueError for removed workflow presets with migration guidance.

    Parameters
    ----------
    preset_name : str
        The name of the removed preset.

    Raises
    ------
    ValueError
        Always raised with migration instructions.
    """
    migration_hint = REMOVED_PRESETS.get(preset_name, "Use workflow='auto' instead.")
    raise ValueError(
        f"Workflow preset '{preset_name}' was removed in v0.6.3.\n\n"
        f"{migration_hint}\n\n"
        f"See migration guide: {_MIGRATION_DOCS_URL}",
    )


def _parse_goal_parameter(goal: str | Any | None) -> Any:
    """Parse goal parameter into OptimizationGoal enum.

    Parameters
    ----------
    goal : str, OptimizationGoal, or None
        The goal specification to parse.

    Returns
    -------
    OptimizationGoal or None
        The parsed goal enum, or None if goal was None.

    Raises
    ------
    ValueError
        If goal string is not recognized.
    """
    from nlsq.core.workflow import OptimizationGoal

    if goal is None:
        return None

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
                f"Unknown goal '{goal}'. Must be one of: {list(goal_map.keys())}",
            )
        return goal_map[goal_lower]

    if isinstance(goal, OptimizationGoal):
        return goal

    # Handle enum identity issue in parallel test execution (pytest-xdist)
    # where the same enum type may be loaded from different module imports
    if type(goal).__name__ == "OptimizationGoal":
        return OptimizationGoal[goal.name]

    raise ValueError(
        f"goal must be a string or OptimizationGoal enum, got {type(goal)}",
    )


def _infer_n_features(xdata: np.ndarray) -> int:
    """Count features in the public SciPy layout ``(n_features, n_points)``.

    Workflow inputs use this layout before any streaming adapter transposes
    them to points-first. Coordinate tuples have already become 2-D arrays.
    """
    return xdata.shape[0] if xdata.ndim == 2 else 1


def fit(  # noqa: C901
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    check_finite: bool = True,
    bounds: tuple = (-float("inf"), float("inf")),
    method: str | None = None,
    workflow: (
        str | LDMemoryConfig | HybridStreamingConfig | GlobalOptimizationConfig
    ) = "auto",
    goal: str | OptimizationGoal | None = None,
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
        Optimization algorithm ('trf', 'lm', or None for auto). ``'lm'`` is
        currently an alias that runs the same trust-region solver as ``'trf'``.
    workflow : str or config object, optional
        Workflow selection:

        - 'auto': Memory-aware local optimization; auto-selects standard,
          chunked, or streaming based on dataset size and memory (default)
        - 'auto_global': Memory-aware global optimization (requires bounds);
          auto-selects CMA-ES vs multi-start based on parameter scale ratio
        - 'hpc': 'auto_global' plus cluster detection for HPC environments
        - Custom config object: LDMemoryConfig, HybridStreamingConfig, or GlobalOptimizationConfig

        The named presets from pre-v0.6.3 ('standard', 'quality', 'fast',
        'large_robust', 'streaming', 'hpc_distributed', 'cmaes',
        'cmaes-global', 'global_auto') were removed in v0.6.3 and raise
        ValueError with migration guidance -- use 'auto'/'auto_global'/'hpc'
        with explicit ``goal``/``bounds``/tolerance kwargs instead.
    goal : str or OptimizationGoal, optional
        Optimization goal that modifies tolerances:

        - 'fast': Prioritize speed, use looser tolerances
        - 'robust' or 'global': Standard tolerances
        - 'memory_efficient': Minimize memory usage
        - 'quality': Highest precision, tighter tolerances

        ``goal`` only sets tolerances; pass ``multistart=True`` (and
        ``n_starts``) to enable multi-start.

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

    >>> result = fit(model, xdata, ydata, p0=[1, 2], workflow='auto_global',
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
    MemoryBudgetSelector : Memory-based automatic strategy selection
    """
    # Import workflow module components
    from nlsq.core.workflow import MemoryBudgetSelector

    # Convert data to arrays for size calculations
    xdata_arr = np.asarray(xdata)
    ydata_arr = np.asarray(ydata)
    n_points = len(ydata_arr)
    n_features = _infer_n_features(xdata_arr)

    # Handle empty data
    if n_points == 0:
        raise ValueError("`ydata` must not be empty!")

    # Resolve the sentinel before chunked/CMA-ES adapters convert numeric
    # guesses to arrays: a 0-D string array is not a parameter vector.
    if _is_auto_p0(p0):
        p0 = _resolve_auto_p0(f, xdata_arr, ydata_arr, sigma, kwargs)

    # Determine number of parameters (see nlsq.core.problem._determine_parameter_count/_is_auto_p0).
    if p0 is not None and not _is_auto_p0(p0):
        p0_arr = np.atleast_1d(p0)
        n_params = len(p0_arr)
    else:
        # Try to auto-estimate p0 to determine n_params
        # This handles functions with *args signatures (e.g., polynomials) correctly
        estimated_success = False
        if hasattr(f, "estimate_p0"):
            try:
                # Use estimator to get p0 and count
                # We reuse this p0 to avoid re-estimation overhead later
                estimated_p0_val = f.estimate_p0(
                    *rows_used_by_fit(
                        xdata_arr,
                        ydata_arr,
                        sigma,
                        kwargs.get("data_mask"),
                        kwargs.get("nan_policy"),
                    )
                )
                # Convert to list/array to check length safely
                p0_temp = np.atleast_1d(estimated_p0_val)
                n_params = len(p0_temp)
                p0 = estimated_p0_val
                estimated_success = True
            except Exception as e:
                # Log debug warning but continue to fallback.
                # Note: We don't want to crash here if estimation fails, just fallback.
                if _logger.logger.isEnabledFor(logging.DEBUG):
                    _logger.debug(f"Automatic p0 estimation failed: {e}")

        if not estimated_success:
            # Fallback to inference from function signature
            n_params = _n_params_from_signature(f)
            if n_params < 1:
                raise ValueError("Unable to determine number of fit parameters.")

    # Convert goal string to OptimizationGoal enum if needed
    goal_enum = _parse_goal_parameter(goal)

    # Process workflow parameter
    # Import config classes for isinstance checks
    from nlsq.global_optimization import GlobalOptimizationConfig
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.large_dataset import LDMemoryConfig

    if isinstance(
        workflow,
        (LDMemoryConfig, HybridStreamingConfig, GlobalOptimizationConfig),
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

        # Check if this is a removed preset first
        if workflow_lower in REMOVED_PRESETS:
            _raise_removed_preset_error(workflow_lower)

        if workflow_lower == "auto":
            # Auto-select workflow based on memory budget
            _logger.info(
                f"workflow='auto' starting: n_points={n_points:,}, n_params={n_params}",
            )

            selector = MemoryBudgetSelector()
            memory_limit_gb = kwargs.pop("memory_limit_gb", None)

            # Compute memory budget for logging
            from nlsq.core.workflow import MemoryBudget

            bounded = _is_bounded(bounds, n_params)
            budget = MemoryBudget.compute(
                n_points=n_points,
                n_params=n_params,
                n_features=n_features,
                memory_limit_gb=memory_limit_gb,
                bounded=bounded,
            )
            _logger.info(
                f"workflow='auto' memory budget: "
                f"available={budget.available_gb:.1f}GB, "
                f"threshold={budget.threshold_gb:.1f}GB, "
                f"peak_estimate={budget.peak_gb:.2f}GB",
            )

            _strategy, config = selector.select(
                n_points=n_points,
                n_params=n_params,
                n_features=n_features,
                memory_limit_gb=memory_limit_gb,
                goal=goal_enum,
                budget=budget,
                bounded=bounded,
            )

            # Log strategy selection with reasoning. The chunked/standard
            # boundary applies a 10% safety margin (budget.chunked_threshold_gb)
            # -- log against that, not the raw threshold, so this message
            # can't disagree with what selector.select() actually decided.
            if _strategy == "streaming":
                _logger.info(
                    f"workflow='auto' selected: STREAMING "
                    f"(data {budget.data_gb:.2f}GB > threshold {budget.threshold_gb:.1f}GB)",
                )
            elif _strategy == "chunked":
                _logger.info(
                    f"workflow='auto' selected: CHUNKED "
                    f"(peak {budget.peak_gb:.2f}GB > safety threshold "
                    f"{budget.chunked_threshold_gb:.1f}GB)",
                )
            else:
                _logger.info(
                    f"workflow='auto' selected: STANDARD "
                    f"(peak {budget.peak_gb:.2f}GB fits in "
                    f"{budget.chunked_threshold_gb:.1f}GB)",
                )

            # Log bounds info if provided
            if bounds is not None:
                lb, ub = prepare_bounds(bounds, n_params)
                has_bounds = not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))
                _logger.debug(
                    f"workflow='auto' bounds: {'provided' if has_bounds else 'unbounded'}",
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

        if workflow_lower == "auto_global":
            # Memory-aware global optimization (requires bounds)
            return _fit_with_auto_global(
                f=f,
                xdata=xdata_arr,
                ydata=ydata_arr,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                n_points=n_points,
                n_params=n_params,
                goal=goal_enum,
                method=method,
                **kwargs,
            )

        if workflow_lower == "hpc":
            # HPC workflow with checkpointing (wraps auto_global)
            return _fit_with_hpc(
                f=f,
                xdata=xdata_arr,
                ydata=ydata_arr,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                n_points=n_points,
                n_params=n_params,
                goal=goal_enum,
                method=method,
                **kwargs,
            )

        accepted = ["auto", "auto_global", "hpc"]
        raise ValueError(
            f"Unknown workflow '{workflow}'. Must be one of "
            f"{accepted}, or a config object.",
        )

    raise ValueError(
        f"workflow must be a string or config object, got {type(workflow)}",
    )


def _check_finite_for_streaming(
    xdata: np.ndarray,
    ydata: np.ndarray,
    check_finite: bool,
) -> None:
    """Validate xdata/ydata for NaN/Inf before an AdaptiveHybridStreamingOptimizer
    call. That optimizer has no check_finite parameter and its 4-phase
    pipeline never validates raw input itself (unlike LargeDatasetFitter/
    curve_fit(), which do) -- without this, a NaN/Inf silently runs to a
    bogus success=True result (parameters stuck at p0, NaN std errors)
    instead of the documented check_finite=True behavior. Shared by every
    reachable caller that constructs that optimizer directly, so a new
    caller can't reintroduce the gap by copy-pasting the construction
    without the check.
    """
    if not check_finite:
        return
    if not np.all(np.isfinite(xdata)):
        n_bad = int(np.sum(~np.isfinite(xdata)))
        raise ValueError(f"xdata contains {n_bad} NaN or Inf values")
    if not np.all(np.isfinite(ydata)):
        n_bad = int(np.sum(~np.isfinite(ydata)))
        raise ValueError(f"ydata contains {n_bad} NaN or Inf values")


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
    from nlsq.streaming.large_dataset import LDMemoryConfig
    from nlsq.streaming.strategy import (
        build_large_strategy,
        default_p0,
        scipy_layout_to_points_first,
    )

    reject_all_zero_weight(sigma, kwargs.get("data_mask"))
    n_points = len(ydata)
    user_kwargs = set(kwargs)  # before goal-derived tolerances are injected

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
            p0=np.asarray(p0) if p0 is not None and not _is_auto_p0(p0) else None,
            bounds=bounds,
            sigma=np.asarray(sigma) if sigma is not None else None,
            absolute_sigma=absolute_sigma,
            method=method,
            check_finite=check_finite,
            **kwargs,
        )
        return result

    if isinstance(config, HybridStreamingConfig):
        # Streaming optimization path
        _check_finite_for_streaming(xdata, ydata, check_finite)

        # The hybrid engine minimizes plain least squares; only callback,
        # verbose and gtol are read from kwargs. Fail loudly instead of
        # silently returning a non-robust fit.
        _reject_options_ignored_by_hybrid(
            kwargs,
            user_kwargs & {"f_scale", "max_nfev", "ftol", "xtol", "x_scale", "jac"},
        )

        # Like method="hybrid_streaming": the engine always computes pcov and
        # returns a CurveFitResult, so these cannot be honoured; fail loudly.
        if kwargs.get("compute_covariance", True) is False:
            raise ValueError(
                "compute_covariance=False is not supported by the hybrid "
                "streaming workflow (the engine always computes the covariance)."
            )
        if kwargs.get("full_output"):
            raise ValueError(
                "full_output=True is not supported by the hybrid streaming "
                "workflow; fit() returns a CurveFitResult."
            )

        # Prepare p0
        if p0 is None or _is_auto_p0(p0):
            p0 = default_p0(f)
        p0_arr = np.atleast_1d(p0)

        # Prepare bounds
        lb, ub = prepare_bounds(bounds, len(p0_arr))
        bounds_tuple = (
            (lb, ub)
            if not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))
            else None
        )

        # The goal-derived (or user-supplied) gtol computed into kwargs above
        # maps onto config.gauss_newton_tol (on a copy) inside the hybrid
        # strategy; ftol/xtol have no streaming equivalent to map to. gtol
        # None/<=0 ("test disabled") has no positive-tolerance equivalent: keep
        # the config's own tolerance.
        gtol = kwargs.get("gtol")
        large_strategy = build_large_strategy(
            "hybrid",
            config,
            **({"gtol": gtol} if gtol is not None and gtol > 0 else {}),
        )
        f_engine, x_engine = scipy_layout_to_points_first(f, xdata)
        result_dict = large_strategy.fit(
            f_engine,
            x_engine,
            ydata,
            p0_arr,
            bounds=bounds_tuple,
            sigma=np.asarray(sigma) if sigma is not None else None,
            absolute_sigma=absolute_sigma,
            callback=kwargs.get("callback"),
            verbose=kwargs.get("verbose", 0),
        )

        # Convert to CurveFitResult
        result = CurveFitResult(result_dict)
        result["model"] = f
        result["xdata"] = xdata
        result["ydata"] = ydata
        result["pcov"] = result_dict.get(
            "pcov",
            np.full((len(p0_arr), len(p0_arr)), np.inf),
        )
        _store_sigma(result, sigma, absolute_sigma)
        return result

    if isinstance(config, LDMemoryConfig):
        # An explicit config owns the budget. Public fit() also forwards its
        # routing-only memory_limit_gb default; keep it out of the chunk solver.
        kwargs.pop("memory_limit_gb", None)
        # Chunked processing path. MemoryBudgetSelector already decided
        # chunking is needed because peak memory (data + n_params-scaled
        # Jacobian) exceeds the safe threshold -- that can happen at any
        # n_points when n_params is large, so don't second-guess it by
        # n_points alone here (that used to silently fall back to plain
        # curve_fit() and defeat the memory-based decision).
        large_strategy = build_large_strategy(
            "chunked", config, memory_limit_gb=config.memory_limit_gb
        )
        result = large_strategy.fit(
            f,
            xdata,
            ydata,
            np.asarray(p0) if p0 is not None else None,
            bounds=bounds,
            method=method or "trf",
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            **kwargs,
        )

        # Ensure we have a CurveFitResult
        x_fit, y_fit, s_fit = _rows_actually_fit(xdata, ydata, sigma, kwargs)
        if not isinstance(result, CurveFitResult):
            # Convert from dict or OptimizeResult
            result = CurveFitResult(result)
            result["model"] = f
            result["xdata"] = x_fit
            result["ydata"] = y_fit
        _store_sigma(result, s_fit, absolute_sigma)

        return result

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


# =============================================================================
# NEW WORKFLOW HANDLERS (v0.6.3)
# =============================================================================


def _fit_with_auto_global(
    f: ModelFunction,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: ArrayLike | None,
    sigma: ArrayLike | None,
    absolute_sigma: bool,
    check_finite: bool,
    bounds: tuple,
    n_points: int,
    n_params: int,
    goal: Any,
    method: str | None = None,
    _workflow_name: str = "auto_global",
    **kwargs: Any,
) -> CurveFitResult:
    """Memory-aware global optimization workflow (T026-T035).

    This function implements the auto_global workflow which:
    1. Validates that bounds are provided (required for global optimization)
    2. Uses MemoryBudgetSelector to select memory strategy (standard/chunked/streaming)
    3. Uses MethodSelector to select global method (CMA-ES or Multi-Start)
    4. Routes to appropriate backend based on the 6 strategy×method combinations

    Parameters
    ----------
    f : callable
        Model function f(x, *params) -> y.
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    p0 : ArrayLike | None
        Initial parameter guess.
    sigma : ArrayLike | None
        Uncertainties in ydata.
    absolute_sigma : bool
        Whether sigma is absolute.
    check_finite : bool
        Check for finite input values.
    bounds : tuple
        Parameter bounds (lower, upper). Required.
    n_points : int
        Number of data points.
    n_params : int
        Number of parameters.
    goal : OptimizationGoal | None
        Optimization goal.
    method : str or None, optional
        Requested global method, passed to ``MethodSelector.select`` as
        ``requested_method`` (``None`` is treated as ``"auto"``).
    _workflow_name : str, default="auto_global"
        Workflow name used in error messages and forwarded as ``workflow_name``
        to the multi-start route. ``_fit_with_hpc`` passes ``"hpc"``.
    **kwargs : Any
        Additional arguments including:
        - n_starts: int (default 10) - Number of multi-start runs
        - cmaes_config: CMAESConfig | None - Override CMA-ES configuration
        - memory_limit_gb: float | None - Override memory detection

    Returns
    -------
    CurveFitResult
        Optimization result.

    Raises
    ------
    ValueError
        If bounds are not provided.
    """
    from nlsq.core.workflow import (
        MemoryBudget,
        MemoryBudgetSelector,
        calculate_adaptive_tolerances,
    )
    from nlsq.global_optimization.method_selector import MethodSelector

    # Log workflow initialization
    _logger.info(
        f"workflow='auto_global' starting: n_points={n_points:,}, n_params={n_params}",
    )

    # FR-002: Validate bounds are provided
    lb, ub = prepare_bounds(bounds, n_params)
    if np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)):
        raise ValueError(
            f"workflow='{_workflow_name}' requires bounds. "
            "Provide bounds as (lower, upper) arrays:\n"
            f"    fit(model, x, y, workflow='{_workflow_name}', bounds=([0, 0], [10, 10]))",
        )
    # Partially-unbounded (e.g. one parameter's upper bound left at +inf)
    # previously slipped past the fully-unbounded check above, then fed an
    # infinite range into the CMA-ES sigmoid bounds transform
    # (lb + (ub-lb)*sigmoid(x)): inf * (a sufficiently negative x's sigmoid,
    # which underflows to exactly 0.0) is nan. Require every bound finite.
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise ValueError(
            f"workflow='{_workflow_name}' requires every parameter to have a "
            "finite lower AND upper bound (no +/-inf). "
            f"Got lower={lb}, upper={ub}.",
        )
    if np.any(lb > ub):
        raise ValueError(
            f"workflow='{_workflow_name}' bounds are inverted: lower={lb} must "
            f"be <= upper={ub} for every parameter.",
        )
    # Individually-finite bounds can still overflow when subtracted (e.g.
    # lb=-1e308, ub=1e308): ub-lb -> inf reproduces the exact NaN failure
    # mode the finite-bounds check above exists to prevent, since
    # transform_to_bounds computes lb + (ub-lb)*sigmoid(x). Range, not just
    # the endpoints, must be finite.
    if not np.all(np.isfinite(ub - lb)):
        raise ValueError(
            f"workflow='{_workflow_name}' requires each parameter's bound range "
            "(upper - lower) to be finite -- the bounds are individually "
            f"finite but too far apart to subtract without overflow. Got "
            f"lower={lb}, upper={ub}.",
        )

    # Log bounds information
    _logger.debug(f"workflow='auto_global' bounds: lower={lb}, upper={ub}")

    # Extract optional parameters
    # nlsq.fit forwards n_starts=None when the caller gave none: None means default.
    n_starts = kwargs.pop("n_starts", None)
    n_starts = 10 if n_starts is None else n_starts
    cmaes_config = kwargs.pop("cmaes_config", None)
    memory_limit_gb = kwargs.pop("memory_limit_gb", None)

    # Checkpoint/resume is implemented only for the CMA-ES route
    # (restart_strategy="none") -- see
    # docs/superpowers/plans/2026-08-27-cmaes-checkpoint-resume-spec.md.
    # This extraction lives here (not only in _fit_with_hpc) so checkpoint_dir
    # works identically regardless of entry point: workflow='hpc' forwards its
    # kwargs through unconverted and hits this same logic, and a direct
    # workflow='auto_global' call (which never passes through _fit_with_hpc)
    # gets the same capability instead of checkpoint_dir silently riding
    # through as an unrecognized kwarg. Whether the CMA-ES route is even
    # selected depends on MethodSelector's choice below, so thread the
    # checkpoint kwargs through unconditionally and let _fit_global_cmaes wire
    # them into CMAESConfig; every OTHER route (multistart/chunked/streaming)
    # still needs the warning, which _fit_global_multistart already emits.
    checkpoint_dir = kwargs.pop("checkpoint_dir", None)
    checkpoint_interval = kwargs.pop("checkpoint_interval", None)
    run_id = kwargs.pop("run_id", None)
    model_id = kwargs.pop("model_id", None)
    if checkpoint_dir is not None:
        kwargs["_hpc_checkpoint_dir"] = checkpoint_dir
        kwargs["_hpc_checkpoint_interval"] = checkpoint_interval
        kwargs["_hpc_run_id"] = run_id
        kwargs["_hpc_model_id"] = model_id
    elif checkpoint_interval is not None or run_id is not None or model_id is not None:
        # These only make sense alongside checkpoint_dir -- e.g. a caller
        # passing run_id='job-042' to override a run_id already baked into
        # a pre-built cmaes_config, expecting it to parameterize that
        # config's checkpoint file. Without checkpoint_dir at the top
        # level there is no marker for _fit_global_cmaes to apply these
        # against, so they would otherwise be silently discarded here.
        warnings.warn(
            "checkpoint_interval/run_id/model_id were provided without "
            "checkpoint_dir -- they have no effect. If checkpoint_dir is "
            "set on a pre-built cmaes_config, pass these overrides on "
            "that CMAESConfig directly instead of as fit() kwargs.",
            UserWarning,
            stacklevel=2,
        )

    # FR-000: Select memory strategy with detailed logging
    selector = MemoryBudgetSelector()

    n_features = _infer_n_features(xdata)

    # Compute memory budget for logging
    # bounds are finite on every parameter here (validated above)
    budget = MemoryBudget.compute(
        n_points=n_points,
        n_params=n_params,
        n_features=n_features,
        memory_limit_gb=memory_limit_gb,
        bounded=True,
    )
    _logger.info(
        f"workflow='auto_global' memory budget: "
        f"available={budget.available_gb:.1f}GB, "
        f"threshold={budget.threshold_gb:.1f}GB, "
        f"peak_estimate={budget.peak_gb:.2f}GB",
    )

    strategy, memory_config = selector.select(
        n_points=n_points,
        n_params=n_params,
        n_features=n_features,
        memory_limit_gb=memory_limit_gb,
        goal=goal,
        budget=budget,
        bounded=True,
    )

    # Log strategy selection with reasoning. The chunked/standard boundary
    # applies a 10% safety margin (budget.chunked_threshold_gb) -- log
    # against that, not the raw threshold, so this can't disagree with
    # what selector.select() actually decided.
    if strategy == "streaming":
        _logger.info(
            f"workflow='auto_global' memory strategy: STREAMING "
            f"(data {budget.data_gb:.2f}GB > threshold {budget.threshold_gb:.1f}GB)",
        )
    elif strategy == "chunked":
        _logger.info(
            f"workflow='auto_global' memory strategy: CHUNKED "
            f"(peak {budget.peak_gb:.2f}GB > safety threshold "
            f"{budget.chunked_threshold_gb:.1f}GB)",
        )
    else:
        _logger.info(
            f"workflow='auto_global' memory strategy: STANDARD "
            f"(peak {budget.peak_gb:.2f}GB fits in "
            f"{budget.chunked_threshold_gb:.1f}GB)",
        )

    # FR-005: Select global method (CMA-ES vs Multi-Start)
    method_selector = MethodSelector()
    # Compute via the same helper .select() uses internally, so this
    # logged ratio can never diverge from the one that actually drove
    # the method decision (a hand-rolled re-derivation here previously
    # collapsed to 1.0 for any zero/negative-width bound instead of
    # filtering it out the way compute_scale_ratio does).
    scale_ratio = method_selector.compute_scale_ratio(lb, ub)
    global_method = method_selector.select(
        requested_method=method or "auto",
        lower_bounds=lb,
        upper_bounds=ub,
    )

    # Log global method selection with reasoning
    if global_method == "cmaes":
        _logger.info(
            f"workflow='auto_global' global method: CMA-ES "
            f"(scale_ratio={scale_ratio:.1f} > 1000 or evosax available)",
        )
    else:
        _logger.info(
            f"workflow='auto_global' global method: Multi-Start "
            f"(n_starts={n_starts}, scale_ratio={scale_ratio:.1f})",
        )

    # Apply adaptive tolerances if goal is specified
    user_kwargs = frozenset(kwargs)  # before goal-derived tolerances are injected
    if goal is not None:
        adaptive_tols = calculate_adaptive_tolerances(n_points, goal)
        for tol_key in ["gtol", "ftol", "xtol"]:
            if tol_key not in kwargs:
                kwargs[tol_key] = adaptive_tols[tol_key]

    # Route to appropriate backend based on strategy x method combination.
    # memory_config carries the budget MemoryBudgetSelector actually computed
    # (real detected memory, or the caller's memory_limit_gb override) so the
    # chunked/streaming backends don't have to re-derive (and potentially
    # mis-derive) their own sizing from n_points alone.
    if global_method == "cmaes":
        return _fit_global_cmaes(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            bounds=(lb, ub),
            strategy=strategy,
            memory_config=memory_config,
            cmaes_config=cmaes_config,
            check_finite=check_finite,
            **kwargs,
        )
    # Multi-start path
    return _fit_global_multistart(
        f=f,
        xdata=xdata,
        ydata=ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        check_finite=check_finite,
        bounds=(lb, ub),
        strategy=strategy,
        memory_config=memory_config,
        n_starts=n_starts,
        workflow_name=_workflow_name,
        user_kwargs=user_kwargs,
        **kwargs,
    )


def _fit_with_hpc(
    f: ModelFunction,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: ArrayLike | None,
    sigma: ArrayLike | None,
    absolute_sigma: bool,
    check_finite: bool,
    bounds: tuple,
    n_points: int,
    n_params: int,
    goal: Any,
    method: str | None = None,
    **kwargs: Any,
) -> CurveFitResult:
    """HPC workflow with automatic checkpointing (T042-T048).

    This function wraps auto_global with automatic checkpointing for
    crash recovery in HPC environments.

    Parameters
    ----------
    f : callable
        Model function.
    xdata, ydata : np.ndarray
        Data arrays.
    p0 : ArrayLike | None
        Initial parameters.
    sigma : ArrayLike | None
        Uncertainties.
    absolute_sigma : bool
        Whether sigma is absolute.
    check_finite : bool
        Check for finite input values.
    bounds : tuple
        Parameter bounds (required).
    n_points, n_params : int
        Data dimensions.
    goal : OptimizationGoal | None
        Optimization goal.
    method : str or None, optional
        Requested global method, forwarded to ``_fit_with_auto_global``.
    **kwargs : Any
        Additional arguments including:
        - checkpoint_dir: str | None - Checkpoint directory
        - checkpoint_interval: int - Save every N generations/starts

    Returns
    -------
    CurveFitResult
        Optimization result.

    Raises
    ------
    ValueError
        If bounds are not provided.
    """
    # Log workflow initialization
    _logger.info(f"workflow='hpc' starting: n_points={n_points:,}, n_params={n_params}")

    # FR-002: Validate bounds are provided
    lb, ub = prepare_bounds(bounds, n_params)
    if np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)):
        raise ValueError(
            "workflow='hpc' requires bounds. "
            "Provide bounds as (lower, upper) arrays:\n"
            "    fit(model, x, y, workflow='hpc', bounds=([0, 0], [10, 10]))",
        )

    # checkpoint_dir/checkpoint_interval/run_id/model_id are intentionally
    # NOT extracted here: they're forwarded unconverted in **kwargs to
    # _fit_with_auto_global below, which does the checkpoint_dir ->
    # _hpc_checkpoint_dir marker conversion itself -- this way the
    # conversion happens identically whether the caller reaches CMA-ES
    # through workflow='hpc' (here) or workflow='auto_global' directly.

    # Detect cluster environment
    from nlsq.core.workflow import ClusterDetector

    detector = ClusterDetector()
    cluster_info = detector.detect()

    if cluster_info is not None:
        _logger.info(
            f"workflow='hpc' cluster detected: "
            f"scheduler={cluster_info.scheduler}, "
            f"nodes={cluster_info.node_count}, "
            f"gpus={cluster_info.total_gpus}",
        )
        if cluster_info.job_id:
            _logger.debug(f"workflow='hpc' job_id={cluster_info.job_id}")
    else:
        _logger.info("workflow='hpc' running on local machine (no cluster detected)")

    # Delegates to auto_global, which does the checkpoint_dir ->
    # _hpc_checkpoint_dir marker conversion for the CMA-ES route (see the
    # comment on that conversion in _fit_with_auto_global).
    return _fit_with_auto_global(
        f=f,
        xdata=xdata,
        ydata=ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        check_finite=check_finite,
        bounds=bounds,
        n_points=n_points,
        n_params=n_params,
        goal=goal,
        method=method,
        _workflow_name="hpc",
        **kwargs,
    )


def _fit_global_cmaes(
    f: ModelFunction,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: ArrayLike | None,
    sigma: ArrayLike | None,
    absolute_sigma: bool,
    bounds: tuple[np.ndarray, np.ndarray],
    strategy: str,
    cmaes_config: Any | None,
    memory_config: HybridStreamingConfig | LDMemoryConfig | None = None,
    check_finite: bool = True,
    **kwargs: Any,
) -> CurveFitResult:
    """Run CMA-ES global optimization with memory-aware strategy.

    Parameters
    ----------
    f : callable
        Model function f(x, *params) -> y.
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    p0 : ArrayLike | None
        Initial parameter guess.
    sigma : ArrayLike | None
        Uncertainties in ydata.
    absolute_sigma : bool
        Whether sigma is absolute; forwarded to the NLSQ refinement fit.
    bounds : tuple[np.ndarray, np.ndarray]
        Parameter bounds (lower, upper).
    strategy : str
        Memory strategy: 'standard', 'chunked', or 'streaming'.
    memory_config : HybridStreamingConfig | LDMemoryConfig | None
        The config MemoryBudgetSelector actually computed from the real
        (detected or overridden) memory budget. Preferred over re-deriving
        a chunk size from n_points alone when available.
    cmaes_config : CMAESConfig | None
        CMA-ES configuration override.
    check_finite : bool, default=True
        Check for finite input values.
    **kwargs : Any
        Optional ``seed``, ``data_mask`` and the internal ``_hpc_checkpoint_dir`` /
        ``_hpc_checkpoint_interval`` / ``_hpc_run_id`` / ``_hpc_model_id`` markers
        are consumed here; ``nan_policy`` and ``compute_covariance`` are read here
        and also forwarded. The
        multi-start sampler options ``sampler``, ``center_on_p0`` and
        ``scale_factor`` are dropped with a ``UserWarning``. The remaining
        keywords are forwarded to ``CMAESOptimizer.fit``, except ``timeit``,
        ``return_eval`` and ``full_output``, which are applied to the result.
    """
    import dataclasses

    from nlsq.global_optimization.cmaes_config import CMAESConfig
    from nlsq.global_optimization.cmaes_optimizer import (
        CMAESOptimizer,
        _apply_data_mask,
    )

    # Create or use provided config
    if cmaes_config is None:
        cmaes_config = CMAESConfig()

    checkpoint_dir = kwargs.pop("_hpc_checkpoint_dir", None)
    checkpoint_interval = kwargs.pop("_hpc_checkpoint_interval", None)
    run_id = kwargs.pop("_hpc_run_id", None)
    model_id = kwargs.pop("_hpc_model_id", None)
    seed = kwargs.pop("seed", None)

    # fit(workflow="auto_global") forwards the multistart sampler options; only
    # the multi-start route uses them, so say so here instead of dropping them.
    unused = sorted(
        k for k in ("sampler", "center_on_p0", "scale_factor") if k in kwargs
    )
    for k in unused:
        kwargs.pop(k)
    if unused:
        warnings.warn(
            f"{unused} are not used on the CMA-ES route selected by "
            "workflow='auto_global' and are ignored; they apply only to its "
            "multi-start route, or use curve_fit()/curve_fit_large().",
            UserWarning,
            stacklevel=2,
        )

    # seed applies whenever provided, independent of checkpoint_dir -- it's
    # the general CMA-ES reproducibility knob, not a checkpoint-only concern.
    if seed is not None:
        cmaes_config = dataclasses.replace(cmaes_config, seed=seed)

    if checkpoint_dir is not None:
        cmaes_config = dataclasses.replace(
            cmaes_config,
            checkpoint_dir=checkpoint_dir,
            # No literal default here: fall back to CMAESConfig's own
            # checkpoint_interval default (10) instead of silently using a
            # different, undocumented one when the caller doesn't specify it.
            checkpoint_interval=(
                checkpoint_interval
                if checkpoint_interval is not None
                else cmaes_config.checkpoint_interval
            ),
            # Same None-preserves-existing-value treatment as
            # checkpoint_interval above -- don't wipe a pre-built config's
            # own run_id/model_id when the caller overrides checkpoint_dir
            # alone.
            run_id=run_id if run_id is not None else cmaes_config.run_id,
            model_id=(model_id if model_id is not None else cmaes_config.model_id),
        )

    # Budget-derived chunk size (HybridStreamingConfig.chunk_size for
    # streaming, LDMemoryConfig.streaming_batch_size for chunked).
    # MemoryBudgetSelector's own floor for these is 1_000 (or lower for
    # small n_points -- see MemoryBudgetSelector._create_chunked_config),
    # but CMAESConfig.data_chunk_size requires >= 1024 (numerical
    # stability) and raises ValueError otherwise -- clamp up to that
    # CMA-ES-specific minimum rather than passing the budget's value
    # through unchanged.
    budget_chunk_size = getattr(memory_config, "chunk_size", None) or getattr(
        memory_config, "streaming_batch_size", None
    )
    if budget_chunk_size is not None:
        budget_chunk_size = max(budget_chunk_size, 1024)

    # FR-003: Set data_chunk_size for streaming/chunked strategies
    if strategy == "streaming":
        if cmaes_config.data_chunk_size is None:
            n_points = len(ydata)
            chunk_size = budget_chunk_size or min(100_000, max(10_000, n_points // 10))
            cmaes_config = dataclasses.replace(
                cmaes_config,
                data_chunk_size=chunk_size,
            )
    elif strategy == "chunked":
        if cmaes_config.data_chunk_size is None:
            n_points = len(ydata)
            chunk_size = budget_chunk_size or min(500_000, max(50_000, n_points // 5))
            cmaes_config = dataclasses.replace(
                cmaes_config,
                data_chunk_size=chunk_size,
            )

    reject_all_zero_weight(sigma, kwargs.get("data_mask"))
    xdata, ydata, sigma = _preprocess_nonfinite_for_cmaes(
        xdata, ydata, sigma, kwargs.get("nan_policy"), check_finite, kwargs
    )
    xdata_in = xdata  # return_eval covers the rows before the user data_mask
    n_eval_rows = len(ydata)
    data_mask = kwargs.pop("data_mask", None)
    xdata, ydata, sigma = _apply_data_mask(xdata, ydata, sigma, data_mask)

    optimizer = CMAESOptimizer(config=cmaes_config)
    t_search = time.perf_counter()

    # timeit/return_eval/full_output would flow through **kwargs into
    # _nlsq_refinement's internal curve_fit() call and make it return a
    # plain tuple instead of a CurveFitResult; _nlsq_refinement then does
    # result.x/result.pcov on that tuple, which raises AttributeError,
    # silently swallowed by a broad except there and falling back to the
    # unrefined CMA-ES point estimate (pcov=inf) while this function still
    # reports success=True below. Mirrors the same filter already applied
    # in the standalone method='cmaes' entry point above.
    extra_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ("timeit", "return_eval", "full_output")
    }

    result_dict = optimizer.fit(
        f=f,
        xdata=xdata,
        ydata=ydata,
        p0=np.asarray(p0) if p0 is not None else None,
        bounds=bounds,
        sigma=np.asarray(sigma) if sigma is not None else None,
        # absolute_sigma has no dedicated param on CMAESOptimizer.fit(), but
        # its **kwargs flows through to _nlsq_refinement's curve_fit() call,
        # which does accept it -- forward it there instead of dropping it.
        absolute_sigma=absolute_sigma,
        check_finite=check_finite,
        **extra_kwargs,
    )

    result = _cmaes_curve_fit_result(
        result_dict,
        f,
        xdata,
        ydata,
        sigma,
        absolute_sigma,
        # an unrefined CMA-ES result carries an inf placeholder, not a covariance
        result_dict["pcov"] if kwargs.get("compute_covariance", True) else None,
    )
    # the flags were stripped from the nested fit above: format the result here
    return _apply_output_flags(
        result,
        f,
        xdata_in,
        {**kwargs, "data_mask": data_mask, "n_eval_rows": n_eval_rows},
        ctime=time.perf_counter() - t_search,
    )


def _fit_global_multistart(
    f: ModelFunction,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: ArrayLike | None,
    sigma: ArrayLike | None,
    absolute_sigma: bool,
    check_finite: bool,
    bounds: tuple[np.ndarray, np.ndarray],
    strategy: str,
    n_starts: int,
    memory_config: LDMemoryConfig | HybridStreamingConfig | None = None,
    workflow_name: str = "auto_global",
    user_kwargs: frozenset[str] = frozenset(),
    **kwargs: Any,
) -> CurveFitResult:
    """Run multi-start optimization with memory-aware strategy.

    Parameters
    ----------
    f : callable
        Model function f(x, *params) -> y.
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    p0 : ArrayLike | None
        Initial parameter guess.
    sigma : ArrayLike | None
        Uncertainties in ydata.
    absolute_sigma : bool
        Whether sigma is absolute.
    check_finite : bool
        Check for finite input values.
    bounds : tuple[np.ndarray, np.ndarray]
        Parameter bounds (lower, upper).
    strategy : str
        Memory strategy: 'standard', 'chunked', or 'streaming'.
    memory_config : LDMemoryConfig | HybridStreamingConfig | None
        The config MemoryBudgetSelector actually computed from the real
        (detected or overridden) memory budget. Used as-is when it matches
        the selected strategy so the real budget isn't silently replaced by
        a hardcoded default.
    n_starts : int
        Number of multi-start runs.
    workflow_name : str
        The caller's actual `workflow=` value ('hpc' or 'auto_global') --
        used only for accurate warning/error text below, since this
        function is reached from both entry points.
    user_kwargs : frozenset[str]
        Option names the caller passed, before goal-derived tolerances were
        injected into ``kwargs`` (so an ignored ``ftol``/``xtol`` is only
        reported when the caller set it).
    **kwargs : Any
        Additional fit options. The internal ``_hpc_checkpoint_*``,
        ``_hpc_run_id`` and ``_hpc_model_id`` markers are dropped (a
        ``UserWarning`` is emitted when ``_hpc_checkpoint_dir`` is set); the
        rest are forwarded to the selected fitting route.
    """
    reject_all_zero_weight(sigma, kwargs.get("data_mask"))
    if kwargs.pop("_hpc_checkpoint_dir", None) is not None:
        warnings.warn(
            f"workflow='{workflow_name}': checkpoint_dir was provided, but "
            "this route (non-CMA-ES) does not support checkpoint/crash-"
            "recovery yet -- no checkpoint file will be written. Only the "
            "CMA-ES route (restart_strategy='none') supports checkpointing "
            "currently.",
            UserWarning,
            stacklevel=2,
        )
        kwargs.pop("_hpc_checkpoint_interval", None)
        kwargs.pop("_hpc_run_id", None)
        kwargs.pop("_hpc_model_id", None)

    if strategy == "standard":
        # Standard curve_fit with multi-start
        return curve_fit(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            multistart=True,
            n_starts=n_starts,
            **kwargs,
        )
    if strategy == "chunked":
        # Chunked processing with multi-start
        from nlsq.streaming.large_dataset import LDMemoryConfig
        from nlsq.streaming.strategy import build_large_strategy

        # Use the selector's real budget-derived config when present;
        # only fall back to a default if this was reached without one
        # (e.g. a direct/legacy caller of this function).
        config = (
            memory_config
            if isinstance(memory_config, LDMemoryConfig)
            else LDMemoryConfig(
                memory_limit_gb=8.0,
                min_chunk_size=1000,
                max_chunk_size=1_000_000,
            )
        )
        large_strategy = build_large_strategy(
            "chunked", config, memory_limit_gb=config.memory_limit_gb
        )
        result = large_strategy.fit(
            f,
            xdata,
            ydata,
            np.asarray(p0) if p0 is not None else None,
            bounds=bounds,
            method="trf",
            multistart=True,
            n_starts=n_starts,
            check_finite=check_finite,
            absolute_sigma=absolute_sigma,
            **({"sigma": sigma} if sigma is not None else {}),
            **kwargs,
        )
        # LargeDatasetFitter's own exception handling returns a
        # "successful-looking" OptimizeResult with success=False on failure
        # rather than raising -- check it explicitly rather than silently
        # treating a failed chunked fit as a normal result.
        if not result.get("success", True) and not result.get("nonconverged", False):
            raise RuntimeError(
                f"workflow='{workflow_name}' chunked multi-start fit "
                f"failed: {result.get('message', 'unknown error')}",
            )
        x_fit, y_fit, s_fit = _rows_actually_fit(xdata, ydata, sigma, kwargs)
        if not isinstance(result, CurveFitResult):
            result = CurveFitResult(result)
            result["model"] = f
            result["xdata"] = x_fit
            result["ydata"] = y_fit
        _store_sigma(result, s_fit, absolute_sigma)
        return result
    # Streaming with multi-start (FR-004)
    from nlsq.streaming.hybrid_config import HybridStreamingConfig
    from nlsq.streaming.strategy import (
        build_large_strategy,
        scipy_layout_to_points_first,
    )

    _check_finite_for_streaming(xdata, ydata, check_finite)
    _reject_options_ignored_by_hybrid(
        kwargs,
        (set(kwargs) & {"f_scale", "max_nfev", "x_scale", "jac"})
        | (set(user_kwargs) & {"ftol", "xtol"}),
    )

    # Prepare p0
    if p0 is None or _is_auto_p0(p0):
        n_params = _n_params_from_signature(f)
        # the default ones may lie outside the (required, finite) bounds
        p0 = _initialize_feasible(*prepare_bounds(bounds, n_params))
    p0_arr = np.atleast_1d(p0)

    import dataclasses

    # Reuse the selector's budget-derived chunk_size (real detected/
    # overridden memory) when available; only build a fresh default
    # config if this was reached without one.
    base_config = (
        memory_config
        if isinstance(memory_config, HybridStreamingConfig)
        else HybridStreamingConfig()
    )
    config = dataclasses.replace(
        base_config,
        normalize=True,
        warmup_iterations=200,
        gauss_newton_tol=(
            kwargs["gtol"]
            if kwargs.get("gtol") is not None and kwargs["gtol"] > 0
            else 1e-8
        ),
        enable_multistart=True,
        n_starts=n_starts,
    )

    large_strategy = build_large_strategy("hybrid", config)
    f_engine, x_engine = scipy_layout_to_points_first(f, xdata)
    result_dict = large_strategy.fit(
        f_engine,
        x_engine,
        ydata,
        p0_arr,
        bounds=bounds,
        sigma=np.asarray(sigma) if sigma is not None else None,
        absolute_sigma=absolute_sigma,
        callback=kwargs.get("callback"),
        verbose=kwargs.get("verbose", 0),
    )

    result = CurveFitResult(result_dict)
    result["model"] = f
    result["xdata"] = xdata
    result["ydata"] = ydata
    result["pcov"] = result_dict.get(
        "pcov",
        np.full((len(p0_arr), len(p0_arr)), np.inf),
    )
    _store_sigma(result, sigma, absolute_sigma)
    return result


# =============================================================================
# Helper functions for curve_fit complexity reduction (T003-T007)
# =============================================================================


def _extract_p0_from_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> NDArray[np.floating[Any]] | None:
    """Extract p0 initial guess from positional or keyword arguments.

    Parameters
    ----------
    args : tuple
        Positional arguments passed to curve_fit.
    kwargs : dict
        Keyword arguments passed to curve_fit.

    Returns
    -------
    p0 : NDArray or None
        Initial parameter guess, or None if not provided.
    """
    if args and len(args) >= 1:
        return args[0]
    return kwargs.get("p0")


def _is_bounded(bounds: Any, n_params: int) -> bool:
    """Whether any parameter has a finite lower or upper bound.

    Selects the bounded trust-region memory layout (extra p x p blocks) for the
    memory budget; accepts ``None``, scalar/broadcast bounds and SciPy ``Bounds``.
    """
    if bounds is None:
        return False
    lb, ub = prepare_bounds(_bounds_as_tuple(bounds), n_params)
    return bool(np.any(np.isfinite(lb)) or np.any(np.isfinite(ub)))


def _log_memory_budget_diagnostics(
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: NDArray[np.floating[Any]] | None,
    bounds: Any = None,
) -> None:
    """Log memory budget diagnostics when verbose >= 2.

    Shows memory estimates and suggested strategy based on current system.

    Parameters
    ----------
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    p0 : array_like or None
        Initial parameter guess.
    bounds : tuple or None, optional
        Parameter bounds. Only used to decide whether the bounded trust-region
        memory layout applies to the estimate.
    """
    from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector

    logger = get_logger("nlsq")

    # Determine n_params from p0 or estimate from xdata/ydata shape.
    if p0 is not None and not _is_auto_p0(p0):
        n_params = len(np.atleast_1d(p0))
    else:
        n_params = 3  # Default estimate

    n_points = len(np.asarray(ydata))  # points, not features of a (k, n) xdata

    # Compute memory budget once and reuse it for selection so the logged
    # numbers can't disagree with the actual decision (memory detection is
    # a live psutil/GPU read, not guaranteed identical across two calls).
    try:
        bounded = _is_bounded(bounds, n_params)
        budget = MemoryBudget.compute(
            n_points=n_points, n_params=n_params, bounded=bounded
        )

        # Get strategy recommendation
        selector = MemoryBudgetSelector()
        strategy, _ = selector.select(
            n_points=n_points,
            n_params=n_params,
            verbose=False,  # Already logging here
            budget=budget,
            bounded=bounded,
        )

        logger.info(
            f"[NLSQ] Memory budget: available={budget.available_gb:.1f} GB, "
            f"threshold={budget.threshold_gb:.1f} GB",
        )
        logger.info(
            f"[NLSQ] Estimates: data={budget.data_gb:.2f} GB, "
            f"jacobian={budget.jacobian_gb:.2f} GB, peak={budget.peak_gb:.2f} GB",
        )
        # The chunked/standard boundary applies a 10% safety margin
        # (budget.chunked_threshold_gb, FR-010) -- compare against that,
        # not the raw threshold, to match what select() actually decided.
        logger.info(
            f"[NLSQ] Strategy: {strategy} "
            f"(peak {budget.peak_gb:.2f} GB "
            f"{'<' if budget.peak_gb <= budget.chunked_threshold_gb else '>'} "
            f"safety threshold {budget.chunked_threshold_gb:.1f} GB)",
        )
    except Exception as e:
        logger.debug(f"[NLSQ] Memory budget diagnostics unavailable: {e}")


def _prepare_wrapper_p0(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: Any,
    stability: Any,
    multistart: bool,
    method: str | None,
    kwargs: dict[str, Any],
) -> Any:
    """Validate method="cmaes" and resolve p0="auto" for the wrapper routes.

    The stability, CMA-ES and multi-start routes need a numeric p0; the plain
    CurveFit path resolves the "auto" sentinel itself.
    """
    bounds = kwargs.get("bounds")
    if method == "cmaes" and bounds is None:
        raise ValueError('method="cmaes" requires bounds')
    if _is_auto_p0(p0) and (
        stability
        or multistart
        or method in ("cmaes", "multi-start")
        or (method == "auto" and bounds is not None)
    ):
        p0 = _resolve_auto_p0(f, xdata, ydata, kwargs.get("sigma"), kwargs)
        kwargs["p0"] = p0
    return p0


def _resolve_auto_p0(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    sigma: ArrayLike | None = None,
    options: dict[str, Any] | None = None,
) -> NDArray[np.floating[Any]] | None:
    """Replace the ``p0="auto"`` sentinel with an estimate (None if it fails).

    Mirrors ``problem._determine_parameter_count``: on failure p0 becomes None
    so downstream routes use their own defaults. The estimate sees only the rows
    the fit will use: ``options`` may carry ``data_mask`` and ``nan_policy``.
    """
    from nlsq.precision.parameter_estimation import estimate_initial_parameters

    options = options or {}
    try:
        xdata, ydata = rows_used_by_fit(
            xdata,
            ydata,
            sigma,
            options.get("data_mask"),
            options.get("nan_policy"),
        )
        x_f, y_f = finite_rows(xdata, ydata)
        return np.atleast_1d(estimate_initial_parameters(f, x_f, y_f, "auto"))
    except Exception as e:
        _logger.warning(f"p0='auto' estimation failed, using defaults: {e}")
        return None


def _apply_auto_bounds(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: NDArray[np.floating[Any]] | Literal["auto"] | None,
    bounds_safety_factor: float,
    kwargs: dict[str, Any],
    method: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply automatic bounds inference from data characteristics.

    Infers reasonable bounds based on data ranges, initial parameter guess,
    and parameter positivity constraints. User-provided bounds take precedence.

    Parameters
    ----------
    f : callable
        Model function, used to estimate p0 when the caller did not supply one.
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    p0 : NDArray, "auto", or None
        Initial parameter guess. If None or "auto", one is estimated from
        the data so that bounds inference still runs.
    bounds_safety_factor : float
        Safety multiplier for automatic bounds.
    kwargs : dict
        Keyword arguments dict to update with merged bounds.
    method : str or None
        Requested method; 'lm' cannot be combined with inferred bounds.

    Returns
    -------
    user_lb, user_ub : ndarray
        The caller's bounds per parameter (infinite where unset).
    """
    from nlsq.precision.bound_inference import infer_bounds, merge_bounds

    if method == "lm":
        raise ValueError(
            "auto_bounds=True infers parameter bounds, which method='lm' does "
            "not support; use method='trf', or set auto_bounds=False."
        )

    # data_mask-excluded and (nan_policy='omit') non-finite rows are dropped
    # from the fit later; inference must not see them.
    if kwargs.get("data_mask") is not None:
        xdata, ydata, _ = _apply_data_mask_points_last(
            xdata, ydata, None, kwargs["data_mask"]
        )
    xdata, ydata = finite_rows(xdata, ydata)

    if p0 is None or _is_auto_p0(p0):
        from nlsq.precision.parameter_estimation import estimate_initial_parameters

        p0 = estimate_initial_parameters(f, xdata, ydata, None)

    # Infer bounds from data
    inferred_bounds = infer_bounds(
        xdata,
        ydata,
        p0,
        safety_factor=bounds_safety_factor,
    )

    # Get user-provided bounds if any
    user_bounds = kwargs.get("bounds", (-np.inf, np.inf))

    # Merge inferred with user bounds (user takes precedence)
    merged_bounds = merge_bounds(inferred_bounds, user_bounds)

    # Update kwargs with merged bounds
    kwargs["bounds"] = merged_bounds

    # User bounds as per-parameter arrays (+-inf where unset): the sides still
    # infinite are the ones that carry an *inferred* bound.
    n_params = len(merged_bounds[0])
    user_lb, user_ub = (
        np.broadcast_to(np.asarray(b, dtype=float), (n_params,)).copy()
        for b in user_bounds
    )
    return user_lb, user_ub


def _raise_clear_jac_trace_error(exc: Exception, jac: Any) -> None:
    """Re-raise a JAX tracing failure under a user ``jac`` as a clear TypeError."""
    if isinstance(exc, TracerArrayConversionError) and callable(jac):
        raise TypeError(
            "jac (and the model) must be JAX-traceable: use jax.numpy, not "
            "numpy, inside a user-supplied jac."
        ) from exc


def _drop_internal_routing_kwargs(kwargs: dict[str, Any]) -> None:
    """Drop routing keys no optimizer layer reads (curve_fit_large / multi-start
    plumbing), so anything left for least_squares is a genuinely unknown keyword."""
    for key in (
        "nan_policy",
        "multistart",
        "n_starts",
        "sampler",
        "center_on_p0",
        "scale_factor",
    ):
        kwargs.pop(key, None)


_OUTPUT_FLAGS = ("full_output", "return_eval", "timeit")

# A ``return_eval`` return is ``(popt, pcov, f(x))`` and carries no status or
# cost. While a list is installed here, the routes append the result object
# behind each such return, so auto_bounds can rank attempts by their real
# convergence status and cost without another solve.
_ATTEMPT_RECORD: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar(
    "_ATTEMPT_RECORD", default=None
)


def _record_attempt(result: Any) -> None:
    rec = _ATTEMPT_RECORD.get()
    if rec is not None:
        rec.append(result)


def _eval_attempt_cost(
    result: Any,
    kwargs: dict[str, Any],
    xdata: Any,
    ydata: Any,
    loss: Any,
    f_scale: Any,
) -> float:
    """Loss-aware cost of a ``return_eval`` return, rebuilt from ``f(x)``.

    The evaluation is over the rows left after ``nan_policy='omit'`` (the main
    path evaluates masked rows too; CMA-ES returns the masked rows already
    dropped). ``sigma`` may be a scalar, per-point values or a covariance
    matrix (whitened over the retained block).
    """
    from nlsq.stability.fallback import _robust_cost

    sigma = kwargs.get("sigma")
    opts = {"data_mask": kwargs.get("data_mask")}
    x, y = np.asarray(xdata, dtype=float), np.asarray(ydata, dtype=float)
    if kwargs.get("nan_policy") == "omit":
        x, y, sigma = _preprocess_nonfinite_for_cmaes(
            x, y, sigma, "omit", False, opts, allow_rowwise=True
        )
    mask = None if opts["data_mask"] is None else np.asarray(opts["data_mask"], bool)
    feval = np.asarray(result[2], dtype=float)
    if feval.shape == y.shape:
        res = feval - y
        keep = np.ones(len(y), bool) if mask is None else mask
    elif mask is not None and feval.shape == (int(mask.sum()),):
        res, keep = feval - y[mask], np.ones(int(mask.sum()), bool)
        sigma = None if sigma is None else _mask_sigma(sigma, mask)
        mask = None
    else:
        return np.inf
    if mask is not None and sigma is not None:
        sigma = _mask_sigma(sigma, mask)
    res = res[keep] if mask is not None else res
    if sigma is not None:
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 2:
            import scipy.linalg as sla

            res = sla.solve_triangular(np.linalg.cholesky(sig), res, lower=True)
        else:
            res = res / sig
    return _robust_cost(res, loss, f_scale)


def _mask_sigma(sigma: Any, mask: np.ndarray) -> np.ndarray:
    """Retained entries of a per-point sigma or the retained covariance block."""
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim == 0:
        return sig
    return sig[np.ix_(mask, mask)] if sig.ndim == 2 else sig[mask]


def _auto_bounds_cost(
    result: Any,
    flags: dict[str, Any],
    kwargs: dict[str, Any],
    xdata: Any,
    ydata: Any,
) -> float:
    """Loss-aware cost of an attempt's return, ``inf`` when it cannot be derived.

    Result objects carry ``cost``; ``full_output`` carries the residual vector
    and ``timeit`` the raw result. ``return_eval`` carries neither, so its cost
    is rebuilt from ``f(x)`` (see ``_eval_attempt_cost``).
    """
    from nlsq.stability.fallback import _attempt_cost, _tuple_kind

    loss = kwargs.get("loss", "linear")
    f_scale = kwargs.get("f_scale") or 1.0
    try:
        if _tuple_kind(result, flags) != "eval":
            cost = _attempt_cost(result, loss, loss, flags, f_scale)
        else:
            cost = _eval_attempt_cost(result, kwargs, xdata, ydata, loss, f_scale)
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        np.linalg.LinAlgError,
    ):
        return np.inf
    return cost if np.isfinite(cost) else np.inf


def _curve_fit_with_auto_bounds(
    run: Callable[..., Any],
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    bounds_safety_factor: float,
    method: str | None,
    kwargs: dict[str, Any],
) -> Any:
    """Run ``curve_fit`` with inferred bounds, guarding against heuristic clipping.

    ``run(**kwargs)`` is a bound ``curve_fit`` (``auto_bounds`` off). If a fitted
    parameter sits on a bound that was *inferred* (not set by the caller), refit
    once with that side relaxed to the caller's bound, starting from the result;
    the refit is kept (with an OptimizeWarning) only if it converges at least
    as well and its loss-aware cost is lower.
    The guard works on every route (multistart, fallback, CMA-ES, ...). Every
    attempt runs with the caller's output flags (``full_output``,
    ``return_eval``, ``timeit``), so the chosen attempt's return is handed back
    verbatim: the flag never changes the fit and callbacks run once per
    attempt. A ``return_eval``-only return carries no status or cost; it is
    ranked by its rebuilt cost where possible, otherwise the original is kept.
    """
    from nlsq.stability.fallback import _fit_succeeded, _fitted_params

    user_lb, user_ub = _apply_auto_bounds(
        f,
        xdata,
        ydata,
        kwargs.get("p0"),
        bounds_safety_factor,
        kwargs,
        kwargs.get("method") or method,
    )
    flags = {k: kwargs[k] for k in _OUTPUT_FLAGS if kwargs.get(k)}
    records: dict[int, Any] = {}

    def attempt(**kw):
        # Capture the result object behind a return_eval tuple (id-keyed).
        token = _ATTEMPT_RECORD.set([])
        try:
            out = run(**kw)
            rec = _ATTEMPT_RECORD.get()
        finally:
            _ATTEMPT_RECORD.reset(token)
        if flags.get("return_eval") and rec:
            records[id(out)] = rec[-1]
        return out

    result = attempt(**kwargs)

    popt = _fitted_params(result, flags)
    if popt is not None:
        lb, ub = (np.asarray(b, dtype=float) for b in kwargs["bounds"])
        at_lb = np.isinf(user_lb) & np.isfinite(lb)
        at_lb &= popt - lb <= 1e-6 * np.maximum(1.0, np.abs(lb))
        at_ub = np.isinf(user_ub) & np.isfinite(ub)
        at_ub &= ub - popt <= 1e-6 * np.maximum(1.0, np.abs(ub))
        if at_lb.any() or at_ub.any():
            relaxed = {
                "bounds": (np.where(at_lb, user_lb, lb), np.where(at_ub, user_ub, ub)),
                "p0": popt,
            }
            try:
                refit = attempt(**{**kwargs, **relaxed})
            except Exception:  # keep the original result if the relaxed fit fails
                refit = None
            refit_popt = None if refit is None else _fitted_params(refit, flags)
            if refit_popt is not None and np.all(np.isfinite(refit_popt)):

                def rank(r):
                    # Prefer the recorded result object (real status and cost).
                    rec = records.get(id(r))
                    r, fl = (r, flags) if rec is None else (rec, {})
                    return (
                        not _fit_succeeded(r, fl),
                        _auto_bounds_cost(r, fl, kwargs, xdata, ydata),
                    )

                rank_refit, rank_result = rank(refit), rank(result)
                if (
                    rank_refit[0] == rank_result[0]
                    and rank_refit[1] == rank_result[1] == np.inf
                ):
                    import warnings

                    why = (
                        "the cost of the return_eval output could not be rebuilt"
                        if flags.get("return_eval")
                        else "neither fit reports a cost"
                    )
                    warnings.warn(
                        f"auto_bounds: parameter(s) ended on an inferred bound but "
                        f"{why}, so the relaxed refit cannot be compared; keeping "
                        "the fit clipped to the inferred bound. Pass explicit "
                        "bounds to override.",
                        OptimizeWarning,
                        stacklevel=3,
                    )
                elif rank_refit < rank_result:
                    import warnings

                    idx = np.flatnonzero(at_lb | at_ub).tolist()
                    warnings.warn(
                        f"auto_bounds: parameter(s) {idx} ended on an inferred "
                        "bound, which is only a heuristic; refit without it. "
                        "Pass explicit bounds to override.",
                        OptimizeWarning,
                        stacklevel=3,
                    )
                    result = refit
    return result


def _apply_stability_checks(
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: NDArray[np.floating[Any]] | None,
    f: ModelFunction,
    stability: Literal["auto", "check"],
    rescale_data: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[ArrayLike, ArrayLike, tuple[Any, ...]]:
    """Apply numerical stability checks and optional automatic fixes.

    Parameters
    ----------
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    p0 : NDArray or None
        Initial parameter guess.
    f : ModelFunction
        The model function.
    stability : {'auto', 'check'}
        Stability mode - both warn; neither modifies the data or p0.
    rescale_data : bool
        When True and stability='auto', warn (rather than silently do
        nothing) if the data would have qualified for rescaling. The
        p0 scale-mismatch warning is emitted regardless of this flag. Nothing
        is rescaled or normalized -- see the "auto" fix note in this
        function's implementation for why.
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    xdata, ydata, args
        Returned unchanged (rescaling and NaN/Inf replacement are disabled);
        kept in the signature for compatibility.
    """
    from nlsq.stability.guard import apply_automatic_fixes, check_problem_stability

    logger = get_logger("minpack")

    # Check stability
    stability_report = check_problem_stability(xdata, ydata, p0, f)

    # Handle based on stability mode
    if stability == "check":
        # Just check and warn
        if stability_report["severity"] == "critical":
            logger.warning(
                f"Critical stability issues detected ({len(stability_report['issues'])} issues):",
            )
            for issue_type, message, severity in stability_report["issues"]:
                logger.warning(f"  [{severity.upper()}] {message}")
            if stability_report["recommendations"]:
                logger.info("Recommendations:")
                for rec in stability_report["recommendations"]:
                    logger.info(f"  - {rec}")
        elif stability_report["severity"] == "warning":
            logger.warning(
                f"Stability warnings detected ({len(stability_report['issues'])} issues)",
            )
            for issue_type, message, severity in stability_report["issues"]:
                logger.warning(f"  [{severity.upper()}] {message}")

    elif stability == "auto":
        # Apply automatic fixes if issues detected
        if stability_report["severity"] in ["warning", "critical"]:
            logger.info(
                f"Applying automatic fixes for {len(stability_report['issues'])} stability issues...",
            )

            # NOTE: apply_automatic_fixes() can also rescale xdata/ydata to
            # [0, 1] (data rescaling) and split p0 by order of magnitude
            # (p0 normalization).
            # Those two are NOT applied here: they change the coordinate
            # system / initial-guess semantics that `f` is evaluated in
            # without reparameterizing `f` itself, so the popt this function
            # (and curve_fit) ultimately returns would be silently wrong --
            # in the wrong units for a rescaled fit, or started from a
            # nonsensical point for a "normalized" p0, with no way to invert
            # either for an arbitrary nonlinear model. NaN/Inf imputation is
            # also disabled (replace_nonfinite=False): it would silently alter
            # the data before nan_policy/check_finite can raise or omit.
            # rescale_data=False is forced for this call; the caller's flag
            # only controls the warning below.
            xdata_fixed, ydata_fixed, _p0_unused, fix_info = apply_automatic_fixes(
                xdata,
                ydata,
                p0,
                stability_report=stability_report,
                rescale_data=False,
                replace_nonfinite=False,
            )

            # Update data (no-op: NaN/Inf is left to nan_policy/check_finite)
            xdata = xdata_fixed
            ydata = ydata_fixed

            # Log applied fixes -- excluding the p0-normalization message.
            # apply_automatic_fixes()'s Fix 4 (p0 order-of-magnitude
            # normalization) is unconditional in guard.py, not gated by
            # rescale_data like Fix 1/2 are, so it still runs and appends
            # "Normalized p0..." to applied_fixes even though we deliberately
            # discard p0_fixed (_p0_unused) above. Logging that message would
            # claim a fix happened when the fit is actually still using the
            # original, un-normalized p0.
            for fix in fix_info["applied_fixes"]:
                if "Normalized p0" not in fix:
                    logger.info(f"  - {fix}")

            # Inform the caller when a fix would have triggered but was
            # skipped, since it is now a no-op rather than a silent-but-wrong
            # transform.
            cond = stability_report["condition_number"]
            x_range = np.ptp(np.asarray(xdata)) if np.size(xdata) else 0
            y_range = np.ptp(np.asarray(ydata)) if np.size(ydata) else 0
            would_rescale_data = (
                (cond is not None and cond > 1e10) or x_range > 1e4 or y_range > 1e4
            )
            # Matches Fix 4's own trigger condition in guard.py exactly (it
            # is unconditional, not gated by rescale_data).
            would_normalize_p0 = (
                p0 is not None
                and stability_report["parameter_scale_ratio"]
                and stability_report["parameter_scale_ratio"] > 1e6
            )
            if rescale_data and would_rescale_data:
                logger.warning(
                    "  Data rescaling was NOT applied: automatically rescaling "
                    "xdata/ydata cannot be soundly inverted back to your model's "
                    "original units for an arbitrary nonlinear model, so it would "
                    "silently return parameters in the wrong coordinate system. "
                    "If conditioning is a problem, pass x_scale='jac' (or an "
                    "explicit per-parameter scale) to curve_fit, or rescale your "
                    "data yourself and interpret the fitted parameters accordingly.",
                )
            if would_normalize_p0:
                logger.warning(
                    "  p0 normalization was NOT applied: rescaling your initial "
                    "guess to unit order of magnitude without also "
                    "reparameterizing the model would start the optimizer from "
                    "a point that no longer matches the model's actual "
                    "parameter units. If p0 components span many orders of "
                    "magnitude, pass x_scale='jac' to curve_fit or rescale p0 "
                    "yourself alongside the model.",
                )

    return xdata, ydata, args


def _reject_options_ignored_by_hybrid(
    kwargs: dict[str, Any], ignored: set[str] | frozenset[str] = frozenset()
) -> None:
    """Fail loudly on options the hybrid streaming engine cannot honour.

    It minimizes plain least squares over every row, so a robust loss,
    ``data_mask`` and ``nan_policy='omit'`` would otherwise be silently dropped;
    ``ignored`` option names (tolerances etc.) only warn.
    """
    if kwargs.get("loss", "linear") != "linear":
        raise ValueError(
            "The hybrid streaming workflow does not support robust loss "
            f"(loss={kwargs['loss']!r}); it minimizes plain least squares. "
            "Use a standard workflow (e.g. fit(..., workflow='auto', "
            "loss=...) or curve_fit(..., loss=...)) for robust loss."
        )
    if kwargs.get("data_mask") is not None or kwargs.get("nan_policy") == "omit":
        raise ValueError(
            "The hybrid streaming workflow does not support data_mask or "
            "nan_policy='omit'; filter the data before calling fit()."
        )
    if ignored:
        warnings.warn(
            f"The hybrid streaming workflow ignores {sorted(ignored)}; configure "
            "HybridStreamingConfig instead.",
            OptimizeWarning,
            stacklevel=4,
        )


def _apply_data_mask_points_last(
    xdata: Any, ydata: Any, sigma: Any, data_mask: Any
) -> tuple[Any, Any, Any]:
    """``_apply_data_mask`` for SciPy-layout wrappers: a square ``(k, n)`` xdata is
    unambiguous there (the points are on the last axis)."""
    from nlsq.global_optimization.cmaes_optimizer import _apply_data_mask

    x = np.asarray(xdata)
    if data_mask is None or not (
        x.ndim > 1 and x.shape[0] == x.shape[-1] == len(np.asarray(ydata))
    ):
        return _apply_data_mask(xdata, ydata, sigma, data_mask)
    # validate/filter y and sigma with an unambiguous 1-D stand-in for x
    _, ydata, sigma = _apply_data_mask(
        np.empty(len(np.asarray(ydata))), ydata, sigma, data_mask
    )
    return np.compress(np.asarray(data_mask, bool), x, axis=-1), ydata, sigma


def _rows_actually_fit(
    xdata: Any,
    ydata: Any,
    sigma: Any,
    kwargs: dict[str, Any],
    apply_mask: bool = True,
) -> tuple[Any, Any, Any]:
    """x/y/sigma restricted to the rows a chunked fit used.

    The chunk engines apply ``nan_policy='omit'`` and ``data_mask`` internally
    and return only the parameters, so the stored data must be filtered the
    same way or every statistic is computed over rows the caller dropped.
    ``apply_mask=False`` keeps the ``data_mask`` rows (the ``return_eval`` rows),
    using the mask only to tolerate non-finite sigma on rows it excludes.
    """
    opts = {"data_mask": kwargs.get("data_mask")}
    if kwargs.get("nan_policy") == "omit":
        xdata, ydata, sigma = _preprocess_nonfinite_for_cmaes(
            xdata, ydata, sigma, "omit", False, opts, allow_rowwise=True
        )
    if not apply_mask:
        return xdata, ydata, sigma
    return _apply_data_mask_points_last(xdata, ydata, sigma, opts["data_mask"])


def _store_sigma(result: Any, sigma: Any, absolute_sigma: bool) -> None:
    """Keep the fit's weighting on the result.

    ``CurveFitResult`` weights aic/bic and the prediction-interval noise term by
    these two keys, so every construction site must set them (``sigma`` must
    already be aligned with the stored, possibly masked, ``ydata``).
    """
    result["sigma"] = sigma
    result["absolute_sigma"] = absolute_sigma


def _cmaes_termination_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Preserve optimizer/refinement termination when adapting a CMA-ES result."""
    refinement = result.get("nlsq_result") or {}
    success = result.get("success", refinement.get("success", True))
    status = result.get("status", refinement.get("status", 1 if success else 0))
    return {
        "success": bool(success),
        "status": status,
        "message": result.get(
            "message", refinement.get("message", "CMA-ES optimization completed")
        ),
    }


def _residuals_in_chunks(
    f: ModelFunction,
    xdata: Any,
    ydata: Any,
    popt: Any,
    chunk: int = 1 << 20,
) -> np.ndarray:
    """``f(x, *popt) - y`` evaluated ``chunk`` points at a time.

    ``xdata`` is ``(n,)`` or scipy-layout ``(k, n)``.
    """
    y = np.asarray(ydata, dtype=float)
    x = np.asarray(xdata)
    out = np.empty(len(y))
    for lo in range(0, len(y), chunk):
        xc = x[..., lo : lo + chunk]
        out[lo : lo + chunk] = (
            np.asarray(f(jnp.asarray(xc), *popt)) - y[lo : lo + chunk]
        )
    return out


def _cmaes_curve_fit_result(
    result_dict: dict[str, Any],
    f: ModelFunction,
    xdata: Any,
    ydata: Any,
    sigma: Any,
    absolute_sigma: bool,
    pcov: Any,
) -> CurveFitResult:
    """Adapt a ``CMAESOptimizer.fit`` dict to a ``CurveFitResult``.

    The NLSQ refinement (when run) carries the real cost/jacobian/evaluation
    counts; without refinement CMA-ES provides none of them. Shared by the
    ``curve_fit(method='cmaes')`` and ``fit(workflow='auto_global')`` routes so
    both expose the same fields.
    """
    refined = result_dict.get("nlsq_result")

    def _ref(name: str, default: Any = None) -> Any:
        try:
            return refined[name] if refined is not None else default
        except (KeyError, TypeError, IndexError):
            return getattr(refined, name, default)

    result = CurveFitResult(
        x=result_dict["popt"],
        # historical key of the workflow route; consumers index it
        popt=result_dict["popt"],
        pcov=pcov,
        cost=_ref("cost"),
        fun=_ref("fun"),
        jac=_ref("jac"),
        grad=_ref("grad"),
        optimality=_ref("optimality"),
        active_mask=_ref("active_mask"),
        nfev=_ref("nfev", 0),
        njev=_ref("njev", 0),
        nit=_ref("nit", 0),
        **_cmaes_termination_fields(result_dict),
        xdata=np.asarray(xdata),
        ydata=np.asarray(ydata),
        model=f,
    )
    _store_sigma(result, sigma, absolute_sigma)
    if "cmaes_diagnostics" in result_dict:
        result["cmaes_diagnostics"] = result_dict["cmaes_diagnostics"]
    if _ref("_diagnostics_report") is not None:
        result["_diagnostics_report"] = _ref("_diagnostics_report")
    return result


_CURVE_FIT_METHODS = (
    "auto",
    "cmaes",
    "multi-start",
    "trf",
    "lm",
    "hybrid_streaming",
)


def _check_method(method: Any, valid: tuple[str, ...]) -> None:
    if method is not None and method not in valid:
        import difflib

        hint = difflib.get_close_matches(str(method), valid, n=1)
        raise ValueError(
            f"Unknown method {method!r}; valid methods are {valid}."
            + (f" Did you mean {hint[0]!r}?" if hint else "")
        )


def _check_optimizer_kwargs(kwargs: dict[str, Any]) -> None:
    """Cheap least_squares kwarg checks, so the CMA-ES path raises them too.

    The CMA-ES refinement swallows exceptions, which would turn these input
    errors into ``success=False`` results. Messages match ``least_squares``.
    """
    for name in ("ftol", "xtol", "gtol"):
        if kwargs.get(name) is not None:
            check_not_nan(name, kwargs[name])
    for key in ("max_nfev", "maxfev"):  # maxfev is the SciPy alias of max_nfev
        if key in kwargs:
            kwargs[key] = validate_max_nfev(kwargs[key], allow_fractional=True)
    validate_tr_solver(kwargs.get("tr_solver"))
    f_scale = _unwrap_scalar(kwargs.get("f_scale", 1.0))
    if isinstance(f_scale, (int, float, np.number)):
        validate_f_scale(f_scale)


def _validate_wrapper_options(
    method: Any,
    stability: Any,
    absolute_sigma: Any,
    kwargs: dict[str, Any] | None = None,
) -> None:
    """Reject option values the pipeline would otherwise silently ignore."""
    _check_method(method, _CURVE_FIT_METHODS)
    _check_optimizer_kwargs(kwargs or {})
    if stability not in (False, None, "auto", "check"):
        raise ValueError(
            f"stability must be False, 'auto' or 'check', got {stability!r}"
        )
    coerced = coerce_absolute_sigma(absolute_sigma)
    if kwargs is not None and "absolute_sigma" in kwargs:
        kwargs["absolute_sigma"] = coerced


def _check_1d_sigma(sigma: Any, kwargs: dict[str, Any]) -> None:
    """Reject 0/NaN/inf 1-D sigma like the TRF and chunked paths do.

    Skipped when ``data_mask`` is given: masked-out rows may legitimately hold
    bad sigma values and the mask is applied later.
    """
    if sigma is None:
        return
    try:
        s = np.asarray(sigma, float)
    except (TypeError, ValueError):
        return
    if s.ndim == 0:  # a scalar sigma weights every retained row
        check_sigma_values(s)
        return
    if s.ndim != 1 or kwargs.get("data_mask") is not None:
        return
    check_sigma_values(s)


def _check_sigma_length(sigma: Any, n: int) -> None:
    """A 1-D ``sigma`` must have one entry per observation (as on the TRF path)."""
    if sigma is None:
        return
    try:
        s = np.asarray(sigma, float)
    except (TypeError, ValueError):
        return
    if s.ndim == 1 and s.shape[0] != n:
        raise ValueError(
            f"sigma must have same shape as ydata: got {s.shape[0]} entries for "
            f"{n} observations"
        )


def _preprocess_nonfinite_for_cmaes(
    xdata: Any,
    ydata: Any,
    sigma: Any,
    nan_policy: str | None,
    check_finite: bool,
    kwargs: dict[str, Any],
    *,
    allow_rowwise: bool = False,
) -> tuple[Any, Any, Any]:
    """Apply the TRF path's nan_policy/check_finite handling before CMA-ES.

    CMA-ES evaluates fitness on the raw arrays, so a single NaN makes every
    candidate infinite. 'raise' (with check_finite) raises, 'omit' drops
    non-finite rows of x/y/sigma consistently, 'propagate' leaves data as is.
    """
    from nlsq.core.problem import _VALID_NAN_POLICIES, _omit_nonfinite_rows

    policy = nan_policy or "raise"
    if policy not in _VALID_NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {sorted(_VALID_NAN_POLICIES)}, got {policy!r}"
        )
    if policy == "propagate":
        return xdata, ydata, sigma
    try:
        x = np.asarray(xdata, float)
        y = np.asarray(ydata, float)
    except (TypeError, ValueError):
        return xdata, ydata, sigma
    if policy == "raise":
        _check_sigma_length(sigma, y.shape[0] if y.ndim else 1)
        if check_finite:
            np.asarray_chkfinite(y)
            np.asarray_chkfinite(x)
            _check_1d_sigma(sigma, kwargs)
        return xdata, ydata, sigma
    if (
        y.ndim != 1
        or x.ndim == 0
        or (
            x.shape[-1] != y.shape[0]
            and (not allow_rowwise or x.shape[0] != y.shape[0])
        )
    ):
        # Skipping would leave the NaNs in the data the user asked to drop.
        raise ValueError(
            "nan_policy='omit' needs xdata with shape (n,) or (k, n) matching "
            f"ydata of shape (n,); got xdata {x.shape} and ydata {y.shape}"
        )
    points_axis_last = x.shape[-1] == y.shape[0]
    ambiguous = allow_rowwise and x.ndim > 1 and x.shape[0] == x.shape[-1] == len(y)
    original_x = x
    if allow_rowwise and x.ndim > 2:
        x = x.reshape((-1, len(y)) if points_axis_last else (len(y), -1))
    x, y, sigma, _, omit_mask = _omit_nonfinite_rows(
        x,
        y,
        sigma,
        x.ndim,
        points_axis_last,
        # non-finite covariance entries on rows the caller masks out are fine
        data_mask=kwargs.get("data_mask"),
    )
    if ambiguous and not omit_mask.all():
        raise ValueError("Cannot omit nonfinite data: sample axis is ambiguous")
    if original_x.ndim > 2 and allow_rowwise:
        x = np.compress(omit_mask, original_x, axis=-1 if points_axis_last else 0)
    if kwargs.get("data_mask") is not None:
        mask = np.asarray(kwargs["data_mask"])
        if mask.shape != omit_mask.shape:
            raise ValueError("Data mask doesn't match data lengths.")
        kwargs["data_mask"] = mask[omit_mask]
    return x, y, sigma


def _auto_bipop_budget(config: CMAESConfig, n_params: int) -> int:
    """Worst-case evaluations of a fixed-2x-large BIPOP schedule for ``config``."""
    from nlsq.global_optimization.bounds_transform import compute_default_popsize

    base = config.popsize or compute_default_popsize(n_params)
    runs = 1 + (config.max_restarts if config.restart_strategy == "bipop" else 0)
    large = (runs + 1) // 2
    return (large * 2 * base + (runs - large) * base) * config.max_generations


def _run_cmaes_optimization(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: NDArray[np.floating[Any]] | None,
    bounds: tuple[ArrayLike, ArrayLike],
    method: str,
    cmaes_config: CMAESConfig | None,
    kwargs: dict[str, Any],
    compute_covariance: bool = True,
) -> CurveFitResult | None:
    """Run CMA-ES global optimization if appropriate.

    Parameters
    ----------
    f : ModelFunction
        The model function.
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    p0 : NDArray or None
        Initial parameter guess.
    bounds : tuple
        Lower and upper bounds for parameters.
    method : str
        Method requested ('cmaes' or 'auto').
    cmaes_config : CMAESConfig or None
        Configuration for CMA-ES optimization.
    kwargs : dict
        Additional keyword arguments.
    compute_covariance : bool, default=True
        When False, the covariance is skipped in the NLSQ refinement and
        ``pcov`` of the returned result is ``None``.

    Returns
    -------
    result : CurveFitResult or None
        Optimization result if CMA-ES was used, None if fallback to standard.
    """
    from nlsq.global_optimization.bounds_transform import compute_default_popsize
    from nlsq.global_optimization.cmaes_config import (
        CMAESConfig,
        auto_configure_cmaes_memory,
        cmaes_resident_bytes,
    )
    from nlsq.global_optimization.method_selector import MethodSelector

    # Convert bounds to numpy arrays for scale ratio computation. bounds may
    # be SciPy-style scalars (e.g. (0, 10)) meant to broadcast to every
    # parameter -- prepare_bounds() resizes 0-d bounds to length n_params
    # (same convention as the p0-is-None fallback used elsewhere in this file).
    lower, upper = prepare_bounds(
        bounds,
        len(np.atleast_1d(p0)) if p0 is not None else 1,
    )

    # Use MethodSelector to decide if CMA-ES should be used
    selector = MethodSelector()
    selected_method = selector.select(
        requested_method=method if method != "auto" else "auto",
        lower_bounds=lower,
        upper_bounds=upper,
    )

    if selected_method != "cmaes":
        # Fall through to standard optimization
        return None

    # Import CMAESOptimizer lazily to avoid evosax import at module load
    from nlsq.global_optimization.cmaes_optimizer import (
        CMAESOptimizer,
        _apply_data_mask,
    )

    # Use provided config or default
    config = cmaes_config if cmaes_config is not None else CMAESConfig()
    budget_label = None
    if cmaes_config is None and method == "auto":
        # The implicit route must not cost more than the fixed-2x BIPOP schedule it
        # replaced (explicit method='cmaes' stays unlimited, a documented opt-in).
        from dataclasses import replace as _replace

        config = _replace(
            config,
            max_total_evaluations=_auto_bipop_budget(config, len(lower)),
        )
        budget_label = (
            f"automatic budget max_total_evaluations={config.max_total_evaluations} "
            "for method='auto'; pass method='cmaes' for an unlimited search"
        )

    # Check if memory-based chunking is needed for large datasets
    n_data = len(np.asarray(ydata))  # points, not features of a (k, n) xdata
    n_params = len(lower)

    # Only auto-configure if chunk sizes not already set
    if config.data_chunk_size is None and config.population_batch_size is None:
        # Get available memory from MemoryBudget
        from nlsq.core.workflow import MemoryBudget

        budget = MemoryBudget.compute(
            n_points=n_data,
            n_params=n_params,
            memory_limit_gb=kwargs.get("memory_limit_gb"),
        )

        # Compute population size for memory estimation
        popsize = config.popsize
        if popsize is None:
            popsize = compute_default_popsize(n_params)
            # Double for cmaes-global preset (detected by max_generations == 200)
            if config.max_generations == 200 and config.restart_strategy == "bipop":
                popsize = popsize * 2

        # Auto-configure batch sizes based on available memory
        data_bytes, fixed_bytes = cmaes_resident_bytes(
            xdata, ydata, kwargs.get("sigma")
        )
        pop_batch, data_chunk = auto_configure_cmaes_memory(
            n_data=n_data,
            popsize=popsize,
            available_memory_gb=budget.available_gb,
            safety_factor=0.7,
            data_bytes=data_bytes,
            fixed_bytes=fixed_bytes,
        )

        # Create new config with computed chunk sizes if needed
        if pop_batch is not None or data_chunk is not None:
            from dataclasses import replace

            config = replace(
                config,
                population_batch_size=pop_batch,
                data_chunk_size=data_chunk,
            )

    # Create optimizer
    optimizer = CMAESOptimizer(config=config, budget_label=budget_label)

    # Extract sigma and absolute_sigma from kwargs if provided
    sigma = kwargs.get("sigma")
    absolute_sigma = kwargs.get("absolute_sigma", False)
    reject_all_zero_weight(sigma, kwargs.get("data_mask"))
    xdata, ydata, sigma = _preprocess_nonfinite_for_cmaes(
        xdata,
        ydata,
        sigma,
        kwargs.get("nan_policy"),
        kwargs.get("check_finite", True),
        kwargs,
    )
    # return_eval: rows after nan omission, before the user data_mask (as plain)
    xdata_in = xdata
    n_eval_rows = len(ydata)
    data_mask = kwargs.pop("data_mask", None)
    xdata, ydata, sigma = _apply_data_mask(xdata, ydata, sigma, data_mask)
    # Forward remaining optimizer kwargs (ftol, xtol, gtol, max_nfev, ...) to
    # the NLSQ refinement step -- mirrors _fit_global_cmaes, whose **kwargs
    # flow through to the same CMAESOptimizer.fit() call. Without this, this
    # entry point (method='cmaes') silently drops user tolerances/limits that
    # the fit(workflow='auto_global', method='cmaes') entry point honors.
    # p0/bounds/sigma/absolute_sigma are excluded since they're already
    # passed explicitly below -- forwarding them too would raise
    # "got multiple values for argument". timeit/return_eval/full_output are
    # excluded because they make the nested curve_fit() call inside NLSQ
    # refinement return a tuple instead of a CurveFitResult; _nlsq_refinement
    # then does result.x on that tuple, which raises AttributeError, which is
    # swallowed by a broad except and silently falls back to the unrefined
    # CMA-ES point estimate (pcov=inf) while still reporting success=True.
    _already_passed = (
        "p0",
        "bounds",
        "sigma",
        "absolute_sigma",
        "timeit",
        "return_eval",
        "full_output",
    )
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in _already_passed}
    if not compute_covariance:
        # Reaches the NLSQ refinement curve_fit call, the only place CMA-ES
        # computes a covariance.
        extra_kwargs["compute_covariance"] = False

    # Run CMA-ES optimization
    t_search = time.perf_counter()
    result_dict = optimizer.fit(
        f,
        xdata,
        ydata,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        **extra_kwargs,
    )

    # CMAESOptimizer returns {"popt": ..., "pcov": ..., ...}
    # pcov is None when the refinement skipped it (compute_covariance=False); the
    # unrefined fallbacks return an inf placeholder, which is dropped here.
    pcov = result_dict["pcov"] if compute_covariance else None
    if not compute_covariance and result_dict["pcov"] is not None:
        # refinement already logged it when it returned None; log only the dropped placeholder
        _logger.debug("covariance skipped", reason="compute_covariance=False")

    t_post = time.perf_counter()
    ctime = t_post - t_search  # search plus NLSQ refinement
    result = _cmaes_curve_fit_result(
        result_dict, f, xdata, ydata, sigma, absolute_sigma, pcov
    )

    # Add multi-start diagnostics placeholder
    result["multistart_diagnostics"] = {
        "n_starts_configured": 0,
        "bypassed": True,
        "method": "cmaes",
    }

    return _apply_output_flags(
        result,
        f,
        xdata_in,
        {**kwargs, "data_mask": data_mask, "n_eval_rows": n_eval_rows},
        ctime=ctime,
        post_time=time.perf_counter() - t_post,
    )


def _apply_output_flags(
    result: CurveFitResult,
    f: ModelFunction,
    xdata: ArrayLike,
    kwargs: dict[str, Any],
    ctime: float = 0.0,
    post_time: float = 0.0,
) -> Any:
    """Format a CurveFitResult as the SciPy-style tuple the caller asked for.

    Same shapes as ``CurveFit.curve_fit``: ``return_eval`` -> (popt, pcov, f(x));
    ``full_output`` -> (popt, pcov, infodict, mesg, ier); ``timeit`` ->
    (popt, pcov, result, post_time, ctime) as given by the caller (the timings of
    its own phases; ``result`` loses ``fun``/``jac`` like the ordinary timeit path,
    and ``all_times`` is ``{}`` when the route has no per-solver timings).
    ``infodict`` holds ``nfev`` and ``fvec`` (the refinement residuals, or None
    when no refinement ran); ``ier`` is the result's ``status`` (5 on failure),
    as in ``scipy.optimize.curve_fit``. Otherwise the result is returned unchanged.
    """
    popt, pcov = result.popt, result.pcov
    if kwargs.get("return_eval"):
        _record_attempt(result)
        # As CurveFit.curve_fit: nan_policy='omit' returns f over the retained rows,
        # a user data_mask returns f over all of the caller's rows.
        x_fit = result.get("xdata")
        mask = kwargs.get("data_mask")
        use_full = mask is not None or x_fit is None
        feval = np.asarray(f(jnp.asarray(xdata if use_full else x_fit), *popt))
        if (
            feval.ndim == 0
        ):  # a scalar model output is broadcast over the rows, as in CurveFit
            y_fit = result.get("ydata")
            # the rows f is evaluated over: those of xdata (already nan-omitted by
            # the caller) when a mask asked for all rows, else the fitted rows
            if use_full:
                # the route knows its observation count; the last axis of x is the
                # feature count for a points-first x
                n_rows = kwargs.get("n_eval_rows") or (
                    np.shape(xdata)[-1] if np.ndim(xdata) else 0
                )
            else:
                n_rows = np.size(y_fit) if y_fit is not None else 0
            if n_rows:
                feval = np.broadcast_to(feval, (n_rows,)).copy()
        return popt, pcov, feval
    if kwargs.get("full_output"):
        fun = result.get("fun")
        info = {
            "nfev": result.get("nfev"),
            "fvec": np.asarray(fun) if fun is not None else None,
        }
        status = result.get("status")
        status = 1 if status is None else int(status)
        # SciPy ier convention: the solver's status when converged, 5 otherwise.
        ier = status if result.get("success", False) and status > 0 else 5
        return popt, pcov, info, result.get("message", ""), ier
    if kwargs.get("timeit"):
        result.pop("jac", None)
        result.pop("fun", None)
        result.setdefault("all_times", {})
        return popt, pcov, result, float(post_time), float(ctime)
    return result


_DEFAULT_MAX_JACOBIAN_ELEMENTS = 10_000_000


def _route_options(
    compute_diagnostics: bool,
    diagnostics_level: DiagnosticLevel,
    diagnostics_config: DiagnosticsConfig | None,
    max_jacobian_elements_for_svd: int,
) -> dict[str, Any]:
    """Wrapper options that only the standard route consumes itself.

    The CMA-ES refinement and fallback attempts call ``curve_fit`` again, so
    these keywords reach them through kwargs (non-default values only).
    """
    opts: dict[str, Any] = {}
    if compute_diagnostics:
        opts.update(
            compute_diagnostics=True,
            diagnostics_level=diagnostics_level,
            diagnostics_config=diagnostics_config,
        )
    if max_jacobian_elements_for_svd != _DEFAULT_MAX_JACOBIAN_ELEMENTS:
        opts["max_jacobian_elements_for_svd"] = max_jacobian_elements_for_svd
    return opts


def _local_method_opts(
    method: str | None, route: str, bounds: Any = None
) -> dict[str, Any]:
    """The explicit local solver for a multistart/fallback attempt.

    ``'trf'``/``'lm'``/``'auto'`` go to every inner fit (``'auto'`` keeps the
    memory-based strategy selection in each). ``'lm'`` with finite bounds is
    rejected up front, as in a plain call (the fallback route would otherwise
    swallow the error and silently retry with TRF). ``'hybrid_streaming'``
    cannot be driven by those orchestrators; running a different solver
    instead would be a silent substitution.
    """
    if method == "lm" and bounds is not None:
        bounds = _bounds_as_tuple(bounds)
        if any(np.any(np.isfinite(np.asarray(b, dtype=float))) for b in bounds):
            raise ValueError("Method 'lm' doesn't work when there are bounds.")
    if method == "hybrid_streaming":
        raise ValueError(
            f"method='hybrid_streaming' cannot be combined with the {route} "
            "route; use method='trf' or drop the option."
        )
    return {"method": method} if method in ("trf", "lm", "auto") else {}


def _run_multistart_optimization(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: NDArray[np.floating[Any]] | None,
    n_starts: int,
    sampler: Literal["lhs", "sobol", "halton"],
    center_on_p0: bool,
    scale_factor: float,
    kwargs: dict[str, Any],
) -> CurveFitResult:
    """Run multi-start optimization with Latin Hypercube Sampling.

    Parameters
    ----------
    f : ModelFunction
        The model function.
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    p0 : NDArray or None
        Initial parameter guess.
    n_starts : int
        Number of starting points.
    sampler : {'lhs', 'sobol', 'halton'}
        Sampling strategy for generating starting points.
    center_on_p0 : bool
        Whether to center samples around p0.
    scale_factor : float
        Scale factor for exploration region.
    kwargs : dict
        Additional keyword arguments for each start's ``curve_fit``.
        ``max_jacobian_elements_for_svd`` (if present) configures each start's
        ``CurveFit`` instead.

    Returns
    -------
    result : CurveFitResult
        Optimization result from best starting point.
    """
    from nlsq.global_optimization import (
        GlobalOptimizationConfig,
        MultiStartOrchestrator,
    )

    # ``max_jacobian_elements_for_svd`` configures each start's CurveFit; it
    # is not a ``curve_fit`` keyword.
    kwargs = dict(kwargs)
    svd_limit = kwargs.pop(
        "max_jacobian_elements_for_svd", _DEFAULT_MAX_JACOBIAN_ELEMENTS
    )

    # flength/use_dynamic_sizing/cache_config are CurveFit constructor options too
    flength = kwargs.pop("flength", None)
    use_dynamic_sizing = kwargs.pop("use_dynamic_sizing", False)
    cache_config = kwargs.pop("cache_config", None)

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
    orchestrator = MultiStartOrchestrator(
        config=multistart_config,
        curve_fit_instance=CurveFit(
            flength=flength,
            use_dynamic_sizing=use_dynamic_sizing,
            cache_config=cache_config,
            max_jacobian_elements_for_svd=svd_limit,
        ),
    )

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


def _run_fallback_optimization(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    max_fallback_attempts: int,
    fallback_verbose: bool,
    multistart: bool,
    n_starts: int,
) -> Any:
    """Run optimization with automatic fallback strategies.

    Parameters
    ----------
    f : ModelFunction
        The model function.
    xdata : array_like
        Independent variable data.
    ydata : array_like
        Dependent variable data.
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.
    max_fallback_attempts : int
        Maximum number of fallback attempts.
    fallback_verbose : bool
        Whether to print detailed fallback information.
    multistart : bool
        Whether multistart was requested (for diagnostics).
    n_starts : int
        Number of starts (for diagnostics).

    Returns
    -------
    result : FallbackResult or tuple
        FallbackResult wrapping the CurveFitResult, or the plain tuple for
        timeit/return_eval/full_output.
    """
    from nlsq.stability.fallback import FallbackOrchestrator

    orchestrator = FallbackOrchestrator(
        max_attempts=max_fallback_attempts,
        verbose=fallback_verbose,
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

    # timeit/return_eval/full_output make curve_fit return a plain tuple: hand
    # it back exactly as the non-fallback call would (FallbackResult iteration
    # delegates to the tuple, so `in` / item assignment on it is unsafe).
    inner = result.result
    if not isinstance(inner, CurveFitResult):
        return inner

    # Add empty multi-start diagnostics for consistency
    if "multistart_diagnostics" not in inner:
        inner["multistart_diagnostics"] = {
            "n_starts_configured": n_starts if multistart else 0,
            "bypassed": True,
        }

    return result


def _prepare_optimizer_options(
    kwargs: dict[str, Any],
    max_jacobian_elements_for_svd: int,
    compute_diagnostics: bool,
    diagnostics_level: DiagnosticLevel,
    diagnostics_config: DiagnosticsConfig | None,
) -> tuple[int | None, bool, dict[str, Any] | None]:
    """Extract and prepare optimizer options from kwargs.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments (will be modified to remove extracted options).
    max_jacobian_elements_for_svd : int
        Maximum Jacobian elements for SVD computation.
    compute_diagnostics : bool
        Whether to compute diagnostics.
    diagnostics_level : DiagnosticLevel
        Level of diagnostics to compute. A non-default level overrides
        ``diagnostics_config.level``.
    diagnostics_config : DiagnosticsConfig or None
        Diagnostics configuration. Registered diagnostic plugins run
        automatically and appear in ``result.diagnostics.plugin_results``.

    Returns
    -------
    flength : int or None
        Fixed data length for JAX compilation.
    use_dynamic_sizing : bool
        Whether to use dynamic sizing.
    cache_config : dict or None
        Cache configuration.
    """
    flength = kwargs.pop("flength", None)
    use_dynamic_sizing = kwargs.pop("use_dynamic_sizing", False)
    cache_config = kwargs.pop("cache_config", None)

    # Pass diagnostics parameters through kwargs if requested
    if compute_diagnostics:
        kwargs["compute_diagnostics"] = True
        kwargs["diagnostics_level"] = diagnostics_level
        kwargs["diagnostics_config"] = diagnostics_config

    return flength, use_dynamic_sizing, cache_config


# =============================================================================
# Main curve_fit function (refactored for reduced complexity)
# =============================================================================


def _normalize_bounds_kwarg(kwargs: dict[str, Any]) -> None:
    if "bounds" in kwargs:
        kwargs["bounds"] = _bounds_as_tuple(kwargs["bounds"])


def _bounds_as_tuple(bounds: Any) -> Any:
    """scipy.optimize.Bounds -> (lb, ub); anything else is returned unchanged."""
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        # scipy stores scalar limits as length-1 arrays; keep them scalar so
        # they broadcast to every parameter like the tuple form does
        lb, ub = (
            np.asarray(b).item() if np.size(b) == 1 else b
            for b in (bounds.lb, bounds.ub)
        )
        return (lb, ub)
    return bounds


def curve_fit(
    f: ModelFunction,
    xdata: ArrayLike,
    ydata: ArrayLike,
    *args: Any,
    auto_bounds: bool = False,
    bounds_safety_factor: float = 10.0,
    stability: Literal["auto", "check", False] = False,
    rescale_data: bool = True,
    max_jacobian_elements_for_svd: int = _DEFAULT_MAX_JACOBIAN_ELEMENTS,
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
    # CMA-ES global optimization parameters
    method: Literal["auto", "cmaes", "multi-start", "trf", "lm", "hybrid_streaming"]
    | None = None,
    cmaes_config: CMAESConfig | None = None,
    # Diagnostics parameters (User Story 1)
    compute_diagnostics: bool = False,
    diagnostics_level: DiagnosticLevel = DiagnosticLevel.BASIC,
    diagnostics_config: DiagnosticsConfig | None = None,
    compute_covariance: bool = True,
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
        Inferred bounds are heuristics and can exclude the true solution: if the
        fit ends on an inferred bound, it is refit once without that bound and
        the better result is returned with an ``OptimizeWarning``. Not
        supported with ``method='lm'``. Default: False.
    bounds_safety_factor : float, optional
        Safety multiplier for automatic bounds (larger = more conservative).
        Only used when auto_bounds=True. Default: 10.0.
    stability : {'auto', 'check', False}, optional
        Control numerical stability checks and automatic fixes:

        - 'auto': Check for stability issues and warn about the ones that are
          not safely fixable; no data is modified
        - 'check': Check for stability issues and warn, but don't apply fixes
        - False: Skip stability checks entirely (default)

        When 'auto', detected issues are reported before optimization:

        - NaN/Inf values are not modified; they are handled by ``nan_policy``
          and ``check_finite``
        - Ill-conditioned data (condition number > 1e10), large data ranges
          (> 1e4), and parameter scale mismatches (ratio > 1e6) are *not*
          automatically rescaled/normalized: doing so would silently return
          fitted parameters in the wrong coordinate system for an arbitrary
          nonlinear model function, since the rescale can't be soundly
          inverted. A warning names these cases instead; pass ``x_scale``
          (e.g. ``x_scale='jac'``) to curve_fit for solver-internal scaling
          that doesn't have this problem, or rescale your data yourself.

        Default: False.
    rescale_data : bool, optional
        Kept for compatibility; no data is ever rescaled. When True and
        stability='auto', a warning is logged if the data would have qualified
        for rescaling (condition number > 1e10 or a range > 1e4). The p0
        scale-mismatch warning is emitted regardless. Default: True.
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
        Total number of starting points, including the user's ``p0`` when one
        is given (it is evaluated first, so ``n_starts - 1`` points are
        sampled). Only used when multistart=True or global_search=True.
        Default: 10.
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
    method : {'auto', 'cmaes', 'multi-start', 'trf', 'lm', 'hybrid_streaming'} | None, optional
        Optimization method to use:

        - 'auto': Automatically select based on parameter scale ratio.
          Uses CMA-ES for multi-scale problems (>1000x scale difference,
          finite ``bounds``) when evosax is installed; otherwise the
          memory-based strategy selection (standard TRF, chunked or
          streaming, by the size of the problem) applies. Request
          ``method='multi-start'`` explicitly for a multi-start fit.
        - 'cmaes': Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
          for gradient-free global optimization. Requires bounds.
          Best for multi-scale parameters spanning many orders of magnitude.
          Falls back to multi-start if evosax is not installed.
        - 'multi-start': Use multi-start optimization with Latin Hypercube
          Sampling for initial points.
        - 'trf': Use Trust Region Reflective (default behavior).
        - 'lm': alias for unbounded problems; runs the same trust-region
          solver as 'trf'.
        - 'hybrid_streaming': Adaptive hybrid streaming optimizer; ``xdata``
          is taken as ``(n_points, n_features)``.
        - None: Default behavior (TRF with optional multi-start if enabled).

        Default: None.
    cmaes_config : CMAESConfig | None, optional
        Configuration for CMA-ES optimization. If None, uses default config
        or preset specified by method parameter. See CMAESConfig for options
        including max_generations, restart_strategy, and popsize.
        Only used when method='cmaes' or method='auto' selects CMA-ES.
        Default: None.
    compute_covariance : bool, optional
        When False, ``pcov`` is returned as ``None`` (in the tuple, the
        ``CurveFitResult`` and the ``full_output``/``return_eval``/``timeit``
        tuples) and the final covariance is not computed, which skips an SVD
        of the Jacobian. ``CurveFitResult.confidence_intervals``,
        ``confidence_band`` and ``prediction_interval`` then raise
        ``AttributeError``; ``summary()`` omits the error column. Default: True.
        Per route: the default path, ``fallback=True``,
        ``compute_diagnostics=True``, ``auto_bounds=True``, ``multistart=True``,
        ``global_search=True``, ``method='multi-start'``, ``method='cmaes'``
        (the flag reaches its NLSQ refinement fit) and ``method='auto'`` when
        it selects the in-memory strategy honour the flag: no covariance is
        computed and no covariance warning is emitted.
        ``method='hybrid_streaming'`` and ``method='auto'`` when it selects the
        streaming or chunked strategy raise ``ValueError``, because those
        engines compute the covariance internally (the chunked GLS combiner
        needs the per-chunk covariances).
    compute_diagnostics : bool, optional
        If True, compute a fit-quality report and attach it to the result as
        ``result.diagnostics``. Default: False.
    diagnostics_level : DiagnosticLevel, optional
        Level of diagnostics to compute. A non-default level overrides
        ``diagnostics_config.level``. Default: ``DiagnosticLevel.BASIC``.
    diagnostics_config : DiagnosticsConfig or None, optional
        Diagnostics configuration. Registered diagnostic plugins run
        automatically and appear in ``result.diagnostics.plugin_results``.
    *args, **kwargs
        Additional arguments passed to CurveFit.curve_fit method.

    Returns
    -------
    popt : ndarray
        Optimal values for the parameters.
    pcov : ndarray or None
        The estimated covariance of popt; ``None`` when
        ``compute_covariance=False``.

    When fallback=True, the returned object also contains:

    - fallback_strategy_used : str or None
        Name of the fallback strategy that succeeded, or None if original succeeded
    - fallback_attempts : int
        Number of optimization attempts before success

    Fallback metadata is not available when ``full_output``, ``return_eval``
    or ``timeit`` is set: those return the plain tuple.

    With ``timeit=True`` the tuple is ``(popt, pcov, result, post_time, ctime)``.
    On the default path ``ctime`` is the host-to-device conversion time and
    ``post_time`` the covariance time. On the CMA-ES route and the
    ``method='auto'`` chunked route ``ctime`` is the whole search/fit time and
    ``post_time`` the result-formatting time; those routes have no per-solver
    timings, so ``result['all_times']`` is ``{}``.

    When multistart=True or global_search=True, the returned object contains:

    - multistart_diagnostics : dict
        Dictionary with multi-start exploration details including n_starts,
        best_loss, all_losses, exploration time, etc.

    Raises
    ------
    ValueError
        For an unknown ``method``, ``stability`` other than
        ``False``/``None``/``'auto'``/``'check'``, an unknown ``tr_solver``,
        a non-positive or non-finite ``f_scale``, a NaN ``ftol``/``xtol``/
        ``gtol``, a boolean ``max_nfev``, a negative ``n_starts`` with
        multi-start, an invalid ``nan_policy``, or invalid data, bounds or
        ``p0``.
    TypeError
        If ``absolute_sigma`` is not a bool.

    Notes
    -----
    This function creates a CurveFit instance internally and calls its curve_fit method.

    Unlike ``scipy.optimize.curve_fit``, exhausting ``max_nfev``/``maxfev`` does
    not raise ``RuntimeError``: NLSQ emits an ``OptimizeWarning`` and returns the
    best-effort result with ``success=False``.

    Repeated fits reuse a model's compiled residual and Jacobian (autodiff,
    no ``jac``) when the model traces to an identical computation: every call
    traces ``f`` abstractly, and the traced program's exact structure,
    including every value ``f`` read at trace time (globals, closures,
    attributes, files, ...) and the bytes of its constants, is the cache
    key, so changed model state is a cache miss. Models with side effects or
    callbacks (``jax.debug.print``, ``pure_callback``), models with a custom
    derivative rule (``custom_jvp``/``custom_vjp``, e.g. ``jax.nn.relu``),
    user-defined JAX primitives, or a user ``jac`` are not cached.
    ``nlsq.clear_cache()`` drops the cache; ``NLSQ_DISABLE_WRAPPER_CACHE=1``
    disables it.

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
    # Normalize positional args to keyword args immediately. Downstream code
    # (stability auto-fixes) mutates kwargs["p0"] and previously sliced the
    # raw `args` tuple to drop the old p0 -- but that silently reindexed any
    # positional sigma/absolute_sigma/... that followed it, corrupting the
    # positional mapping when the fixed args were later forwarded to
    # CurveFit.curve_fit / FallbackOrchestrator. Binding everything to
    # kwargs up front (matching CurveFit.curve_fit's own positional order)
    # makes that reindexing impossible.
    _positional_names = (
        "p0",
        "sigma",
        "absolute_sigma",
        "check_finite",
        "bounds",
        "method",
        "solver",
        "batch_size",
        "jac",
        "data_mask",
        "timeit",
        "return_eval",
        "full_output",
        "callback",
        "compute_diagnostics",
        "diagnostics_level",
        "diagnostics_config",
    )
    if len(args) > len(_positional_names):
        raise TypeError(
            f"curve_fit() takes at most {len(_positional_names)} positional "
            f"arguments after (f, xdata, ydata) but {len(args)} were given",
        )
    for _name, _value in zip(_positional_names, args, strict=False):
        kwargs.setdefault(_name, _value)
    args = ()
    # A positional ``method`` was bound into kwargs above; the routing below
    # reads the named parameter.
    method = kwargs.pop("method", None) if method is None else method

    # scipy.optimize.Bounds -> (lb, ub) once, so every route (TRF, CMA-ES,
    # multistart, streaming) sees the tuple form.
    _normalize_bounds_kwarg(kwargs)
    reject_all_zero_weight(
        kwargs.get("sigma", args[1] if len(args) > 1 else None),
        kwargs.get("data_mask"),
    )

    _validate_wrapper_options(
        method if method is not None else kwargs.get("method"),
        stability,
        kwargs.get("absolute_sigma", False),
        kwargs,
    )
    if (
        (multistart or method == "multi-start")
        and n_starts is not None
        and n_starts < 0
    ):
        raise ValueError(f"n_starts must be >= 0, got {n_starts}")

    # Inferred bounds: delegate to a driver that re-invokes this function with
    # auto_bounds off, so the clipping guard covers every route.
    if auto_bounds:
        _inner = {
            "stability": stability,
            "rescale_data": rescale_data,
            "max_jacobian_elements_for_svd": max_jacobian_elements_for_svd,
            "fallback": fallback,
            "max_fallback_attempts": max_fallback_attempts,
            "fallback_verbose": fallback_verbose,
            "multistart": multistart,
            "n_starts": n_starts,
            "global_search": global_search,
            "sampler": sampler,
            "center_on_p0": center_on_p0,
            "scale_factor": scale_factor,
            "method": method,
            "cmaes_config": cmaes_config,
            "compute_diagnostics": compute_diagnostics,
            "diagnostics_level": diagnostics_level,
            "diagnostics_config": diagnostics_config,
            "compute_covariance": compute_covariance,
        }
        return _curve_fit_with_auto_bounds(
            lambda **kw: curve_fit(f, xdata, ydata, **{**_inner, **kw}),
            f,
            xdata,
            ydata,
            bounds_safety_factor,
            method,
            kwargs,
        )

    # Handle global_search shorthand
    if global_search:
        multistart = True
        n_starts = 20

    # Extract p0 for use in preprocessing steps
    p0 = _extract_p0_from_args(args, kwargs)

    p0 = _prepare_wrapper_p0(f, xdata, ydata, p0, stability, multistart, method, kwargs)

    # Verbose memory budget diagnostics (verbose >= 2)
    verbose = kwargs.get("verbose", 1)
    if verbose >= 2:
        _log_memory_budget_diagnostics(xdata, ydata, p0, kwargs.get("bounds"))

    # Handle numerical stability checks and fixes
    if stability:
        xdata, ydata, args = _apply_stability_checks(
            xdata,
            ydata,
            p0,
            f,
            stability,
            rescale_data,
            args,
            kwargs,
        )
        # Re-extract p0 in case it was updated
        p0 = _extract_p0_from_args(args, kwargs)

    route_opts = _route_options(
        compute_diagnostics,
        diagnostics_level,
        diagnostics_config,
        max_jacobian_elements_for_svd,
    )

    # Handle CMA-ES global optimization
    if method in ("cmaes", "auto"):
        bounds = kwargs.get("bounds")
        if bounds is not None:
            result = _run_cmaes_optimization(
                f,
                xdata,
                ydata,
                p0,
                bounds,
                method,
                cmaes_config,
                {**kwargs, **route_opts},
                compute_covariance,
            )
            if result is not None:
                return result
        # method='auto' without bounds, or when the selector declines CMA-ES,
        # continues on the memory-based auto path below; method='cmaes' becomes
        # a multi-start fit (evosax not installed).

    # Multistart, fallback and the standard path all end in CurveFit.curve_fit,
    # which owns the keyword; CMA-ES above took it as an explicit argument.
    kwargs["compute_covariance"] = compute_covariance

    # Handle multi-start optimization. method="multi-start" is a documented
    # Literal value on this function's own signature but was only ever
    # routed here via the separate `multistart` bool flag -- passing
    # method="multi-start" alone fell through to standard TRF, which rejects
    # it with an unrelated "method must be 'trf' or 'lm'" error.
    run_multistart = (
        (multistart or method in ("multi-start", "cmaes"))
        and n_starts is not None
        and n_starts > 0
    )
    if run_multistart and not fallback:
        return _run_multistart_optimization(
            f,
            xdata,
            ydata,
            p0,
            n_starts,
            sampler,
            center_on_p0,
            scale_factor,
            {
                **kwargs,
                **route_opts,
                **_local_method_opts(method, "multistart", kwargs.get("bounds")),
            },
        )

    # Use fallback orchestrator if requested. Combined with multi-start, every
    # fallback attempt is itself a multi-start fit (documented "multi-start +
    # fallback" combination), so recovery is not bypassed.
    if fallback:
        return _run_fallback_optimization(
            f,
            xdata,
            ydata,
            args,
            {
                **kwargs,
                **route_opts,
                **_local_method_opts(method, "fallback", kwargs.get("bounds")),
                **(
                    {
                        "multistart": True,
                        "n_starts": n_starts,
                        "sampler": sampler,
                        "center_on_p0": center_on_p0,
                        "scale_factor": scale_factor,
                    }
                    if run_multistart
                    else {}
                ),
            },
            max_fallback_attempts,
            fallback_verbose,
            multistart,
            n_starts,
        )

    # Standard path without fallback or multi-start
    flength, use_dynamic_sizing, cache_config = _prepare_optimizer_options(
        kwargs,
        max_jacobian_elements_for_svd,
        compute_diagnostics,
        diagnostics_level,
        diagnostics_config,
    )

    # Create CurveFit instance with appropriate parameters
    jcf = CurveFit(
        flength=flength,
        use_dynamic_sizing=use_dynamic_sizing,
        cache_config=cache_config,
        max_jacobian_elements_for_svd=max_jacobian_elements_for_svd,
    )

    # Pass method through to CurveFit if specified (e.g., hybrid_streaming).
    # 'cmaes'/'multi-start' are routed above (n_starts=0 bypasses to plain TRF).
    if method is not None and method not in ("cmaes", "multi-start"):
        kwargs["method"] = method
    result = jcf.curve_fit(f, xdata, ydata, *args, **kwargs)

    # timeit=True, return_eval=True, and full_output=True each make
    # CurveFit.curve_fit return a plain tuple instead of a CurveFitResult
    # (SciPy-style special output modes) -- only a CurveFitResult supports
    # the dict-style assignment below.
    if isinstance(result, CurveFitResult):
        # Add empty multi-start diagnostics for consistency
        result["multistart_diagnostics"] = {
            "n_starts_configured": n_starts if multistart else 0,
            "bypassed": True,  # Bypassed because n_starts was 0 or multistart was False
        }

    # Return enhanced result object that supports both tuple unpacking
    # (popt, pcov = curve_fit(...)) and direct use (result = curve_fit(...))
    return result


# Module-level JIT singletons bound by CurveFit. Per-instance `@jit` closures
# were new function objects on every curve_fit() call (which builds a new
# CurveFit), so JAX re-traced and re-compiled them each time.


@jit
def _sigma_transform1d(
    sigma: jnp.ndarray,
    data_mask: jnp.ndarray,
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
def _sigma_transform2d(
    sigma: jnp.ndarray,
    data_mask: jnp.ndarray,
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
    # Decouple masked/padded rows (zero their rows and columns, unit
    # diagonal) so they neither correlate with nor leak into the
    # observed block of the Cholesky factor.
    mask = jnp.asarray(data_mask, dtype=bool)
    keep = mask[:, None] & mask[None, :]
    sigma = jnp.where(keep, sigma, jnp.eye(sigma.shape[0], dtype=sigma.dtype))
    transform = jax_cholesky(sigma, lower=True)
    return transform


def _covariance_svd_body(jac) -> tuple[jnp.ndarray, jnp.ndarray]:
    R = jnp.linalg.qr(jac, mode="r")
    _, s, VT = jax_svd(R, full_matrices=False)
    return s, VT


_covariance_svd_jit = jit(_covariance_svd_body)


def _covariance_svd(jac) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(s, VT)`` of ``jac``, on the CPU device when the A3 policy says so.

    SVD of the QR R factor; same s and VT up to row signs.
    Results stay where they were computed: ``pcov_from_jacobian`` converts to NumPy.
    """
    if device_policy.use_host_svd_for(jac, device_policy.HOST_COV_SVD_MAX_ELEMENTS):
        jac = jax.device_put(jac, device_policy.host_device())
    return _covariance_svd_jit(jac)


def _resolve_diagnostics_config(
    level: DiagnosticLevel, config: DiagnosticsConfig | None
) -> DiagnosticsConfig:
    """Combine ``diagnostics_level`` and ``diagnostics_config``.

    A non-default ``level`` overrides ``config.level`` (the default BASIC cannot
    be told apart from "not given", so the config's level wins then).
    """
    if config is None:
        return DiagnosticsConfig(level=level, verbose=False, emit_warnings=True)
    level = DiagnosticsConfig(level=level).level  # normalizes str levels
    if level in (DiagnosticLevel.BASIC, config.level):
        return config
    return dataclasses.replace(config, level=level)


def _run_diagnostic_plugins(logger: Any, jacobian, popt, residuals, **context: Any):
    """Run registered diagnostic plugins; never let them discard the fit's report."""
    try:
        from nlsq.diagnostics.plugin import PluginRegistry, run_plugins

        if not PluginRegistry.all():
            return None
        return run_plugins(jacobian, np.asarray(popt), np.asarray(residuals), **context)
    except Exception as e:
        logger.warning("Diagnostic plugins failed", error=str(e))
        return None


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
            Clip the model output to ``[-1e10, 1e10]`` whenever ``max|f|``
            exceeds ``1e10`` during the fit. This changes the objective (and can
            zero the Jacobian when data are larger than ``1e10``); an
            ``OptimizeWarning`` is emitted if ``max|f|`` at the solution exceeds
            ``1e10``. Adds ~30% overhead, so it is separate from other stability
            features.

        cache_config : dict, optional
            Configuration for the unified JIT compilation cache.
            Supported keys: ``'maxsize'`` (int, default 128) — maximum
            number of compiled functions to cache;
            ``'enable_stats'`` (bool, default True) — track cache
            statistics (hits, misses, wrapper-creation time);
            ``'disk_cache_enabled'`` (bool, default False) — not
            implemented; True warns and is ignored.
            If None, uses global cache with default settings.
            This cache is only used by ``cached_jit`` / ``get_or_compile``
            callers; curve_fit's own compilation does not go through it, so
            ``cache_config`` does not change fitting or compile behaviour, and
            ``result["cache_stats"]`` reports this cache only (its
            ``compile_time_ms`` / ``total_compilations`` count ``jax.jit``
            wrapper construction, not XLA compilation).

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
        self.cache_config = cache_config
        self.logger = get_logger("curve_fit")
        self.create_sigma_transform_funcs()
        self.create_covariance_svd()
        self.ls = LeastSquares(
            enable_stability=enable_stability,
            max_jacobian_elements_for_svd=max_jacobian_elements_for_svd,
        )

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
            # Lazy import to reduce module dependencies
            from nlsq.stability.guard import NumericalStabilityGuard

            self.stability_guard = NumericalStabilityGuard(
                max_jacobian_elements_for_svd=max_jacobian_elements_for_svd,
            )
            # Use fast validation mode by default for performance
            self.validator = InputValidator(fast_mode=True)
            self.memory_manager = get_memory_manager()

        if enable_recovery:
            # Lazy imports to reduce module dependencies
            from nlsq.stability.recovery import OptimizationRecovery
            from nlsq.utils.diagnostics import OptimizationDiagnostics

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
        """Bind the JIT-compiled sigma transform functions.

        `sigma_transform1d` and `sigma_transform2d` compute the sigma transform
        for 1D and 2D sigma, respectively.
        """
        self.sigma_transform1d = _sigma_transform1d
        self.sigma_transform2d = _sigma_transform2d

    def create_covariance_svd(self) -> None:
        """Bind the JIT-compiled SVD function for covariance computation."""
        self.covariance_svd = _covariance_svd

    def _select_tr_solver(
        self,
        solver: str,
        m: int,
        n: int,
        batch_size: int | None = None,
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
            # Large problems
            return "lsmr"  # Use iterative LSMR solver
        if solver == "svd":
            return "exact"  # SVD-based exact solver
        if solver == "cg":
            return "lsmr"  # LSMR is the closest to CG in current implementation
        if solver == "lsqr":
            return "lsmr"  # Direct mapping
        if solver == "minibatch":
            # For minibatch, we'll use lsmr but need to handle batching separately
            # This is a placeholder - full minibatch implementation would require
            # more substantial changes to the optimization loop
            self.logger.warning(
                "Minibatch solver not fully implemented yet. Using LSMR solver.",
                requested_batch_size=batch_size,
            )
            return "lsmr"
        return None  # Use default

    def pad_fit_data(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xdims: int,
        len_diff: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad fit data to match the fixed input data length.

        Appends ``len_diff`` entries of ``EPS`` to ``xdata`` (along the second
        axis if multidimensional) and to ``ydata`` so JAX does not retrace the
        JITted functions. See :func:`nlsq.core.problem.pad_fit_data`.

        Parameters
        ----------
        xdata : np.ndarray
            The independent variables of the data.
        ydata : np.ndarray
            The dependent variables of the data.
        xdims : int
            The number of dimensions in the ``xdata`` array.
        len_diff : int
            Number of padding entries to append.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The padded ``xdata`` and ``ydata`` arrays.
        """
        return _pad_fit_data(xdata, ydata, xdims, len_diff)

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

        # Convert scalar/list sigma to float64 before shape checks (SciPy accepts
        # scalar sigma as sigma * ones(m)). A float32 ndarray must be widened too:
        # 1/sigma in float32 overflows for sigma below ~1e-38.
        sigma = np.asarray(sigma, dtype=float)

        # Negative len_diff means the data exceeds flength and was not padded.
        len_diff = max(len_diff, 0)
        ysize = ydata.size - len_diff

        if sigma.ndim == 0:
            # Scalar sigma: expand to 1-D array matching data length
            sigma = np.full(ysize, float(sigma))

        # if 1-D, sigma are errors, define transform = 1/sigma
        if sigma.shape == (ysize,):
            if len_diff > 0:
                sigma = np.concatenate([sigma, np.ones([len_diff])])
            return self.sigma_transform1d(sigma, data_mask)

        # if 2-D, sigma is the covariance matrix,
        # define transform = L such that L L^T = C
        if sigma.shape == (ysize, ysize):
            try:
                if len_diff > 0:
                    sigma_padded = np.identity(m + len_diff)
                    sigma_padded[:m, :m] = sigma
                    sigma = sigma_padded
                # scipy.linalg.cholesky requires lower=True to return L L^T = A
                transform = self.sigma_transform2d(sigma, data_mask)
                # jax_cholesky returns NaN instead of raising on a non-PD
                # input; one scalar host check at setup (not in the hot loop).
                if not bool(jnp.all(jnp.isfinite(transform))):
                    raise np.linalg.LinAlgError("Cholesky produced non-finite values")
                return transform
            except (np.linalg.LinAlgError, ValueError) as e:
                # Check eigenvalues to provide more informative error
                min_eig = None
                try:
                    min_eig = np.min(np.linalg.eigvalsh(sigma[:ysize, :ysize]))
                except Exception as eigenvalue_error:
                    # If eigenvalue check fails, provide generic error (log for debugging)
                    self.logger.debug(
                        f"Eigenvalue check failed (non-critical): {eigenvalue_error}",
                    )
                if min_eig is not None and min_eig <= 0:
                    raise ValueError(
                        f"Covariance matrix `sigma` is not positive definite. "
                        f"Minimum eigenvalue: {min_eig:.6e}. "
                        "All eigenvalues must be positive.",
                    ) from e
                raise ValueError(
                    "Failed to compute Cholesky decomposition of `sigma`. "
                    "The covariance matrix must be symmetric and positive definite.",
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
        # pcov_from_jacobian's warnings are attributed to the first frame outside nlsq/.
        return pcov_from_jacobian(
            res.jac,
            res.cost,  # half sum of squares; the helper applies the factor 2
            ysize,
            p0.size,
            absolute_sigma,
            self.covariance_svd,
            self.logger,
        )

    def _final_covariance(
        self, res, ysize: int, p0: np.ndarray, absolute_sigma: bool, compute: bool
    ) -> tuple[np.ndarray | None, bool]:
        """``_compute_covariance``, or ``(None, False)`` when ``compute`` is False."""
        if compute:
            return self._compute_covariance(res, ysize, p0, absolute_sigma)
        self.logger.debug("covariance skipped", reason="compute_covariance=False")
        return None, False

    @staticmethod
    def _drop_unobserved_rows(res, xdata, ydata, data_mask):
        """Drop padded (flength) and caller-masked rows from stored data/residuals.

        These rows are not observations, so they must stay out of everything
        derived from the stored data (r_squared, rmse, residuals, fvec). The
        covariance was already computed with the mask applied. Slices only when
        needed, so the default (all-valid) path is untouched and copy-free.
        Mutates ``res["fun"]``/``res["jac"]``; returns the sliced (xdata, ydata).
        """
        valid_rows = np.asarray(data_mask, dtype=bool)
        if valid_rows.all():
            return xdata, ydata
        for key in ("fun", "jac"):
            if res.get(key) is not None:
                res[key] = res[key][valid_rows]
        return np.asarray(xdata)[..., valid_rows], np.asarray(ydata)[
            valid_rows
        ]  # (M,) or (k, M)

    @staticmethod
    def _drop_unobserved_sigma(sigma, data_mask):
        """Apply ``data_mask`` to a full-length 1-D or covariance-matrix ``sigma``.

        Keeps the stored sigma aligned with the masked residuals so
        ``prediction_interval`` can weight by it. A sigma of any other shape is
        returned unchanged (the interval then uses the pooled variance).
        """
        if sigma is None:
            return sigma
        valid_rows = np.asarray(data_mask, dtype=bool)
        if valid_rows.all():
            return sigma
        sig = np.asarray(sigma)
        n = sig.shape[0] if sig.ndim in (1, 2) else -1
        if sig.ndim == 2 and sig.shape != (n, n):
            return sigma
        if 0 < n <= valid_rows.size:
            # flength padding is appended after the caller's rows
            keep = valid_rows[:n]
            return sig[keep] if sig.ndim == 1 else sig[np.ix_(keep, keep)]
        return sigma

    @staticmethod
    def _warn_unsupported_args(method: str, **non_default: bool) -> None:
        """Warn (instead of silently dropping) for arguments a path ignores."""
        ignored = sorted(name for name, used in non_default.items() if used)
        if ignored:
            warnings.warn(
                f"{method} does not support {ignored}; the argument(s) are ignored.",
                UserWarning,
                stacklevel=3,
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
        data_mask: np.ndarray | None = None,
        callback: Callable | None = None,
        jac: Callable | None = None,
        solver: str = "auto",
        timeit: bool = False,
        return_eval: bool = False,
        full_output: bool = False,
        compute_covariance: bool = True,
        compute_diagnostics: bool = False,
        **kwargs,
    ) -> CurveFitResult | tuple:
        """Handle curve fitting with hybrid_streaming method.

        This method delegates to AdaptiveHybridStreamingOptimizer for
        large-scale optimization with parameter normalization, L-BFGS warmup,
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
            Not supported by this method; passing one raises NotImplementedError.
        absolute_sigma : bool, default=False
            Whether sigma is absolute or relative
        check_finite : bool, default=True
            Check for finite input values
        bounds : tuple, optional
            Parameter bounds as (lower, upper)
        data_mask : array_like, optional
            Not supported by this method; passing one raises NotImplementedError
            (AdaptiveHybridStreamingOptimizer has no per-point masking).
        callback : callable, optional
            Progress callback function
        jac, solver, timeit : optional
            Not supported by this method; a non-default value emits a
            ``UserWarning`` and is ignored.
        return_eval : bool, default=False
            Return ``(popt, pcov, f(xdata, *popt))``.
        full_output : bool, default=False
            Return the SciPy-style ``(popt, pcov, infodict, mesg, ier)`` tuple.
        compute_covariance : bool, default=True
            ``False`` raises ``ValueError``: the hybrid engine always computes
            the covariance.
        compute_diagnostics : bool, default=False
            Not supported by this method; ``True`` emits a ``UserWarning`` and
            is ignored.
        **kwargs
            Additional parameters (verbose, config overrides, nan_policy)

        Returns
        -------
        result : CurveFitResult
            Optimization result compatible with scipy.optimize.curve_fit
        """
        from nlsq.streaming.hybrid_config import HybridStreamingConfig
        from nlsq.streaming.strategy import build_large_strategy

        reject_all_zero_weight(sigma, data_mask)
        if not compute_covariance:
            raise ValueError(
                "compute_covariance=False is not supported by "
                "method='hybrid_streaming' (the engine always computes the "
                "covariance). Use method='trf' to skip it."
            )

        self._warn_unsupported_args(
            "method='hybrid_streaming'",
            jac=jac is not None,
            solver=solver != "auto",
            timeit=timeit,
            compute_diagnostics=bool(compute_diagnostics),
        )

        # data_mask has no way to reach AdaptiveHybridStreamingOptimizer.fit()
        # (no per-point masking support) -- fail loudly instead of silently
        # ignoring it, matching fit()'s own sigma NotImplementedError below.
        if data_mask is not None:
            raise NotImplementedError(
                "method='hybrid_streaming' does not support the 'data_mask' "
                "parameter yet. Use a different method (e.g. 'trf') if you "
                "need to exclude specific data points, or omit data_mask.",
            )

        # nan_policy is popped here, before the leftover-keyword TypeError check
        # below. prepare_problem owns NaN handling, p0 and bounds; flength=None
        # means no padding. An explicit method is returned unchanged by method
        # selection (it only touches kwargs when method is None), and kwargs is
        # not passed, so nothing is injected into the hybrid's kwargs.
        prepared = prepare_problem(
            f,
            xdata,
            ydata,
            p0=p0,
            bounds=bounds,
            sigma=sigma,
            check_finite=check_finite,
            nan_policy=kwargs.pop("nan_policy", "raise"),
            method="hybrid_streaming",
            # Hybrid takes multi-dimensional x as (n_points, n_features).
            ctx=PrepareContext(
                flength=None, use_dynamic_sizing=False, x_points_axis_last=False
            ),
        )
        xdata, ydata, sigma = prepared.x, prepared.y, prepared.sigma
        n, p0 = prepared.n, prepared.p0
        lb, ub = prepared.lb, prepared.ub
        del prepared  # frees the m-byte mask for the duration of the fit
        bounds_tuple = (
            (lb, ub)
            if not (np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)))
            else None
        )

        # Extract verbosity from kwargs
        verbose = kwargs.pop("verbose", 0)

        # Handle config: either a complete HybridStreamingConfig object or individual overrides
        config = kwargs.pop("config", None)
        if config is None:
            # Create configuration from individual kwargs
            config_overrides = {}
            for key in list(kwargs.keys()):
                if hasattr(HybridStreamingConfig, key):
                    config_overrides[key] = kwargs.pop(key)

            config = (
                HybridStreamingConfig(**config_overrides)
                if config_overrides
                else HybridStreamingConfig()
            )
        elif not isinstance(config, HybridStreamingConfig):
            raise TypeError(
                f"config must be a HybridStreamingConfig, got {type(config).__name__}",
            )

        # Applies whether config came from individual kwargs or was passed
        # directly: a leftover kwarg in the config= case is just as much an
        # error (e.g. a typo'd option) as one that didn't match a
        # HybridStreamingConfig field above.
        if kwargs:
            raise TypeError(
                "Unrecognized keyword argument(s) for method="
                f"'hybrid_streaming': {sorted(kwargs)}. These are not "
                "HybridStreamingConfig fields.",
            )

        # Build the hybrid engine (AdaptiveHybridStreamingOptimizer)
        large_strategy = build_large_strategy("hybrid", config)

        # Run optimization
        result_dict = large_strategy.fit(
            f,
            xdata,
            ydata,
            p0,
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
        result["x_points_first"] = True  # (n_points, k): square xdata stays unambiguous
        _store_sigma(result, sigma, absolute_sigma)

        # timeit is warned about and ignored above (the engine has no timings).
        return _apply_output_flags(
            result,
            f,
            xdata,
            {"return_eval": return_eval, "full_output": full_output},
        )

    def _curve_fit_auto_memory(
        self,
        f: Callable,
        xdata: np.ndarray | tuple[np.ndarray],
        ydata: np.ndarray,
        p0: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        absolute_sigma: bool = False,
        check_finite: bool = True,
        bounds: tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
        solver: str = "auto",
        batch_size: int | None = None,
        jac: Callable | None = None,
        data_mask: np.ndarray | None = None,
        timeit: bool = False,
        return_eval: bool = False,
        callback: Callable | None = None,
        compute_diagnostics: bool = False,
        diagnostics_level: DiagnosticLevel = DiagnosticLevel.BASIC,
        diagnostics_config: DiagnosticsConfig | None = None,
        full_output: bool = False,
        compute_covariance: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray] | CurveFitResult:
        """Handle curve fitting with automatic memory-based strategy selection.

        Uses MemoryBudgetSelector to determine the optimal strategy based on
        available system memory and dataset characteristics. Routes to:
        - "streaming": AdaptiveHybridStreamingOptimizer for data exceeding memory
        - "chunked": LargeDatasetFitter for peak memory exceeding threshold
        - "standard": Standard curve_fit() when everything fits in memory

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
        solver : str, default="auto"
            Solver type for standard curve_fit
        batch_size : int, optional
            Batch size for minibatch solver
        jac : callable, optional
            Jacobian function
        data_mask : array_like, optional
            Mask for data points
        timeit : bool, default=False
            Return timing information
        return_eval : bool, default=False
            Return function evaluation
        callback : callable, optional
            Progress callback function
        compute_diagnostics : bool, default=False
            Compute fit diagnostics
        diagnostics_level : DiagnosticLevel
            Level of diagnostics to compute. A non-default level overrides
            ``diagnostics_config.level``.
        diagnostics_config : DiagnosticsConfig, optional
            Configuration for diagnostics
        full_output : bool, default=False
            Return the SciPy-style ``(popt, pcov, infodict, mesg, ier)`` tuple,
            as in ``curve_fit``.
        compute_covariance : bool, default=True
            When False, ``pcov`` is ``None``. Only the standard strategy
            honours it; the streaming and chunked strategies raise
            ``ValueError`` (their engines always compute the covariance).
        **kwargs
            Additional parameters

        Returns
        -------
        result : tuple or CurveFitResult
            Optimization result (popt, pcov) or CurveFitResult object
        """
        from nlsq.core.workflow import MemoryBudgetSelector
        from nlsq.streaming.strategy import scipy_layout_to_points_first

        reject_complex_inputs(xdata, ydata, sigma)  # as prepare_problem
        sigma = coerce_real_sigma(sigma)
        reject_all_zero_weight(sigma, data_mask)

        # Convert ydata to array for size determination
        ydata_arr = np.asarray(ydata, float)
        n_points = len(ydata_arr)

        # Determine parameter count.
        if p0 is not None and not _is_auto_p0(p0):
            n_params = len(np.atleast_1d(p0))
        else:
            # Estimate parameter count by inspecting function signature
            n_params = max(1, _n_params_from_signature(f))

        # Get memory_limit_gb from kwargs if provided
        memory_limit_gb = kwargs.pop("memory_limit_gb", None)

        # Get verbosity for logging
        verbose = kwargs.get("verbose", 1)

        # Select strategy based on memory budget
        selector = MemoryBudgetSelector()
        strategy, config = selector.select(
            n_points=n_points,
            n_params=n_params,
            n_features=_infer_n_features(np.asarray(xdata)),
            memory_limit_gb=memory_limit_gb,
            verbose=(verbose >= 2),
            bounded=_is_bounded(bounds, n_params),
        )

        if verbose >= 1:
            self.logger.info(
                "Memory-based auto selection",
                strategy=strategy,
                n_points=n_points,
                n_params=n_params,
            )

        # Route to appropriate strategy
        if strategy == "streaming":
            # Delegate to hybrid streaming optimizer. Pass the memory-derived
            # config through (as the "chunked" branch below already does) --
            # without it, _curve_fit_hybrid_streaming falls back to its own
            # default HybridStreamingConfig() and silently ignores the
            # memory-budget-computed batch/chunk size that was just selected.
            # This method takes scipy's (k, M) x; the hybrid method is
            # points-first, so adapt the model and x on the way in.
            f_engine, x_engine = scipy_layout_to_points_first(f, xdata)
            # An explicit config makes _curve_fit_hybrid_streaming reject every
            # leftover kwarg, but the standard/chunked routes of this same
            # method="auto" call accept the ordinary tolerances: map gtol, warn
            # for the ones the hybrid engine has no equivalent for.
            if "gtol" in kwargs:
                gtol = kwargs.pop("gtol")
                if gtol is None or gtol <= 0:
                    # "test disabled" on the standard route. The hybrid engine
                    # uses gauss_newton_tol for its gradient AND cost-change
                    # stops, so a tiny value would leave it no stop and every
                    # fit would report success=False: keep the config default.
                    self._warn_unsupported_args(
                        "method='auto' (streaming strategy)", gtol=True
                    )
                else:
                    config = dataclasses.replace(config, gauss_newton_tol=gtol)
            self._warn_unsupported_args(
                "method='auto' (streaming strategy)",
                compute_diagnostics=bool(compute_diagnostics),
                **{
                    k: kwargs.pop(k, None) is not None
                    for k in ("ftol", "xtol", "max_nfev")
                },
            )
            result = self._curve_fit_hybrid_streaming(
                f=f_engine,
                xdata=x_engine,
                ydata=ydata,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                data_mask=data_mask,
                callback=callback,
                jac=jac,
                solver=solver,
                timeit=timeit,
                return_eval=return_eval,
                full_output=full_output,
                compute_covariance=compute_covariance,
                config=config,
                **kwargs,
            )
            if isinstance(result, CurveFitResult):
                # Report the caller's model and x (scipy layout), not the
                # points-first adapters. Transpose the x the hybrid fit
                # actually used: nan_policy='omit' may have dropped rows.
                result["model"] = f
                result["x_points_first"] = False  # now back in the caller's layout
                x_used = np.asarray(result["xdata"])
                result["xdata"] = x_used.T if x_used.ndim > 1 else x_used
            return result

        if strategy == "chunked":
            if not compute_covariance:
                raise ValueError(
                    "compute_covariance=False is not supported by the "
                    "method='auto' chunked strategy (the GLS combiner needs "
                    "the per-chunk covariances). Use method='trf' to skip it."
                )
            self._warn_unsupported_args(
                "method='auto' (chunked strategy)",
                jac=jac is not None,
                solver=solver != "auto",
                callback=callback is not None,
                compute_diagnostics=bool(compute_diagnostics),
            )
            if _is_auto_p0(p0):
                p0 = _resolve_auto_p0(
                    f, xdata, ydata, sigma, {**kwargs, "data_mask": data_mask}
                )

            # Delegate to LargeDatasetFitter
            from nlsq.streaming.strategy import build_large_strategy

            # Build the chunked engine with the raw memory_limit_gb kwarg,
            # not config.memory_limit_gb
            large_strategy = build_large_strategy(
                "chunked", config, memory_limit_gb=memory_limit_gb
            )

            # Run fit
            t_fit = time.perf_counter()
            result = large_strategy.fit(
                f,
                np.asarray(xdata, float),
                ydata_arr,
                p0,
                bounds=bounds,
                check_finite=check_finite,
                absolute_sigma=absolute_sigma,
                **({"sigma": sigma} if sigma is not None else {}),
                **({"data_mask": data_mask} if data_mask is not None else {}),
                **kwargs,
            )

            # LargeDatasetFitter's own exception handlers catch every
            # failure (including this function's own NotImplementedError
            # guards, if a caller reaches them via a nested chunked fit) and
            # return a "successful-looking" OptimizeResult with success=False
            # rather than raising -- check it explicitly instead of silently
            # treating a failed fit as a normal result.
            if not result.get("success", True) and not result.get(
                "nonconverged", False
            ):
                raise RuntimeError(
                    "method='auto' chunked fit failed: "
                    f"{result.get('message', 'unknown error')}",
                )

            # Convert OptimizeResult to CurveFitResult format
            t_post = time.perf_counter()
            ctime = t_post - t_fit
            pcov = result.get("pcov", np.full((n_params, n_params), np.inf))
            curve_fit_result = CurveFitResult(result)
            # The chunk engine applied nan_policy itself; store the rows it fit.
            x_fit, y_fit, s_fit = _rows_actually_fit(
                xdata, ydata, sigma, {**kwargs, "data_mask": data_mask}
            )
            curve_fit_result["model"] = f
            curve_fit_result["xdata"] = x_fit
            curve_fit_result["ydata"] = y_fit
            curve_fit_result["pcov"] = pcov
            _store_sigma(curve_fit_result, s_fit, absolute_sigma)
            if full_output and not return_eval:
                # infodict['fvec'] as on the standard path (sigma-weighted);
                # evaluated chunk-wise, but the vector itself is O(n) like the
                # standard path's.
                fvec = _residuals_in_chunks(f, x_fit, y_fit, curve_fit_result.popt)
                if s_fit is not None:
                    import scipy.linalg as sla

                    s_arr = np.asarray(s_fit, float)
                    # a covariance matrix is whitened by its Cholesky factor
                    fvec = (
                        sla.solve_triangular(
                            np.linalg.cholesky(s_arr), fvec, lower=True
                        )
                        if s_arr.ndim == 2
                        else fvec / s_arr
                    )
                curve_fit_result["fun"] = fvec
            # return_eval covers every caller row after nan omission, before the
            # data_mask (as on the standard route), not just the fitted rows.
            x_eval = (
                _rows_actually_fit(
                    xdata,
                    ydata,
                    sigma,
                    {**kwargs, "data_mask": data_mask},
                    apply_mask=False,
                )[0]
                if return_eval
                else xdata
            )
            out = _apply_output_flags(
                curve_fit_result,
                f,
                np.asarray(x_eval, float),
                {
                    "return_eval": return_eval,
                    "full_output": full_output,
                    "timeit": timeit,
                    "data_mask": data_mask,
                },
                ctime=ctime,
                post_time=time.perf_counter() - t_post,
            )
            if timeit and not (return_eval or full_output):
                # The timeit tuple carries the engine's own result object, trimmed
                # like the ordinary timeit path (no per-solver timings: all_times {}).
                result.pop("jac", None)
                result.pop("fun", None)
                result.setdefault("all_times", {})
                out = (out[0], out[1], result, out[3], out[4])
            return out

        # Standard curve_fit path (strategy == "standard")
        # Call curve_fit with method=None to use default TRF
        return self.curve_fit(
            f=f,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            method=None,  # Use default TRF
            solver=solver,
            batch_size=batch_size,
            jac=jac,
            data_mask=data_mask,
            timeit=timeit,
            return_eval=return_eval,
            callback=callback,
            compute_diagnostics=compute_diagnostics,
            diagnostics_level=diagnostics_level,
            diagnostics_config=diagnostics_config,
            full_output=full_output,
            compute_covariance=compute_covariance,
            **kwargs,
        )

    def _warn_if_overflow_clipped(
        self, f: Callable, jnp_xdata, popt, data_mask=None
    ) -> None:
        """Make enable_overflow_check's silent clipping visible (see stable_f)."""
        if not self.enable_overflow_check:
            return
        # Evaluate once at popt (outside JIT). Padding rows (EPS coordinates) and
        # masked rows are not data, so f(EPS) of e.g. a/x must not trigger it.
        fabs = jnp.abs(f(jnp_xdata, *popt))
        if data_mask is not None and fabs.shape == np.shape(data_mask):
            fabs = jnp.where(jnp.asarray(data_mask, dtype=bool), fabs, 0.0)
        fmax = float(jnp.max(fabs))
        if fmax > _OVERFLOW_CLIP_BOUND:
            msg = (
                f"enable_overflow_check clipped the model output "
                f"(max|f| = {fmax:.3g} > {_OVERFLOW_CLIP_BOUND:.0e}); the "
                "fit objective was modified and the result may be unreliable. "
                "Rescale ydata or disable enable_overflow_check."
            )
            self.logger.warning(msg)
            warnings.warn(msg, OptimizeWarning, skip_file_prefixes=_NLSQ_DIR)

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
        prepared_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple:
        """Setup and run the optimization.

        Parameters
        ----------
        f : callable
            Model function f(x, *params) -> y.
        p0 : np.ndarray
            Initial parameter guess.
        xdata : np.ndarray
            Independent variable data.
        ydata : np.ndarray
            Dependent variable data.
        data_mask : np.ndarray
            Boolean mask of the data points that enter the fit.
        transform : callable or None
            Sigma transformation passed to ``least_squares``.
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds (lower, upper).
        method : str
            Optimization method passed to ``least_squares``.
        solver : str
            Solver choice, mapped to ``tr_solver`` by ``_select_tr_solver``
            (``"minibatch"`` maps to ``"lsmr"``).
        batch_size : int or None
            Only logged for ``solver="minibatch"``; no batching is done.
        jac : callable or None
            Jacobian function.
        m : int
            Number of data points.
        n : int
            Number of parameters.
        sigma : np.ndarray or None
            Only used to log whether uncertainties were given; the weighting is
            applied through ``transform``.
        timeit : bool
            Passed to ``least_squares`` to request timing.
        callback : callable or None
            Progress callback passed to ``least_squares``.
        kwargs : dict
            Additional keyword arguments for ``least_squares``. ``maxfev`` is
            accepted as an alias of ``max_nfev``.
        prepared_bounds : tuple[np.ndarray, np.ndarray] or None, optional
            Pre-prepared bounds as (lb, ub). If provided, passes to least_squares
            to skip redundant prepare_bounds call. Default is None.

        Returns
        -------
        res : OptimizeResult
            Optimization result
        jnp_xdata : jnp.ndarray
            JAX array of x data
        ctime : float
            Conversion time (if timeit=True)

        Raises
        ------
        ValueError
            If ``kwargs`` contains ``args``.
        OptimizationError
            If the solver ends with a failure status other than ``max_nfev``.
        RuntimeError
            If error recovery is enabled and recovery is unsuccessful.

        Notes
        -----
        ``kwargs`` is modified in place (``maxfev`` becomes ``max_nfev``,
        ``tr_solver`` is set, internal keys are dropped). With recovery
        disabled, an exception from ``least_squares`` is re-raised.
        """
        # Validate kwargs
        if "args" in kwargs:
            raise ValueError("'args' is not a supported keyword argument.")

        if "max_nfev" not in kwargs:
            kwargs["max_nfev"] = kwargs.pop("maxfev", None)

        _drop_internal_routing_kwargs(kwargs)

        # Determine the appropriate solver and configure tr_solver
        tr_solver = self._select_tr_solver(solver, m, n, batch_size)
        if tr_solver is not None and kwargs.get("tr_solver") is None:
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
        st = time.perf_counter()
        jnp_xdata = jnp.asarray(xdata)
        jnp_ydata = jnp.asarray(ydata)
        ctime = time.perf_counter() - st

        jnp_data_mask = jnp.array(data_mask, dtype=bool)

        # Check memory requirements if stability is enabled
        if self.enable_stability:
            # Scalar/broadcast bounds expand to per-parameter arrays first.
            _blb, _bub = prepare_bounds(bounds, n)
            memory_required = self.memory_manager.predict_memory_requirement(
                m,
                n,
                method,
                bounded=bool(np.any(np.isfinite(_blb)) or np.any(np.isfinite(_bub))),
            )
            is_available, msg = self.memory_manager.check_memory_availability(
                memory_required,
            )
            if not is_available:
                self.logger.warning("Memory constraint detected", details=msg)
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

                def stable_f(x, *params) -> jnp.ndarray:
                    result = original_f(x, *params)
                    max_val = jnp.max(jnp.abs(result))
                    result = jnp.where(
                        max_val > _OVERFLOW_CLIP_BOUND,
                        jnp.clip(result, -_OVERFLOW_CLIP_BOUND, _OVERFLOW_CLIP_BOUND),
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
                    prepared_bounds=prepared_bounds,
                    **kwargs,
                )
            except Exception as e:
                _raise_clear_jac_trace_error(e, jac)
                if self.enable_recovery:
                    self.logger.warning(
                        "Optimization failed, attempting recovery",
                        error=str(e),
                    )
                    recovery_state = {
                        "params": p0,
                        "p0": p0,
                        "xdata": xdata,
                        "ydata": ydata,
                        "method": method if method is not None else "trf",
                        "bounds": bounds,
                    }

                    # ftol/xtol/tr_solver/x_scale/loss are forwarded explicitly
                    # below (so recovery strategies can override them) - drop
                    # them from the **kwargs spread or least_squares() raises
                    # "got multiple values for keyword argument" whenever the
                    # outer kwargs already set one (e.g. tr_solver is set
                    # unconditionally above whenever _select_tr_solver
                    # resolves a value).
                    _recovery_passthrough_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if k not in ("ftol", "xtol", "tr_solver", "x_scale", "loss")
                    }
                    # Classify the failure so recovery strategies gated on
                    # failure_type (e.g. OptimizationRecovery._adjust_regularization,
                    # which only boosts regularization / switches to LSMR for
                    # "numerical"/"ill_conditioned") actually activate instead
                    # of silently no-op'ing on a generic "optimization_error".
                    import re as _re

                    _err_msg = str(e).lower()
                    _numerical_failure = (
                        isinstance(e, np.linalg.LinAlgError)
                        # "nan"/"inf" as whole words only: a bare substring
                        # match would misclassify ordinary messages like
                        # "insufficient information" or "infeasible" as a
                        # numerical failure.
                        or bool(_re.search(r"\b(nan|inf)\b", _err_msg))
                        or any(
                            kw in _err_msg
                            for kw in (
                                "singular",
                                "ill-conditioned",
                                "ill conditioned",
                                "condition number",
                                "cholesky",
                                "svd",
                            )
                        )
                    )
                    failure_type = (
                        "numerical" if _numerical_failure else "optimization_error"
                    )
                    success, result = self.recovery.recover_from_failure(
                        failure_type,
                        recovery_state,
                        lambda **state: self.ls.least_squares(
                            f_to_use,
                            # recovery strategies (e.g. _multi_start) update "p0",
                            # not "params" - fall back to "params" when absent.
                            state.get("p0", state["params"]),
                            jac=jac,
                            xdata=jnp.asarray(state["xdata"]),
                            ydata=jnp.asarray(state["ydata"]),
                            data_mask=jnp_data_mask,
                            transform=transform,
                            bounds=state["bounds"],
                            method=state["method"],
                            ftol=state.get("ftol", kwargs.get("ftol", DEFAULT_FTOL)),
                            xtol=state.get("xtol", kwargs.get("xtol", DEFAULT_XTOL)),
                            tr_solver=state.get("tr_solver", kwargs.get("tr_solver")),
                            x_scale=state.get("x_scale", kwargs.get("x_scale", 1.0)),
                            loss=state.get("loss", kwargs.get("loss", "linear")),
                            timeit=timeit,
                            callback=callback,
                            **_recovery_passthrough_kwargs,
                        ),
                    )

                    if success:
                        res = result
                    else:
                        raise RuntimeError(
                            f"Optimization failed and recovery unsuccessful: {e}",
                        ) from e
                else:
                    raise

        if not res.success:
            if res.status == 0:
                # Maximum iterations reached: warn and return a best-effort
                # result (success=False). SciPy's curve_fit raises RuntimeError
                # here; NLSQ deliberately warns instead (see curve_fit Notes).
                max_nfev = kwargs.get("max_nfev") or len(p0) * 100
                import warnings

                warnings.warn(
                    f"Optimal parameters not found: maximum number of function "
                    f"evaluations ({max_nfev}) exceeded.",
                    OptimizeWarning,
                    stacklevel=3,
                )
            else:
                self.logger.error(
                    "Optimization failed",
                    reason=res.message,
                    status=res.status,
                )
                gtol = kwargs.get("gtol", 1e-8)
                ftol = kwargs.get("ftol", 1e-8)
                xtol = kwargs.get("xtol", 1e-8)
                max_nfev = kwargs.get("max_nfev") or len(p0) * 100
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
        full_output: bool = False,
        callback: Callable | None = None,
        compute_diagnostics: bool = False,
        diagnostics_level: DiagnosticLevel = DiagnosticLevel.BASIC,
        diagnostics_config: DiagnosticsConfig | None = None,
        compute_covariance: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            Initial guess for the parameters (length N). If 'auto', initial
            parameters are estimated from the data characteristics (for best
            results use well-scaled data or provide custom estimation via
            f.estimate_p0()). If None (the SciPy default), every parameter starts
            at 1, or at a feasible point of its finite bounds (box midpoint,
            lower bound + 1, upper bound - 1); the data are not inspected.
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
        compute_covariance : bool, optional
            When False, skip the final covariance and return ``pcov=None``
            in every return shape. See :func:`nlsq.curve_fit`.
        method : str | None, optional
            Optimization algorithm. Options:

            - 'trf' (default): Trust Region Reflective
            - 'lm': accepted for SciPy compatibility (unbounded problems only);
              currently runs the same trust-region solver as 'trf'
            - 'hybrid_streaming': Adaptive Hybrid Streaming Optimizer (for large datasets)
            - 'auto': Memory-based automatic selection (streaming/chunked/standard)

            If None, auto-selects 'trf'.
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
        jac : callable or None, optional
            Function with signature ``jac(x, ...)`` which computes the Jacobian
            matrix of the model function with respect to parameters as a dense
            array_like structure. It will be scaled according to provided `sigma`.
            If None (default), the Jacobian will be determined using JAX's automatic
            differentiation (AD) capabilities. We recommend not using an analytical
            Jacobian, as it is usually faster to use AD. A user ``jac`` is
            traced by JAX, so it must be written with ``jax.numpy`` (a
            NumPy-only callable raises ``TypeError``).
            The returned array must have shape ``(n_params, m)`` (the transpose
            of SciPy's ``(m, n_params)``). SciPy-style ``(m, n_params)`` output is
            also accepted when ``m != n_params``; when ``m == n_params`` the
            shape is ambiguous and is read as ``(n_params, m)``. Any other shape
            raises ``ValueError``.
        callback : callable or None, optional
            Callback function called after each optimization iteration with signature
            ``callback(iteration, cost, params, info)``. Useful for monitoring
            optimization progress, logging, or implementing custom stopping criteria.
            If None (default), no callback is invoked. See ``nlsq.callbacks`` module
            for built-in callbacks (ProgressBar, IterationLogger, EarlyStopping).
            .. versionadded:: 0.2.0
        data_mask : array_like of bool, optional
            Mask of length ``len(ydata)``; rows where it is False are excluded from
            the fit. Default is None (all rows used).
        timeit : bool, optional
            Return ``(popt, pcov, res, post_time, ctime)`` instead of a
            ``CurveFitResult``; the per-evaluation residual and Jacobian times are
            in ``res.all_times`` (unbounded fits only; bounded fits give ``{}``).
            Adds a device sync per evaluation. Default is False.
        return_eval : bool, optional
            Return ``(popt, pcov, f(xdata, *popt))`` with one model value per input
            row (padding removed; rows excluded by ``data_mask`` are still
            evaluated). Default is False.
        full_output : bool, optional
            Return the SciPy-style ``(popt, pcov, infodict, mesg, ier)`` tuple.
            Default is False.
        compute_diagnostics : bool, optional
            If True, compute a fit-quality report and attach it to the result as
            ``result.diagnostics``. Default is False.
        diagnostics_level : DiagnosticLevel, optional
            Level of diagnostics to compute. A non-default level overrides
            ``diagnostics_config.level``. Default is ``DiagnosticLevel.BASIC``.
        diagnostics_config : DiagnosticsConfig or None, optional
            Diagnostics configuration. Registered diagnostic plugins run
            automatically and appear in ``result.diagnostics.plugin_results``.
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
            If the Jacobian matrix at the solution doesn't have a full rank, the
            Moore-Penrose pseudoinverse is used to compute the covariance matrix
            (``'lm'`` is an alias of ``'trf'`` here).

        Raises
        ------
        ValueError
            if either `ydata` or `xdata` contain NaNs, or if incompatible options
            are used.
        OptimizationError
            (a ``RuntimeError`` subclass) if the least-squares minimization
            fails. Exhausting ``max_nfev`` only emits an ``OptimizeWarning``.
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
        >>> from nlsq import CurveFit
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
        absolute_sigma = coerce_absolute_sigma(absolute_sigma)

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
                data_mask=data_mask,
                callback=callback,
                jac=jac,
                solver=solver,
                timeit=timeit,
                return_eval=return_eval,
                full_output=full_output,
                compute_covariance=compute_covariance,
                compute_diagnostics=compute_diagnostics,
                **kwargs,
            )

        # Memory-based automatic method selection (method="auto")
        if method == "auto":
            return self._curve_fit_auto_memory(
                f=f,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                solver=solver,
                batch_size=batch_size,
                jac=jac,
                data_mask=data_mask,
                timeit=timeit,
                return_eval=return_eval,
                callback=callback,
                compute_diagnostics=compute_diagnostics,
                diagnostics_level=diagnostics_level,
                diagnostics_config=diagnostics_config,
                full_output=full_output,
                compute_covariance=compute_covariance,
                **kwargs,
            )

        # scipy.optimize.Bounds -> (lb, ub): the stability-time algorithm
        # selection unpacks the tuple form (the wrapper normalizes too).
        bounds = _bounds_as_tuple(bounds)

        # Prepare and validate all inputs
        prepared = prepare_problem(
            f,
            xdata,
            ydata,
            p0,
            bounds,
            sigma,
            data_mask,
            check_finite,
            kwargs.get("nan_policy", "raise"),
            method,
            solver,
            batch_size,
            kwargs,
            PrepareContext(
                flength=self.flength,
                use_dynamic_sizing=self.use_dynamic_sizing,
                enable_stability=self.enable_stability,
                validator=(
                    InputValidator(fast_mode=True)
                    if getattr(self, "validator", None) is None
                    else self.validator
                ),
                logger=self.logger,
            ),
        )
        n, p0, xdata, ydata = prepared.n, prepared.p0, prepared.x, prepared.y
        data_mask, method, m = prepared.mask, prepared.method, prepared.m_real
        len_diff, none_mask, sigma = (
            prepared.len_diff,
            prepared.none_mask,
            prepared.sigma,
        )
        _lb, _ub = prepared.lb, prepared.ub

        # Setup sigma transformation (sigma is already filtered consistently
        # with xdata/ydata above when nan_policy='omit' dropped rows)
        transform = self._setup_sigma_transform(sigma, ydata, data_mask, len_diff, m)

        # If gradient-health diagnostics were requested, wire a GradientMonitor
        # into the optimizer's iteration callback so a real report exists by
        # the time the diagnostics block below reads it back out. Without this,
        # `compute_diagnostics=True` silently leaves gradient_health as None.
        _gradient_monitor = None
        _resolved_diag_config = None
        if compute_diagnostics:
            _resolved_diag_config = _resolve_diagnostics_config(
                diagnostics_level, diagnostics_config
            )
            from nlsq.diagnostics.gradient_health import GradientMonitor

            _gradient_monitor = GradientMonitor(_resolved_diag_config)
            callback = _gradient_monitor.create_callback(user_callback=callback)

        # Run optimization (pass prepared bounds to avoid redundant prepare_bounds call)
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
            prepared_bounds=(_lb, _ub),
        )

        if _gradient_monitor is not None:
            res["gradient_health_report"] = _gradient_monitor.get_report()

        popt = res.x
        self._warn_if_overflow_clipped(f, jnp_xdata, popt, data_mask)
        self.logger.debug(
            "Optimization succeeded",
            final_cost=res.cost,
            nfev=res.nfev,
            optimality=getattr(res, "optimality", None),
        )

        st = time.perf_counter()
        # Degrees-of-freedom/SVD-threshold must reflect the EFFECTIVE point
        # count, prepared.n_effective (unmasked rows), not the raw m.
        # sigma=inf rows have zero weight: they are not observations for the dof
        ysize = prepared.n_effective - zero_weight_rows(prepared.sigma, prepared.mask)
        cost = 2 * res.cost  # res.cost is half sum of squares!

        # Compute covariance matrix
        pcov, warn_cov = self._final_covariance(
            res, ysize, p0, absolute_sigma, compute_covariance
        )
        _pcov = pcov

        return_full = full_output

        post_time = time.perf_counter() - st

        # Log curve fit completion
        total_time = self.logger.timers.get("curve_fit", 0)
        self.logger.info(
            "Curve fit completed",
            total_time=total_time,
            final_cost=cost,
            covariance_warning=warn_cov,
        )

        if return_eval:
            _record_attempt(res)
            feval = f(jnp_xdata, *popt)
            # the optimizer broadcasts a scalar model output over the observations
            feval = np.array(np.broadcast_to(np.asarray(feval), np.shape(ydata)))
            if none_mask:
                return np.asarray(popt), _pcov, feval[data_mask]
            # A user data_mask leaves padding rows (flength) in feval; strip them.
            return np.asarray(popt), _pcov, feval[:m]

        xdata, ydata = self._drop_unobserved_rows(res, xdata, ydata, data_mask)
        sigma = self._drop_unobserved_sigma(sigma, data_mask)

        if return_full:
            # SciPy-compatible 5-tuple: (popt, pcov, infodict, mesg, ier)
            # fjac/ipvt/qtf are LM-only and not available for TRF/dogbox.
            infodict = {
                "nfev": res.nfev,
                "fvec": np.asarray(res.fun) if res.get("fun") is not None else None,
            }
            mesg = res.message if hasattr(res, "message") else ""
            # SciPy ier convention: 1-4 = converged, 5 = max iterations
            ier = int(res.status) if (res.success and res.status > 0) else 5
            return np.asarray(popt), _pcov, infodict, mesg, ier
        if timeit:
            # lower GPU memory usage before returning raw res
            res.pop("jac", None)
            res.pop("fun", None)
            return np.asarray(popt), _pcov, res, post_time, ctime
        # Create enhanced result object that supports tuple unpacking
        # for backward compatibility: popt, pcov = curve_fit(...)
        # Keep 'fun' (residuals) and 'jac' for statistical computations
        result = CurveFitResult(res)
        result["model"] = f
        result["xdata"] = xdata
        result["ydata"] = ydata
        result["pcov"] = _pcov
        _store_sigma(result, sigma, absolute_sigma)

        # Compute diagnostics if requested
        if compute_diagnostics:
            # Fetched outside the try block below: this already succeeded
            # (or was never computed) before identifiability/sensitivity
            # analysis runs, so a failure in either of those must not cause
            # an already-computed gradient-health report to be discarded.
            gradient_health_report = res.get("gradient_health_report")
            try:
                from nlsq.diagnostics.health_report import create_health_report
                from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer

                # Reuse the config resolved before optimization (needed there
                # to construct the GradientMonitor) instead of rebuilding it.
                config = _resolved_diag_config

                # res.jac is already unpadded (padded/masked rows are not part of
                # it), so no slicing is needed before the analyzers.
                jacobian = np.asarray(res.jac)

                # Run identifiability analysis
                analyzer = IdentifiabilityAnalyzer(config=config)
                ident_report = analyzer.analyze(jacobian)

                # Run parameter sensitivity analysis if diagnostics_level is FULL (User Story 4)
                sloppy_model_report = None
                if config.level == DiagnosticLevel.FULL:
                    from nlsq.diagnostics import ParameterSensitivityAnalyzer

                    sensitivity_analyzer = ParameterSensitivityAnalyzer(
                        config=config,
                    )
                    sloppy_model_report = sensitivity_analyzer.analyze(jacobian)

                # Registered diagnostic plugins (isolated per plugin by run_plugins;
                # a failure here must not discard the built-in reports).
                plugin_results = _run_diagnostic_plugins(
                    self.logger,
                    jacobian,
                    popt,
                    res.fun,
                    xdata=xdata,
                    ydata=ydata,
                    bounds=bounds,
                    model=f,
                    config=config,
                )

                # Create aggregated health report using factory function
                # Note: verbose and emit_warnings are handled by create_health_report
                health_report = create_health_report(
                    identifiability=ident_report,
                    gradient_health=gradient_health_report,
                    sloppy_model=sloppy_model_report,
                    plugin_results=plugin_results,
                    config=config,
                )

                # Attach to result
                result["_diagnostics_report"] = health_report

                # Log diagnostics summary if user requested verbose
                if diagnostics_config and diagnostics_config.verbose:
                    self.logger.info(
                        "Diagnostics computed",
                        health_status=health_report.status.name,
                        health_score=f"{health_report.health_score:.2f}",
                        n_issues=len(health_report.all_issues),
                    )

            except Exception as e:
                self.logger.warning(
                    "Failed to compute diagnostics",
                    error=str(e),
                    exc_info=True,
                )
                # Create unavailable diagnostics report
                from nlsq.diagnostics.health_report import create_health_report
                from nlsq.diagnostics.types import (
                    HealthStatus,
                    IdentifiabilityReport,
                )

                unavailable_ident = IdentifiabilityReport(
                    available=False,
                    error_message=str(e),
                    n_params=n,
                    health_status=HealthStatus.CRITICAL,
                )
                # Preserve gradient_health_report fetched above -- a failure
                # in identifiability/sensitivity analysis shouldn't discard
                # an already-successfully-computed gradient-health result.
                # Respect the caller's own emit_warnings setting instead of
                # hardcoding False, so a caller who asked for warnings still
                # gets them on this degraded path.
                fallback_config = DiagnosticsConfig(
                    verbose=False,
                    emit_warnings=(
                        _resolved_diag_config.emit_warnings
                        if _resolved_diag_config is not None
                        else True
                    ),
                )
                health_report = create_health_report(
                    identifiability=unavailable_ident,
                    gradient_health=gradient_health_report,
                    config=fallback_config,
                )
                result["_diagnostics_report"] = health_report

        # Add cache statistics if available
        try:
            cache_stats = self.cache.get_stats()
            if cache_stats.get("enabled", True):
                # Extract relevant cache metrics
                result["cache_stats"] = {
                    "hit": cache_stats.get("hits", 0) > cache_stats.get("misses", 0),
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
